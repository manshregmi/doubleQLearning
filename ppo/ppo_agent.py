import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_NODES = 3
N_MODES = 3
N_LAYERS = 7
STATE_DIM = 15

NEG = -1e9   # finite mask value; must NOT be -inf (breaks log_softmax)


def encode_state(state, deadline_levels):
    """Encode the 10-tuple state as a fixed-size float vector."""
    (bw, rtt, ctime, layer, prev_action, surplus,
     neg_count, chain_len, d_idx, recent_ack_loss) = state

    x = np.zeros(STATE_DIM, dtype=np.float32)
    x[0] = float(bw) / 15.0
    x[1] = (float(rtt) - 1.0) / 5.0
    x[2] = float(ctime) / 500.0
    x[3] = float(layer) / 7.0

    if prev_action is not None:
        for i, m in enumerate(prev_action[:MAX_NODES]):
            x[4 + i] = float(int(m)) / 2.0
        for i in range(len(prev_action), MAX_NODES):
            x[4 + i] = -1.0
    else:
        x[4:7] = -1.0

    deadline = float(deadline_levels[int(d_idx)])
    x[7] = float(surplus) / max(deadline, 1e-9)
    x[8] = min(float(neg_count), 5.0) / 5.0
    x[9] = float(chain_len) / 7.0
    x[10] = float(d_idx) / max(1, len(deadline_levels) - 1)
    x[11] = float(recent_ack_loss) / 3.0
    return x


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_layers=N_LAYERS,
                 max_nodes=MAX_NODES, n_modes=N_MODES, hidden=128):
        super().__init__()
        self.layer_emb = nn.Embedding(n_layers, 8)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + 8, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, max_nodes * n_modes)
        self.critic = nn.Linear(hidden, 1)
        self.max_nodes = max_nodes
        self.n_modes = n_modes

    def forward(self, x, layer_idx):
        h = self.trunk(torch.cat([x, self.layer_emb(layer_idx)], dim=-1))
        logits = self.actor(h).view(-1, self.max_nodes, self.n_modes)
        value = self.critic(h).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(self, deadline_levels, is_test=False,
                 lr=3e-4, gamma=0.95, gae_lambda=0.95, clip=0.2,
                 entropy_coef=0.01, value_coef=0.5, n_epochs=4,
                 minibatch=64, max_grad_norm=0.5,
                 rollout_target_steps=2048, reward_scale=0.01,
                 device=None):
        self.deadline_levels = list(deadline_levels)
        self.is_test = is_test
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.minibatch = minibatch
        self.max_grad_norm = max_grad_norm
        self.rollout_target_steps = rollout_target_steps
        self.reward_scale = reward_scale

        self.device = device or torch.device("cpu")
        self.net = ActorCriticNet().to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.buffer = []
        self.eval_decisions = 0
        self.eval_unseen_states = 0

    # ------------------------------------------------------------------
    def _policy_dist(self, state):
        x = torch.tensor(
            encode_state(state, self.deadline_levels),
            dtype=torch.float32, device=self.device).unsqueeze(0)
        layer = int(state[3])
        layer_idx = torch.tensor([layer], dtype=torch.long, device=self.device)
        logits, value = self.net(x, layer_idx)
        return logits[0], float(value.item())

    def _mask_logits(self, logits, n_real_nodes):
        """logits: (MAX_NODES, N_MODES). Use finite NEG for padded rows."""
        masked = torch.full_like(logits, NEG)
        masked[:n_real_nodes] = logits[:n_real_nodes]
        return masked

    # ------------------------------------------------------------------
    def choose_action(self, state, n_real_nodes):
        logits, value = self._policy_dist(state)

        # sanitize the raw logits before softmax
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = logits.clamp(-20.0, 20.0)

        masked = self._mask_logits(logits, n_real_nodes)
        real = masked[:n_real_nodes]                    # (n_real, N_MODES)

        log_probs = F.log_softmax(real, dim=-1)

        if self.is_test:
            modes = torch.argmax(real, dim=-1).cpu().numpy()
        else:
            probs = log_probs.exp()
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs.clamp_min(0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            modes = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()

        modes_t = torch.tensor(modes, dtype=torch.long, device=self.device)
        chosen_logp = float(
            log_probs.gather(1, modes_t.unsqueeze(-1)).squeeze(-1).sum().item()
        )
        return modes, chosen_logp, value

    # ------------------------------------------------------------------
    def store_episode(self, episode):
        self.buffer.append(episode)

    def buffer_steps(self):
        return sum(len(ep) for ep in self.buffer)

    def should_update(self):
        return self.buffer_steps() >= self.rollout_target_steps

    # ------------------------------------------------------------------
    def update(self):
        if not self.buffer:
            return {}

        states, layers, modes_list, logps, rewards, dones, values = ([] for _ in range(7))
        episode_lengths = []
        for ep in self.buffer:
            episode_lengths.append(len(ep))
            for step in ep:
                states.append(step["state"])
                layers.append(int(step["state"][3]))
                modes_list.append(step["modes"])
                logps.append(step["logp"])
                rewards.append(step["reward_scaled"])
                dones.append(step["done"])
                values.append(step["value"])
        self.buffer = []

        # ---- GAE per episode ----
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        idx = 0
        for L in episode_lengths:
            ep_slice = slice(idx, idx + L)
            ep_r = np.asarray(rewards[ep_slice], dtype=np.float32)
            ep_v = np.asarray(values[ep_slice], dtype=np.float32)
            ep_d = np.asarray(dones[ep_slice], dtype=np.float32)

            last_gae = 0.0
            ep_adv = np.zeros(L, dtype=np.float32)
            for t in reversed(range(L)):
                next_v = ep_v[t + 1] if t + 1 < L else 0.0
                nonterm = 1.0 - ep_d[t]
                delta = ep_r[t] + self.gamma * next_v * nonterm - ep_v[t]
                last_gae = delta + self.gamma * self.gae_lambda * nonterm * last_gae
                ep_adv[t] = last_gae
            advantages[ep_slice] = ep_adv
            returns[ep_slice] = ep_adv + ep_v
            idx += L

        x_all = np.stack([encode_state(s, self.deadline_levels) for s in states])
        layer_all = np.asarray(layers, dtype=np.int64)

        mode_arr = np.zeros((len(modes_list), MAX_NODES), dtype=np.int64)
        for i, m in enumerate(modes_list):
            mode_arr[i, :len(m)] = m

        logp_old = np.asarray(logps, dtype=np.float32)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv = np.clip(adv, -10.0, 10.0)
        ret = returns.astype(np.float32)
        ret = np.clip(ret, -1e3, 1e3)

        x_all = torch.tensor(x_all, device=self.device)
        layer_all = torch.tensor(layer_all, device=self.device)
        mode_all = torch.tensor(mode_arr, device=self.device)
        logp_old = torch.tensor(logp_old, device=self.device)
        adv = torch.tensor(adv, device=self.device)
        ret = torch.tensor(ret, device=self.device)

        n = x_all.shape[0]
        info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch):
                mb = perm[start:start + self.minibatch]
                xb = x_all[mb]
                lb = layer_all[mb]
                mb_modes = mode_all[mb]
                mb_logp_old = logp_old[mb]
                mb_adv = adv[mb]
                mb_ret = ret[mb]

                logits, value = self.net(xb, lb)   # (B, MAX_NODES, N_MODES)

                # ---- sanitize logits ----
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                logits = logits.clamp(-20.0, 20.0)

                # ---- per-row node mask (finite NEG, never -inf) ----
                node_counts = torch.tensor(
                    [_nodes_in_layer(int(l.item())) for l in lb],
                    device=self.device, dtype=torch.long)
                node_mask = (
                    torch.arange(MAX_NODES, device=self.device).unsqueeze(0)
                    < node_counts.unsqueeze(1)
                ).float()                                    # (B, MAX_NODES)

                mask = torch.where(
                    node_mask.unsqueeze(-1).bool(),
                    logits,
                    torch.full_like(logits, NEG),
                )

                log_probs = F.log_softmax(mask, dim=-1)      # (B, MAX_NODES, N_MODES)

                gather = log_probs.gather(
                    2, mb_modes.unsqueeze(-1)).squeeze(-1)   # (B, MAX_NODES)
                new_logp = (gather * node_mask).sum(dim=1)   # (B,)

                # ---- PPO clipped surrogate ----
                ratio = torch.exp(new_logp - mb_logp_old)
                ratio = torch.clamp(ratio, 0.0, 10.0)        # guard
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, mb_ret)
                value_loss = torch.clamp(value_loss, max=1e3)   # guard

                # ---- entropy over real nodes only ----
                probs = log_probs.exp()
                logp_node = torch.log(probs.clamp_min(1e-9))
                ent_per_node = -(probs * logp_node).sum(dim=-1)  # (B, MAX_NODES)
                ent = (ent_per_node * node_mask).sum(dim=1).mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * ent)

                if not torch.isfinite(loss):
                    self.optim.zero_grad()
                    continue

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()

                info["policy_loss"] += float(policy_loss.item())
                info["value_loss"] += float(value_loss.item())
                info["entropy"] += float(ent.item())
                n_updates += 1

        if n_updates:
            for k in info:
                info[k] /= n_updates
        return info

    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.net.load_state_dict(state)


# ---- helpers ----
def _nodes_in_layer(layer_idx):
    widths = [1, 1, 1, 3, 3, 3, 1]
    return widths[max(0, min(int(layer_idx), len(widths) - 1))]