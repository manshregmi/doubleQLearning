import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_NODES = 3
N_MODES = 3
N_LAYERS = 7
STATE_DIM = 15


def encode_state(state, deadline_levels):
    """Same encoding as the PPO agent."""
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


def _nodes_in_layer(layer_idx):
    widths = [1, 1, 1, 3, 3, 3, 1]
    return widths[max(0, min(int(layer_idx), len(widths) - 1))]


class PolicyNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_layers=N_LAYERS,
                 max_nodes=MAX_NODES, n_modes=N_MODES, hidden=128):
        super().__init__()
        self.layer_emb = nn.Embedding(n_layers, 8)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + 8, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, max_nodes * n_modes)
        self.max_nodes = max_nodes
        self.n_modes = n_modes

    def forward(self, x, layer_idx):
        h = self.trunk(torch.cat([x, self.layer_emb(layer_idx)], dim=-1))
        return self.head(h).view(-1, self.max_nodes, self.n_modes)


class QNet(nn.Module):
    """Q(s, a) where a is the per-node mode vector (padded)."""
    def __init__(self, state_dim=STATE_DIM, n_layers=N_LAYERS,
                 max_nodes=MAX_NODES, n_modes=N_MODES, hidden=128):
        super().__init__()
        self.layer_emb = nn.Embedding(n_layers, 8)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + 8 + max_nodes * n_modes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.max_nodes = max_nodes
        self.n_modes = n_modes

    def forward(self, x, layer_idx, a_onehot):
        # a_onehot: (B, MAX_NODES * N_MODES)
        h = torch.cat([x, self.layer_emb(layer_idx), a_onehot], dim=-1)
        return self.trunk(h).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, layer, modes, reward, s_next, layer_next, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, layer, modes, reward, s_next, layer_next, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, layer, modes, r, s2, layer2, d = zip(*batch)
        return s, layer, modes, r, s2, layer2, d

    def __len__(self):
        return len(self.buffer)


class SACDiscreteAgent:
    """SAC for discrete per-node categorical actions.

    Per-node independent categorical policy over {0,1,2}. Twin Q heads,
    target network, automatic entropy temperature. Terminal-only Eq. 11
    penalty in the buffer (standard MDP form).
    """

    def __init__(self, deadline_levels, is_test=False,
                 lr=3e-4, gamma=0.95, tau=0.005,
                 alpha_lr=3e-4, buffer_capacity=200_000,
                 batch_size=128, warmup=5_000, updates_per_step=1,
                 reward_scale=0.01, device=None):
        self.deadline_levels = list(deadline_levels)
        self.is_test = is_test
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup = warmup
        self.updates_per_step = updates_per_step
        self.reward_scale = reward_scale

        self.device = device or torch.device("cpu")
        self.policy = PolicyNet().to(self.device)
        self.q1 = QNet().to(self.device)
        self.q2 = QNet().to(self.device)
        self.q1_target = QNet().to(self.device)
        self.q2_target = QNet().to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr)

        # Automatic entropy temperature.
        self.target_entropy = -0.98 * np.log(1.0 / N_MODES) * MAX_NODES
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.train_steps = 0

        self.eval_decisions = 0
        self.eval_unseen_states = 0

    # ------------------------------------------------------------------
    @property
    def alpha(self):
        return float(self.log_alpha.exp().item())

    def _logits(self, state):
        x = torch.tensor(
            encode_state(state, self.deadline_levels),
            dtype=torch.float32, device=self.device).unsqueeze(0)
        layer = int(state[3])
        layer_t = torch.tensor([layer], dtype=torch.long, device=self.device)
        return self.policy(x, layer_t)[0]  # (MAX_NODES, N_MODES)

    def _mask_logits(self, logits, n_real_nodes):
        mask = torch.full_like(logits, float("-inf"))
        mask[:n_real_nodes] = logits[:n_real_nodes]
        return mask

    def choose_action(self, state, n_real_nodes):
        logits = self._logits(state)
        masked = self._mask_logits(logits, n_real_nodes)
        probs = F.softmax(masked[:n_real_nodes], dim=-1).detach().cpu().numpy()

        if self.is_test:
            self.eval_decisions += 1
            modes = np.argmax(probs, axis=-1)
            return modes
        # Sample per node.
        modes = np.array([np.random.choice(N_MODES, p=probs[i])
                          for i in range(n_real_nodes)], dtype=np.int64)
        return modes

    # ------------------------------------------------------------------
    def store(self, s, modes, reward_scaled, s_next, done):
        layer = int(s[3])
        layer_next = int(s_next[3])
        self.buffer.push(s, layer, modes, reward_scaled, s_next, layer_next, done)

    def ready_to_update(self):
        return len(self.buffer) >= max(self.warmup, self.batch_size)

    # ------------------------------------------------------------------
    def update(self):
        info = {}
        for _ in range(self.updates_per_step):
            if not self.ready_to_update():
                return info
            s, layer, modes, r, s2, layer2, d = self.buffer.sample(self.batch_size)

            x = torch.tensor(
                np.stack([encode_state(si, self.deadline_levels) for si in s]),
                dtype=torch.float32, device=self.device)
            x2 = torch.tensor(
                np.stack([encode_state(si, self.deadline_levels) for si in s2]),
                dtype=torch.float32, device=self.device)
            l_t = torch.tensor(layer, dtype=torch.long, device=self.device)
            l2_t = torch.tensor(layer2, dtype=torch.long, device=self.device)
            r_t = torch.tensor(r, dtype=torch.float32, device=self.device)
            d_t = torch.tensor(d, dtype=torch.float32, device=self.device)

            # One-hot per-node mode vector, padded to MAX_NODES.
            a_oh = torch.zeros(self.batch_size, MAX_NODES, N_MODES, device=self.device)
            for i, m in enumerate(modes):
                for j, mode in enumerate(m):
                    a_oh[i, j, int(mode)] = 1.0
            a_flat = a_oh.view(self.batch_size, -1)

            # ---- Q targets ----
            with torch.no_grad():
                logits_next = self.policy(x2, l2_t)              # (B, MAX_NODES, N_MODES)
                # Mask padded nodes by node count from next state layer.
                nxt_nodes = torch.tensor(
                    [_nodes_in_layer(int(l.item())) for l in l2_t],
                    device=self.device, dtype=torch.long)
                mask_next = torch.full_like(logits_next, float("-inf"))
                for i in range(self.batch_size):
                    mask_next[i, :int(nxt_nodes[i].item())] = logits_next[i, :int(nxt_nodes[i].item())]
                probs_next = F.softmax(mask_next, dim=-1)
                log_probs_next = F.log_softmax(mask_next, dim=-1)

                # Q target needs a flat one-hot; take expectation over each node's
                # categorical independently by averaging Q over sampled mode vectors
                # is expensive; instead, use the soft expectation approximation:
                # Q_target = E_{a ~ pi}[ min(Q1, Q2)(s', a) ] computed as
                # sum over joint a of prod pi_j(a_j) * Q(s', a).
                # MAX_NODES=3, N_MODES=3 => 27 joint actions. Enumerate.
                joint_actions = torch.tensor(
                    list(np.ndindex(*([N_MODES] * MAX_NODES))),
                    device=self.device, dtype=torch.long)       # (27, MAX_NODES)
                joint_probs = torch.ones(27, device=self.device)
                for j in range(MAX_NODES):
                    joint_probs = joint_probs * probs_next[:, j, joint_actions[:, j]]
                # (B, 27)
                joint_probs = joint_probs.T if joint_probs.dim() == 1 else joint_probs

                q_min_next = torch.zeros(self.batch_size, device=self.device)
                # Build one-hot for all 27 actions once.
                a_all_oh = torch.zeros(27, MAX_NODES, N_MODES, device=self.device)
                for i in range(27):
                    for j in range(MAX_NODES):
                        a_all_oh[i, j, joint_actions[i, j]] = 1.0
                a_all_flat = a_all_oh.view(27, -1)              # (27, MAX*N_MODES)

                # For each batch element, compute Q(s', a) for all 27 a.
                for b in range(self.batch_size):
                    xb = x2[b:b+1].expand(27, -1)
                    lb = l2_t[b:b+1].expand(27)
                    q1n = self.q1_target(xb, lb, a_all_flat)
                    q2n = self.q2_target(xb, lb, a_all_flat)
                    qmin = torch.min(q1n, q2n)                  # (27,)
                    p_b = joint_probs[b]                        # (27,)
                    q_min_next[b] = (p_b * qmin).sum()

                target_q = r_t + (1.0 - d_t) * self.gamma * q_min_next

            # ---- Q losses ----
            q1 = self.q1(x, l_t, a_flat)
            q2 = self.q2(x, l_t, a_flat)
            q1_loss = F.mse_loss(q1, target_q)
            q2_loss = F.mse_loss(q2, target_q)

            self.q1_optim.zero_grad()
            q1_loss.backward()
            self.q1_optim.step()

            self.q2_optim.zero_grad()
            q2_loss.backward()
            self.q2_optim.step()

            # ---- Policy loss ----
            logits = self.policy(x, l_t)
            real_nodes = torch.tensor(
                [_nodes_in_layer(int(l.item())) for l in l_t],
                device=self.device, dtype=torch.long)
            mask = torch.full_like(logits, float("-inf"))
            for i in range(self.batch_size):
                mask[i, :int(real_nodes[i].item())] = logits[i, :int(real_nodes[i].item())]
            probs = F.softmax(mask, dim=-1)
            log_probs = F.log_softmax(mask, dim=-1)

            # Expected Q over joint actions.
            joint_actions = torch.tensor(
                list(np.ndindex(*([N_MODES] * MAX_NODES))),
                device=self.device, dtype=torch.long)
            a_all_oh = torch.zeros(27, MAX_NODES, N_MODES, device=self.device)
            for i in range(27):
                for j in range(MAX_NODES):
                    a_all_oh[i, j, joint_actions[i, j]] = 1.0
            a_all_flat = a_all_oh.view(27, -1)

            # Entropy term: sum_j H(pi_j).
            entropy = -(probs * log_probs.clamp_min(-1e9)).sum(-1)   # (B, MAX_NODES)
            entropy = entropy.sum(-1)                                # (B,)

            expected_q = torch.zeros(self.batch_size, device=self.device)
            for b in range(self.batch_size):
                xb = x[b:b+1].expand(27, -1)
                lb = l_t[b:b+1].expand(27)
                q1p = self.q1(xb, lb, a_all_flat)
                q2p = self.q2(xb, lb, a_all_flat)
                qmin = torch.min(q1p, q2p)
                # joint probability of each of the 27 actions under current pi
                pj = torch.ones(27, device=self.device)
                for j in range(MAX_NODES):
                    pj = pj * probs[b, j, joint_actions[:, j]]
                expected_q[b] = (pj * qmin).sum()

            policy_loss = (self.alpha * log_probs.sum(-1).sum(-1) - expected_q).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # ---- Temperature loss ----
            alpha_loss = -(self.log_alpha *
                           (log_probs.sum(-1).sum(-1) + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            # ---- Soft target updates ----
            with torch.no_grad():
                for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

            self.train_steps += 1
            info = {
                "q1_loss": float(q1_loss.item()),
                "q2_loss": float(q2_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "alpha": self.alpha,
                "entropy": float(entropy.mean().item()),
            }
        return info

    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path):
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(state["policy"])
            self.q1.load_state_dict(state["q1"])
            self.q2.load_state_dict(state["q2"])
            self.log_alpha.data = state["log_alpha"].to(self.device)