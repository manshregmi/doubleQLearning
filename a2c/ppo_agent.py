import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions import Categorical
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


class PPOAgent:

    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test=False,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        epochs=10,
        lr=3e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
    ):

        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.state_dim = 6
        self.max_nodes = 3
        self.max_actions = 2 ** self.max_nodes

        self.simulator = CloudEdgeSimulator(profiling_data)

        self.policy = ActionPreferenceNetwork(
            self.state_dim, self.max_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.rollout = []

        self.edge_execution_counts = {}
        self.cloud_execution_counts = {}

    # ================= STATE =================

    def _state_to_array(self, state):
        bw, ctime, layer, prev_action, surplus, neg_count = state
        return np.array(
            [bw, ctime, layer, surplus, neg_count, 0.0],
            dtype=np.float32,
        )

    # ================= ACTION SPACE =================

    def _get_possible_actions(self, layer_idx):
        nodes = len(self.profiling.layers[layer_idx])

        if layer_idx == len(self.profiling.layers) - 1:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            a[:, 1] = 0
            return [a]

        actions = []
        for p in range(2 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            for i in range(nodes):
                a[i, 1] = (p >> i) & 1
            actions.append(a)
        return actions

    def _action_to_index(self, action):
        idx = 0
        for i, (_, loc) in enumerate(action):
            if loc == 1:
                idx |= (1 << i)
        return idx

    def _index_to_action(self, idx, layer):
        nodes = len(self.profiling.layers[layer])
        a = np.zeros((nodes, 2), dtype=int)
        a[:, 0] = layer
        for i in range(nodes):
            a[i, 1] = (idx >> i) & 1
        return a

    def _build_mask(self, layer):
        mask = torch.zeros(self.max_actions, dtype=torch.bool)
        for a in self._get_possible_actions(layer):
            mask[self._action_to_index(a)] = True
        return mask

    # ================= POLICY =================

    def choose_action(self, state):

        layer = int(state[2])
        s = torch.tensor(
            self._state_to_array(state),
            dtype=torch.float32,
        ).to(self.device)

        mask = self._build_mask(layer).to(self.device)

        logits, value = self.policy(s.unsqueeze(0))
        logits = logits.squeeze()

        logits[~mask] = -1e10

        dist = Categorical(logits=logits)

        if self.is_test:
            action_idx = torch.argmax(dist.probs)
        else:
            action_idx = dist.sample()

        log_prob = dist.log_prob(action_idx)

        action = self._index_to_action(action_idx.item(), layer)

        if not self.is_test:
            self.rollout.append(
                {
                    "state": s,
                    "action": action_idx,
                    "log_prob": log_prob.detach(),
                    "value": value.squeeze().detach(),
                    "reward": None,
                    "done": None,
                    "mask": mask,
                }
            )

        return action

    # ================= STEP =================

    def train(self, state):

        action = self.choose_action(state)

        next_cloud = self.simulator.get_next_state_cloud_waiting_time(
            next_layer=min(
                int(state[2]) + 1,
                len(self.profiling.layers) - 1,
            ),
            current_action=action,
            isAllCloud=False,
        )

        energy, completion_time_s = \
            self.simulator.compute_energy_and_time(
                state,
                action,
                next_cloud,
            )

        reward, surplus, neg_count, fractional_deadline = \
            self.simulator.calculate_reward(
                int(state[2]),
                energy,
                completion_time_s,
                state[4],
                state[5],
                isA2C=True,
            )

        next_state, terminal, _ = self.simulator.get_next_state(
            state,
            action,
            surplus,
            neg_count,
            next_cloud,
        )

        if not self.is_test:
            self.rollout[-1]["reward"] = reward
            self.rollout[-1]["done"] = terminal

        return (
            action,
            reward,
            next_state,
            terminal,
            energy,
            completion_time_s,
            next_state[0],
            surplus,
            fractional_deadline,
            neg_count,
        )

    # ================= GAE =================

    def _compute_gae(self, next_value):

        advantages = []
        gae = 0

        for t in reversed(range(len(self.rollout))):

            r = self.rollout[t]["reward"]
            v = self.rollout[t]["value"]
            done = self.rollout[t]["done"]

            delta = r + self.gamma * next_value * (1 - done) - v
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae

            advantages.insert(0, gae)
            next_value = v

        returns = [
            adv + step["value"]
            for adv, step in zip(advantages, self.rollout)
        ]

        return advantages, returns

    # ================= PPO UPDATE =================

    def update(self):

        if len(self.rollout) == 0:
            return

        with torch.no_grad():
            last_state = self.rollout[-1]["state"]
            _, next_value = self.policy(last_state.unsqueeze(0))

        advantages, returns = self._compute_gae(
            next_value.item()
        )

        states = torch.stack(
            [r["state"] for r in self.rollout]
        ).to(self.device)

        actions = torch.stack(
            [r["action"] for r in self.rollout]
        ).to(self.device)

        old_log_probs = torch.stack(
            [r["log_prob"] for r in self.rollout]
        ).to(self.device)

        masks = torch.stack(
            [r["mask"] for r in self.rollout]
        ).to(self.device)

        returns = torch.tensor(
            returns, dtype=torch.float32
        ).to(self.device)

        advantages = torch.tensor(
            advantages, dtype=torch.float32
        ).to(self.device)

        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):

            logits, values = self.policy(states)
            logits[~masks] = -1e10

            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(
                new_log_probs - old_log_probs
            )

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon,
            ) * advantages

            policy_loss = -torch.min(
                surr1, surr2
            ).mean()

            value_loss = (
                (returns - values.squeeze()) ** 2
            ).mean()

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()

        self.rollout = []
    
    def save(self, path="ppo_policy.pth"):

        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

        print(f"PPO model saved to {path}")


    def load(self, path="ppo_policy.pth", load_optimizer=False):

        if not os.path.exists(path):
            print("No PPO model found. Training from scratch.")
            return

        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["model"])

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"PPO model loaded from {path}")
    
    def notify_episode_end(self, episode_reward=None):
        """
        Compatibility wrapper for old simulator logic.
        Simply triggers PPO policy update at episode end.
        """

        if not self.is_test:
            self.update()



class ActionPreferenceNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        num_actions,
        hidden_dim=128,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(
            hidden_dim,
            num_actions,
        )

        self.value_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(self, x):
        f = self.shared(x)
        return self.policy_head(f), self.value_head(f)