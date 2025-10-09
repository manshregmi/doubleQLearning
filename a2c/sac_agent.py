import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


# ===============================
# Replay Buffer
# ===============================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ===============================
# Networks
# ===============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action * self.max_action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        # Flatten inputs to [batch, feature]
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        if action.dim() > 2:
            action = action.view(action.size(0), -1)
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# ===============================
# SAC Agent
# ===============================
class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_capacity=100000,
        batch_size=64,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_action = max_action

        # Actor and Critics
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

    # ---------------------------
    def select_action(self, state, layer, simulator, epsilon=0.2, evaluate=False):
        """
        Returns a discrete action matrix for a given layer using SAC output.

        Args:
            state (np.array): Flattened state for the actor network.
            layer (int): Current layer index.
            simulator (CloudEdgeSimulator): To generate valid actions.
            epsilon (float): Random action probability.
            evaluate (bool): If True, use deterministic action.

        Returns:
            np.ndarray: Discrete action matrix (num_nodes, 2)
        """
        nodes = simulator.profiling.get_num_nodes(layer)

        # 1. First or last layer -> edge only
        if layer == 0 or layer == len(simulator.profiling.layers) - 1:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            a[:, 1] = 0
            return a

        # 2. SAC continuous action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            mean, _ = self.actor.forward(state_tensor)
            cont_action = torch.tanh(mean).detach().cpu().numpy().flatten()
        else:
            cont_action, _ = self.actor.sample(state_tensor)
            cont_action = cont_action.detach().cpu().numpy().flatten()
            # Add small Gaussian noise for exploration
            cont_action += np.random.normal(0, 0.1, size=cont_action.shape)
            cont_action = np.clip(cont_action, -1.0, 1.0)

        # 3. Get all possible discrete actions for this layer
        all_actions = simulator.get_possible_actions(layer)
        if len(all_actions) == 0:
            return np.array([])  # terminal

        # 4. Epsilon-greedy: sometimes pick a completely random action
        if np.random.rand() < epsilon:
            return random.choice(all_actions)

        # 5. Map first continuous value to discrete index
        normalized_action = (cont_action[0] + 1) / 2.0  # [-1,1] -> [0,1]
        idx = int(np.clip(normalized_action * len(all_actions), 0, len(all_actions) - 1))

        return all_actions[idx]



    # ---------------------------
    def update_parameters(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device),
        )

        # Sample next actions
        next_actions, next_log_probs = self.actor.sample(next_states)

        # Target Q-values
        target_q1 = self.target_critic1(next_states, next_actions)
        target_q2 = self.target_critic2(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
        target_value = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_value.detach())
        critic2_loss = F.mse_loss(current_q2, target_value.detach())

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target critics
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    # ===============================

    def save_checkpoint(self, filename="sac_checkpoint.pth"):
        try:
            checkpoint = {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic1_optimizer": self.critic1_optimizer.state_dict(),
                "critic2_optimizer": self.critic2_optimizer.state_dict(),
                "alpha": self.alpha,
            }
            torch.save(checkpoint, filename)
            print(f"[SAC] ✅ Checkpoint saved -> {filename}")
        except Exception as e:
            print(f"[SAC] ⚠️ Failed to save checkpoint: {e}")

    def load_checkpoint(self, filename="sac_checkpoint.pth"):
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic1.load_state_dict(checkpoint["critic1"])
            self.critic2.load_state_dict(checkpoint["critic2"])
            self.target_critic1.load_state_dict(checkpoint["target_critic1"])
            self.target_critic2.load_state_dict(checkpoint["target_critic2"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
            self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
            self.alpha = checkpoint.get("alpha", self.alpha)
            print(f"[SAC] 🔁 Checkpoint loaded -> {filename}")
        except FileNotFoundError:
            print(f"[SAC] ⚠️ No checkpoint found at {filename}. Starting fresh.")
        except Exception as e:
            print(f"[SAC] ⚠️ Failed to load checkpoint: {e}")
