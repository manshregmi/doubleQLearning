import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from profiling.profile import ProfilingData
import os
 
 
class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for cloud edge node assignment.
    Uses a neural network policy with independent Bernoulli actions per node.
    """
 
    def __init__(
        self,
        profiling_data: ProfilingData,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        epochs=10,
        batch_size=64,
        lr=3e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.profiling = profiling_data
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
 
        # State dimension: 4 global + 7 layer one‑hot + 3 prev_mask + 3 prev_assign + 12 node_features + 3 curr_mask = 32
        self.state_dim = 32
 
        # Build network
        self.policy = PolicyNetwork(self.state_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
 
        # Rollout buffer
        self.states = []
        self.actions = []       # list of lists (binary per node)
        self.log_probs = []      # list of floats
        self.values = []         # list of floats
        self.rewards = []        # list of floats
        self.dones = []          # list of bools
 
    def _build_state_vector(self, state):
        """
        Convert the simulator's state tuple into a fixed‑size numpy array.
        state = (bandwidth, cloud_time, layer, prev_action, surplus, neg_count)
        """
        bw, ct, layer, prev_action, surplus, neg_count = state
        layer = int(layer)
 
        # One‑hot encoding for layer (7 layers)
        layer_one_hot = np.zeros(7, dtype=np.float32)
        layer_one_hot[layer] = 1.0
 
        # Previous layer assignments and mask
        prev_mask = np.zeros(3, dtype=np.float32)
        prev_assign = np.zeros(3, dtype=np.float32)
        if layer > 0 and prev_action is not None:
            prev_layer_nodes = len(self.profiling.layers[layer - 1])
            for i in range(prev_layer_nodes):
                prev_mask[i] = 1.0
                prev_assign[i] = float(prev_action[i, 1])   # 0 or 1
 
        # Current layer node features and mask
        curr_features = np.zeros((3, 4), dtype=np.float32)  # 3 slots × 4 features
        curr_mask = np.zeros(3, dtype=np.float32)
        curr_layer_nodes = len(self.profiling.layers[layer])
        for i in range(curr_layer_nodes):
            curr_mask[i] = 1.0
            curr_features[i, 0] = self.profiling.get_node_edge_time(layer, i)      # ms
            curr_features[i, 1] = self.profiling.get_node_cloud_time(layer, i)     # ms
            curr_features[i, 2] = self.profiling.get_node_edge_power(layer, i)     # W
            curr_features[i, 3] = self.profiling.get_output_size(layer, i)         # KB
 
        # Concatenate everything
        state_vec = np.concatenate([
            [bw, ct, surplus, neg_count],          # 4
            layer_one_hot,                          # 7
            prev_mask,                              # 3
            prev_assign,                            # 3
            curr_features.flatten(),                # 12
            curr_mask,                              # 3
        ]).astype(np.float32)
 
        return state_vec
 
    def choose_action(self, state, deterministic=False):
        """
        Returns:
            action: list of 0/1 for each existing node in the current layer
            log_prob: scalar tensor (detached) of the log probability of the action
            value: scalar tensor (detached) of the state value
        """
        # Build state tensor
        s = self._build_state_vector(state)
        s_tensor = torch.from_numpy(s).unsqueeze(0).to(self.device)   # [1, state_dim]
 
        with torch.no_grad():
            logits, value = self.policy(s_tensor)   # logits shape [1, 3], value [1,1]
 
        # Determine which nodes exist in the current layer
        layer = int(state[2])
        num_nodes = len(self.profiling.layers[layer])
        mask = torch.zeros(3, dtype=torch.bool)
        mask[:num_nodes] = True
 
        # Apply mask to logits (set logits of non‑existing nodes to -inf so probability = 0)
        masked_logits = logits.clone()
        masked_logits[0, ~mask] = -float('inf')
 
        # Create Bernoulli distributions for existing nodes
        probs = torch.sigmoid(masked_logits)   # [1,3]
        dist = Bernoulli(probs=probs)
 
        if deterministic:
            # Greedy: choose 1 if prob >= 0.5 else 0
            action_probs = (probs >= 0.5).float()
            action = action_probs.squeeze(0).cpu().numpy().astype(int)
            log_prob = dist.log_prob(action_probs).sum(dim=-1, keepdim=True)
        else:
            action_sample = dist.sample()   # [1,3]
            action = action_sample.squeeze(0).cpu().numpy().astype(int)
            log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
 
        # Keep only the existing nodes' actions
        action_list = [int(action[i]) for i in range(num_nodes)]
 
        return action_list, log_prob.squeeze().cpu().item(), value.squeeze().cpu().item()
 
    def store_transition(self, state, action, log_prob, value, reward, done):
        """
        Store one step in the rollout buffer.
        """
        s_vec = self._build_state_vector(state)
        self.states.append(s_vec)
        self.actions.append(action)          # list of ints
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
 
    def _clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
 
    def update(self):
        """
        Perform a PPO update using the collected rollout buffer.
        """
        # Convert buffer to numpy arrays (states as 2D, others as 1D)
        states = np.array(self.states, dtype=np.float32)
        actions = self.actions          # list of lists (variable length)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)
        old_values = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
 
        # Compute advantages and returns using GAE
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0  # value after terminal state is 0
            else:
                next_val = old_values[t + 1] * (1 - dones[t + 1])
            delta = rewards[t] + self.gamma * next_val - old_values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
        returns = advantages + old_values
 
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
        # Convert to PyTorch tensors
        states_tensor = torch.from_numpy(states).to(self.device)
        actions_tensor = self._actions_to_tensor(actions).to(self.device)   # [len, 3] with -100 for padding
        old_log_probs_tensor = torch.from_numpy(old_log_probs).to(self.device).unsqueeze(1)
        returns_tensor = torch.from_numpy(returns).to(self.device).unsqueeze(1)
        advantages_tensor = torch.from_numpy(advantages).to(self.device).unsqueeze(1)
 
        # Create dataset for mini‑batch updates
        dataset_size = len(states)
        indices = np.arange(dataset_size)
 
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
 
                mb_states = states_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
 
                # Forward pass
                logits, values = self.policy(mb_states)   # logits: [batch, 3], values: [batch,1]
 
                # Create mask for existing nodes in each sample
                masks = self._get_masks_for_batch(mb_indices)   # [batch, 3] boolean
 
                # Compute log probability of taken actions
                log_probs = self._compute_log_probs(logits, mb_actions, masks)   # [batch,1]
 
                # Ratio for PPO clipping
                ratio = torch.exp(log_probs - mb_old_log_probs)
 
                # Surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
 
                # Value loss
                value_loss = nn.MSELoss()(values, mb_returns)
 
                # Entropy bonus (only over existing nodes)
                probs = torch.sigmoid(logits)
                masked_probs = probs * masks.float()
                dist = Bernoulli(probs=probs)
                entropy = dist.entropy() * masks.float()   # [batch,3]
                entropy = entropy.sum(dim=1).mean()
 
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
 
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
 
        # Clear buffer after update
        self._clear_buffer()
 
    def _actions_to_tensor(self, actions):
        """
        Convert list of action lists (variable length) to a fixed size tensor of shape [len, 3].
        Pad non‑existing nodes with -100 (ignored in loss).
        """
        max_nodes = 3
        batch = len(actions)
        tensor = torch.full((batch, max_nodes), -100, dtype=torch.float32)
        for i, act in enumerate(actions):
            tensor[i, :len(act)] = torch.tensor(act, dtype=torch.float32)
        return tensor
 
    def _get_masks_for_batch(self, indices):
        """
        Return boolean mask [batch, 3] indicating which nodes exist for each sample in the batch.
        Uses the stored states to retrieve the layer index.
        """
        masks = []
        for idx in indices:
            state_vec = self.states[idx]   # numpy array of length state_dim
            # The layer one‑hot is stored at indices 4..10 (0‑based after 4 globals)
            layer_one_hot = state_vec[4:11]
            layer = np.argmax(layer_one_hot)
            num_nodes = len(self.profiling.layers[layer])
            mask = np.zeros(3, dtype=bool)
            mask[:num_nodes] = True
            masks.append(mask)
        return torch.from_numpy(np.array(masks)).to(self.device)
 
    def _compute_log_probs(self, logits, actions, masks):
        """
        Compute log probability of taken actions, summing only over existing nodes.
        logits: [batch,3], actions: [batch,3] (padded with -100), masks: [batch,3] bool.
        """
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs=probs)
        # Replace padded values with 0 (dummy) but then mask them
        actions_safe = actions.clone()
        actions_safe[~masks] = 0.0   # dummy, will be masked
        log_probs = dist.log_prob(actions_safe)   # [batch,3]
        # Zero out masked positions and sum over nodes
        log_probs = (log_probs * masks.float()).sum(dim=1, keepdim=True)
        return log_probs
 
    # ==================== Save / Load Methods ====================
    def save(self, filepath="ppo_policy.pth"):
        """
        Save the policy network weights (and optimizer state) to a file.
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
 
    def load(self, filepath="ppo_policy.pth", load_optimizer=False):
        """
        Load the policy network weights (and optionally optimizer state) from a file.
        """
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Starting with fresh weights.")
            return
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
 
 
class PolicyNetwork(nn.Module):
    """
    Simple MLP with two heads: policy (logits) and value.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, 3)   # logits for 3 node slots
        self.value_head = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value