import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import pickle
import os
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator
from collections import defaultdict


class PPOAgent:
    """
    Deep PPO agent that follows the EXACT logical flow of Tabular Actor-Critic:
    - Same state representation
    - Same action space enumeration
    - Same ε-style exploration with temperature
    - Same Monte-Carlo returns
    - Same terminal reward handling
    - Last layer always assigned to edge
    """

    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test=False,
        gamma=0.95,
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
        self.is_test = is_test
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # ===== STATE REPRESENTATION =====
        # State: (bandwidth, cloud_time, layer, prev_action, surplus, neg_count)
        self.state_dim = 6  # Raw state dimensions
        
        # For action enumeration
        self.max_nodes = 3
        self.max_actions = 2 ** self.max_nodes  # 8 possible patterns per layer

        # ===== EXPLORATION PARAMETERS (match tabular) =====
        self.temperature = 1.0
        self.temperature_min = 0.25
        self.temperature_decay = 0.999
        self.temperature_boost = 0.35
        self.epsilon_min = 0.05

        # ===== TRACKING VARIABLES (match tabular) =====
        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0
        self.stagnant_limit = 10000
        self.total_episodes = 0
        
        # Node execution tracking
        self.edge_execution_counts = {}
        self.cloud_execution_counts = {}

        # ===== SIMULATOR =====
        self.simulator = CloudEdgeSimulator(profiling_data)

        # Build network
        self.policy = ActionPreferenceNetwork(self.state_dim, self.max_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffer
        self.trajectories = []
        self.current_trajectory = []

    # ======================================================
    # ACTION SPACE ENUMERATION
    # ======================================================
    def _get_possible_actions(self, layer_idx):
        """Enumerate all possible binary patterns for the current layer"""
        nodes = len(self.profiling.layers[layer_idx])
        
        # Last layer MUST be all edge (location=0)
        if layer_idx == len(self.profiling.layers) - 1:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            a[:, 1] = 0  # All edge
            return [a]
        
        # For non-last layers, enumerate all binary patterns
        max_patterns = 64
        patterns = list(range(2 ** nodes))
        if len(patterns) > max_patterns:
            patterns = random.sample(patterns, max_patterns)
        
        actions = []
        for p in patterns:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            for i in range(nodes):
                a[i, 1] = (p >> i) & 1  # 0=edge, 1=cloud
            actions.append(a)
        return actions

    def _action_to_index(self, action):
        """Convert action to a unique index for preference lookup"""
        locations = action[:, 1]
        # Convert binary pattern to index (e.g., [0,1,0] -> 2)
        idx = 0
        for i, loc in enumerate(locations):
            if loc == 1:
                idx |= (1 << i)
        return idx

    def _index_to_action(self, idx, layer_idx):
        """Convert index back to action array"""
        nodes = len(self.profiling.layers[layer_idx])
        a = np.zeros((nodes, 2), dtype=int)
        a[:, 0] = layer_idx
        
        # Last layer must be all edge
        if layer_idx == len(self.profiling.layers) - 1:
            a[:, 1] = 0
            return a
        
        # Convert index to binary pattern
        for i in range(nodes):
            a[i, 1] = (idx >> i) & 1
        return a

    # ======================================================
    # ACTION SELECTION (with NaN protection)
    # ======================================================
    def choose_action(self, state):
        """
        Returns action in the same format as tabular: (nodes, 2) array
        """
        layer = int(state[2])
        possible_actions = self._get_possible_actions(layer)
        
        # ε-greedy exploration
        if not self.is_test and random.random() < self.epsilon_min:
            return random.choice(possible_actions)
        
        # Build state tensor
        s_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get preferences for all possible actions
            all_preferences, value = self.policy(s_tensor)  # [1, max_actions]
        
        # Get preferences only for valid actions in this layer
        valid_indices = [self._action_to_index(a) for a in possible_actions]
        valid_preferences = all_preferences[0, valid_indices].cpu().numpy()
        
        # ===== SAFEGUARDS AGAINST NaN =====
        # Check for NaN or Inf and replace with zeros
        if np.any(np.isnan(valid_preferences)) or np.any(np.isinf(valid_preferences)):
            print(f"Warning: NaN/Inf detected in preferences. Using uniform distribution.")
            # Return random action as fallback
            return random.choice(possible_actions)
        
        # Apply temperature scaling
        prefs = valid_preferences / max(self.temperature, 1e-6)
        
        # Subtract max for numerical stability
        prefs = prefs - np.max(prefs)
        
        # Clip extremely negative values
        prefs = np.clip(prefs, -500, 500)
        
        # Softmax with numerical safeguards
        exp_prefs = np.exp(prefs)
        probs = exp_prefs / (np.sum(exp_prefs) + 1e-10)
        
        # Final check for valid probabilities
        if np.any(np.isnan(probs)) or np.sum(probs) < 0.99:
            print(f"Warning: Invalid probabilities. Using uniform distribution.")
            return random.choice(possible_actions)
        
        # Ensure exact sum to 1.0
        probs = probs / np.sum(probs)
        
        # Select action
        if self.is_test:
            # Greedy selection for testing
            best_idx = np.argmax(probs)
            return possible_actions[best_idx]
        else:
            try:
                # Stochastic selection for training
                selected_idx = np.random.choice(len(possible_actions), p=probs)
                selected_action = possible_actions[selected_idx]
                
                # Track execution
                self.track_action_execution(selected_action, layer)
                
                return selected_action
            except ValueError:
                # Fallback if random.choice fails
                print(f"Random choice failed. Using uniform fallback.")
                return random.choice(possible_actions)

    # ======================================================
    # TRACK ACTION EXECUTION
    # ======================================================
    def track_action_execution(self, action, layer):
        """Track where each node was executed (edge=0, cloud=1)"""
        for node_idx, (_, location) in enumerate(action):
            key = (layer, node_idx)
            if location == 0:  # Edge execution
                self.edge_execution_counts[key] = self.edge_execution_counts.get(key, 0) + 1
            else:  # Cloud execution
                self.cloud_execution_counts[key] = self.cloud_execution_counts.get(key, 0) + 1

    # ======================================================
    # ENVIRONMENT STEP
    # ======================================================
    def train(self, current_state):
        """Execute one step in the environment"""
        action = self.choose_action(current_state)
        
        # Get next cloud waiting time
        next_cloud = self.simulator.get_next_state_cloud_waiting_time(
            next_layer=min(int(current_state[2]) + 1, len(self.profiling.layers) - 1),
            current_action=action,
            isAllCloud=False,
        )
        
        # Compute energy and time
        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state,
            current_action=action,
            cloud_pending_ms=next_cloud,
        )
        
        # Calculate reward
        reward, surplus, neg_count, fractional_deadline = \
            self.simulator.calculate_reward(
                int(current_state[2]),
                energy,
                completion_time_s,
                current_state[4],
                current_state[5],
                isA2C=True,
            )
        
        # Get next state
        next_state, terminal, _ = self.simulator.get_next_state(
            current_state,
            action,
            surplus,
            neg_count,
            new_cloud_pending=next_cloud,
        )
        
        # Store transition in current trajectory
        if not self.is_test:
            # ===== FIX: Store state as tuple but convert properly later =====
            self.current_trajectory.append({
                'state': current_state,  # Keep as tuple for now
                'action': action,
                'action_idx': self._action_to_index(action),
                'reward': reward,
                'done': terminal
            })
        
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
    # ======================================================
    # PPO UPDATE
    # ======================================================
    def update(self):
        """Perform PPO update using Monte-Carlo returns"""
        if not self.current_trajectory:
            return
        
        # Add current trajectory to buffer
        self.trajectories.append(self.current_trajectory)
        self.current_trajectory = []
        
        # Prepare data for all trajectories
        all_states = []
        all_actions = []
        all_returns = []
        all_old_values = []
        all_old_preferences = []
        
        for trajectory in self.trajectories:
            # Compute Monte-Carlo returns
            returns = []
            G = 0
            for step in reversed(trajectory):
                G = step['reward'] + self.gamma * G
                G = np.clip(G, -1000.0, 1000.0)
                returns.insert(0, G)
            
            # Get old preferences and values
            for i, step in enumerate(trajectory):
                # ===== FIX: Convert state tuple to float array properly =====
                state_array = np.array([
                    float(step['state'][0]),  # bandwidth
                    float(step['state'][1]),  # cloud_time
                    float(step['state'][2]),  # layer
                    0.0,                       # prev_action (placeholder)
                    float(step['state'][4]),  # surplus
                    float(step['state'][5])   # neg_count
                ], dtype=np.float32)
                
                s_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prefs, value = self.policy(s_tensor)
                
                all_states.append(state_array)
                all_actions.append(step['action_idx'])
                all_returns.append(returns[i])
                all_old_values.append(value.squeeze().cpu().item())
                all_old_preferences.append(prefs.squeeze()[step['action_idx']].cpu().item())
        
        # ===== FIX: Ensure proper conversion to tensor =====
        # Convert lists to numpy arrays first
        all_states = np.array(all_states, dtype=np.float32)
        all_actions = np.array(all_actions, dtype=np.int64)
        all_returns = np.array(all_returns, dtype=np.float32)
        all_old_values = np.array(all_old_values, dtype=np.float32)
        all_old_preferences = np.array(all_old_preferences, dtype=np.float32)
        
        # Convert to tensors
        states_tensor = torch.from_numpy(all_states).to(self.device)
        actions_tensor = torch.from_numpy(all_actions).to(self.device)
        returns_tensor = torch.from_numpy(all_returns).to(self.device).unsqueeze(1)
        old_values_tensor = torch.from_numpy(all_old_values).to(self.device).unsqueeze(1)
        old_preferences_tensor = torch.from_numpy(all_old_preferences).to(self.device).unsqueeze(1)
        
        # Compute advantages
        advantages = returns_tensor - old_values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        dataset_size = len(all_states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                mb_indices = indices[start:end]
                
                mb_states = states_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_old_prefs = old_preferences_tensor[mb_indices]
                
                # Forward pass
                all_prefs, values = self.policy(mb_states)
                
                # Get preferences for taken actions
                action_prefs = all_prefs[range(len(mb_actions)), mb_actions].unsqueeze(1)
                
                # Compute ratio with clipping to prevent extreme values
                ratio = torch.exp(torch.clamp(action_prefs - mb_old_prefs, -5, 5))
                
                # PPO clipped objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, mb_returns)
                
                # Entropy bonus
                probs = torch.softmax(all_prefs / self.temperature, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Clear trajectories after update
        self.trajectories = []
    # ======================================================
    # EPISODE MANAGEMENT
    # ======================================================
    def notify_episode_end(self, episode_reward):
        """Match tabular's temperature boosting logic"""
        self.total_episodes += 1
        
        # Ensure temperature doesn't go too low
        self.temperature = max(self.temperature, 0.1)
        
        if episode_reward > self.best_episode_reward + 1e-6:
            self.best_episode_reward = episode_reward
            self.episodes_since_improvement = 0
            self.temperature = max(
                self.temperature_min,
                self.temperature * 0.995
            )
        else:
            self.episodes_since_improvement += 1
            if self.episodes_since_improvement >= self.stagnant_limit:
                self.temperature = min(
                    2.0,
                    self.temperature * self.temperature_boost
                )
                self.episodes_since_improvement = 0
                print(f"🔥 Temperature boosted to {self.temperature:.2f}")
            else:
                self.temperature = max(
                    self.temperature_min,
                    self.temperature * self.temperature_decay
                )
        
        # Perform update at episode end
        if not self.is_test:
            self.update()

    # ======================================================
    # STATISTICS
    # ======================================================
    def get_execution_stats(self):
        """Return execution statistics"""
        return {
            'edge_counts': self.edge_execution_counts,
            'cloud_counts': self.cloud_execution_counts,
            'total_episodes': self.total_episodes
        }

    # ======================================================
    # PERSISTENCE
    # ======================================================
    def save(self, filepath="ppo_policy.pth"):
        """Save model weights and tracking variables"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'temperature': self.temperature,
            'best_episode_reward': self.best_episode_reward,
            'episodes_since_improvement': self.episodes_since_improvement,
            'total_episodes': self.total_episodes,
            'edge_counts': self.edge_execution_counts,
            'cloud_counts': self.cloud_execution_counts,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath="ppo_policy.pth", load_optimizer=False):
        """Load model weights and tracking variables"""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Starting with fresh weights.")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load tracking variables
        self.temperature = checkpoint.get('temperature', 1.0)
        self.best_episode_reward = checkpoint.get('best_episode_reward', -1e9)
        self.episodes_since_improvement = checkpoint.get('episodes_since_improvement', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.edge_execution_counts = checkpoint.get('edge_counts', {})
        self.cloud_execution_counts = checkpoint.get('cloud_counts', {})
        
        print(f"Model loaded from {filepath}")


class ActionPreferenceNetwork(nn.Module):
    """
    Neural network that outputs preferences for all possible actions
    and state value.
    """
    def __init__(self, input_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.preference_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x):
        features = self.shared(x)
        preferences = self.preference_head(features)
        value = self.value_head(features)
        return preferences, value