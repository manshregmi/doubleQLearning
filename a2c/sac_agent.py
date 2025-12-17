import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
import os
from typing import Tuple, Any, List, Dict, Optional
from collections import deque
import math
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


class DiscreteActor(nn.Module):
    """
    Actor network for discrete action space using Gumbel-Softmax.
    Outputs logits for each possible action pattern.
    """
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # Logits for each action


class Critic(nn.Module):
    """Q-network for SAC."""
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action_one_hot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_one_hot], dim=-1)
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action_idx: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action_idx, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class SoftActorCriticAgent:
    """
    Soft Actor-Critic agent for layer-by-layer offloading with discrete actions.
    
    Key features:
    - Discrete SAC with Gumbel-Softmax reparameterization
    - Automatic entropy tuning
    - Double Q-networks for stability
    - Experience replay
    """
    
    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test: bool = False,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.01,  # Fixed entropy coefficient (if auto_entropy=False)
        hidden_dim: int = 256,
        buffer_size: int = 100000,
        batch_size: int = 256,
        auto_entropy: bool = True,
        target_entropy_scale: float = 0.98,  # -0.98 * log(1/num_actions)
        BW_bins: int = 15,
        CT_bins: int = 20,
        surplus_bins: int = 25,
    ):
        """
        Args:
            profiling_data: ProfilingData instance
            is_test: If True, no training updates
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            alpha: Entropy regularization coefficient
            hidden_dim: Hidden layer dimension
            buffer_size: Replay buffer size
            batch_size: Training batch size
            auto_entropy: Whether to automatically tune entropy coefficient
            target_entropy_scale: Scale for target entropy
        """
        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy = auto_entropy
        
        # Discretization bins (same as before)
        self.bandwidth_bins = np.linspace(1, 15, BW_bins)
        self.cloudtime_bins = np.linspace(0, 100, CT_bins)
        self.surplus_bins = np.linspace(-25, 25, surplus_bins)

        # State dimension after discretization
        self.state_dim = 6  # [bw_disc, ctime_disc, layer, surplus_disc, neg_count, prev_action_encoded]
        
        # Maximum number of actions per layer
        self.max_actions_per_layer = 64  # As in your code
        
        # Initialize networks
        self.num_actions = self.max_actions_per_layer
        self.actor = DiscreteActor(self.state_dim, self.num_actions, hidden_dim)
        self.critic1 = Critic(self.state_dim, self.num_actions, hidden_dim)
        self.critic2 = Critic(self.state_dim, self.num_actions, hidden_dim)
        self.critic1_target = Critic(self.state_dim, self.num_actions, hidden_dim)
        self.critic2_target = Critic(self.state_dim, self.num_actions, hidden_dim)
        
        # Copy weights to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Entropy temperature
        self.target_entropy = target_entropy_scale * -np.log(1 / self.num_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.alpha = alpha if not auto_entropy else self.log_alpha.exp().item()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Simulator
        self.simulator = CloudEdgeSimulator(profiling_data)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Bookkeeping
        self.optimum_action_layer_count = [0] * 7
        self.last_layer_not_optimum = 0
        self.training_step = 0
        
        # Action cache for each layer
        self.action_cache = {}
        
    def to(self, device):
        """Move networks to device."""
        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.critic1_target.to(device)
        self.critic2_target.to(device)
        if self.auto_entropy:
            self.log_alpha.to(device)
    
    # ---------------------------
    # State discretization (similar to Double Q-learning)
    # ---------------------------
    def _discretize(self, value: float, bins: np.ndarray) -> float:
        idx = np.digitize([value], bins, right=True)[0] - 1
        idx = max(0, min(idx, len(bins) - 1))
        return float(bins[idx])
    
    def _action_to_idx(self, action: np.ndarray, layer_idx: int) -> int:
        """Convert action array to index in action list."""
        actions = self._get_possible_actions(layer_idx)
        for idx, act in enumerate(actions):
            if np.array_equal(action, act):
                return idx
        return 0  # Fallback
    
    def _idx_to_action(self, idx: int, layer_idx: int) -> np.ndarray:
        """Convert index to action array."""
        actions = self._get_possible_actions(layer_idx)
        return actions[idx] if idx < len(actions) else actions[0]
    
    def _state_to_vector(self, state: Tuple[Any, ...]) -> np.ndarray:
        """
        Convert state to vector representation.
        """
        bw, ctime, layer, prev_action, surplus, negative_surplus_count = state
        
        # Discretize continuous components
        bw_disc = self._discretize(float(bw), self.bandwidth_bins)
        ctime_disc = self._discretize(float(ctime), self.cloudtime_bins)
        surplus_disc = self._discretize(float(surplus), self.surplus_bins)
        
        # Encode previous action (one-hot like)
        prev_action_encoded = 0.0
        if prev_action is not None:
            # Simple encoding: sum of action decisions
            prev_action_encoded = prev_action[:, 1].sum() / max(len(prev_action), 1)
        
        # Normalize components
        state_vector = np.array([
            bw_disc / 15.0,  # Normalized bandwidth
            ctime_disc / 100.0,  # Normalized cloud time
            layer / 6.0,  # Normalized layer index
            (surplus_disc + 25) / 50.0,  # Normalized surplus
            negative_surplus_count / 10.0,  # Normalized negative count
            prev_action_encoded  # Already normalized
        ])
        
        return state_vector
    
    # ---------------------------
    # Action generation (same as before)
    # ---------------------------
    def _get_possible_actions(self, layer_idx: int) -> List[np.ndarray]:
        """Generate feasible offloading action patterns for a given layer."""
        if layer_idx in self.action_cache:
            return self.action_cache[layer_idx]
            
        nodes = self.profiling.get_num_nodes(layer_idx)
        
        # Terminal layer → fixed local execution
        if layer_idx == (len(self.profiling.layers) - 1):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            a[:, 1] = 0
            self.action_cache[layer_idx] = [a]
            return [a]
        
        # Limit pattern explosion
        max_patterns = self.max_actions_per_layer
        all_patterns = list(range(2 ** nodes))
        if len(all_patterns) > max_patterns:
            patterns = random.sample(all_patterns, max_patterns)
        else:
            patterns = all_patterns
        
        actions = []
        for pattern in patterns:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            for i in range(nodes):
                a[i, 1] = (pattern >> i) & 1
            actions.append(a)
        
        self.action_cache[layer_idx] = actions
        return actions
    
    # ---------------------------
    # Action selection with Gumbel-Softmax
    # ---------------------------
    def select_action(self, state: Tuple[Any, ...], evaluate: bool = False) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            evaluate: If True, use deterministic action for evaluation
        
        Returns:
            Selected action array
        """
        layer_idx = int(state[2])
        
        if self.is_test or evaluate:
            # Deterministic evaluation
            state_vec = self._state_to_vector(state)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.actor(state_tensor)
                if layer_idx == (len(self.profiling.layers) - 1):
                    # Terminal layer: always choose first action (local execution)
                    action_idx = 0
                else:
                    action_idx = torch.argmax(logits, dim=1).item()
            
            return self._idx_to_action(action_idx, layer_idx)
        
        else:
            # Training: sample from policy
            state_vec = self._state_to_vector(state)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.actor(state_tensor)
                
                if layer_idx == (len(self.profiling.layers) - 1):
                    # Terminal layer: deterministic
                    action_idx = 0
                else:
                    # Sample using Gumbel-Softmax
                    temperature = max(0.1, self.alpha)  # Use entropy temperature
                    action_probs = F.softmax(logits / temperature, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action_idx = action_dist.sample().item()
            
            return self._idx_to_action(action_idx, layer_idx)
    
    # ---------------------------
    # Training step
    # ---------------------------
    def train(self, current_state: Tuple[Any, ...]) -> Tuple:
        """
        Perform one environment step and train if not in test mode.
        
        Returns:
            Same tuple as Double Q-learning: (action, reward, next_state, terminal, 
            energy, completion_time, next_bandwidth, surplus, fractional_deadline)
        """
        # Select action
        action = self.select_action(current_state, evaluate=self.is_test)
        
        # Count optimum actions (for diagnostics)
        self.is_optimum_action(action, int(current_state[2]))
        
        # Get next state from simulator
        next_state_cloud_processing = self.simulator.get_next_state_cloud_waiting_time(
            next_layer=(int(current_state[2]))
            if ((int(current_state[2]) + 1) < len(self.profiling.layers))
            else int(current_state[2]),
            current_action=action,
            isAllCloud=False,
        )
        
        # Simulator step
        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state,
            current_action=action,
            cloud_pending_ms=next_state_cloud_processing,
        )
        
        # Reward computation
        reward, surplus, negative_surplus_count, fractional_deadline = self.simulator.calculate_reward(
            int(current_state[2]),
            energy,
            completion_time_s,
            current_state[4],
            current_state[5],
            isA2C=True,
        )
        
        # Next state
        next_state, terminal, _ = self.simulator.get_next_state(
            current_state,
            action,
            surplus,
            negative_surplus_count,
            new_cloud_pending=next_state_cloud_processing,
        )
        
        # Store transition in replay buffer (if not testing)
        if not self.is_test:
            state_vec = self._state_to_vector(current_state)
            next_state_vec = self._state_to_vector(next_state)
            action_idx = self._action_to_idx(action, int(current_state[2]))
            
            self.replay_buffer.push(
                state_vec, action_idx, reward, next_state_vec, terminal
            )
            
            # Train from replay buffer
            if len(self.replay_buffer) >= self.batch_size:
                self._update_networks()
        
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
        )
    
    def _update_networks(self):
        """Update SAC networks from replay buffer."""
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return
        
        states, action_idxs, rewards, next_states, dones = batch
        states = states.to(self.device)
        action_idxs = action_idxs.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Convert action indices to one-hot
        action_one_hot = F.one_hot(action_idxs, num_classes=self.num_actions).float()
        
        # Update critics
        with torch.no_grad():
            # Get next action probabilities from target actor
            next_logits = self.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            
            # Compute next Q-values
            next_action_one_hot = next_probs  # Soft action representation
            next_q1 = self.critic1_target(next_states, next_action_one_hot)
            next_q2 = self.critic2_target(next_states, next_action_one_hot)
            next_q = torch.min(next_q1, next_q2)
            
            # Add entropy term
            next_q = next_q - self.alpha * (next_probs * next_log_probs).sum(dim=-1, keepdim=True)
            
            # Target Q-value
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Current Q-values
        current_q1 = self.critic1(states, action_one_hot)
        current_q2 = self.critic2(states, action_one_hot)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # Update actor
        logits = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            # Sample actions from current policy for Q-value computation
            q1 = self.critic1(states, probs)
            q2 = self.critic2(states, probs)
            q = torch.min(q1, q2)
        
        # Actor loss with entropy regularization
        actor_loss = (probs * (self.alpha * log_probs - q)).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update entropy temperature (if auto-tuning)
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy).mean())
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        
        self.training_step += 1
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    # ---------------------------
    # Optimum action counting (same as before)
    # ---------------------------
    def is_optimum_action(self, chosen_action: np.ndarray, layer_idx: int) -> bool:
        optimum_actions = [
            np.array([[0, 0]]),
            np.array([[1, 1]]),
            np.array([[2, 1]]),
            np.array([[3, 1], [3, 1], [3, 1]]),
            np.array([[4, 1], [4, 1], [4, 1]]),
            np.array([[5, 1], [5, 1], [5, 1]]),
            np.array([[6, 0]]),
        ]

        if layer_idx < 0 or layer_idx >= len(optimum_actions):
            print(f"⚠️ Layer {layer_idx} not in optimum action list.")
            return False

        optimum = optimum_actions[layer_idx]
        same_shape = chosen_action.shape == optimum.shape
        same_values = np.array_equal(chosen_action, optimum)

        if same_shape and same_values:
            self.optimum_action_layer_count[layer_idx] += 1
            return True
        elif layer_idx == len(self.profiling.layers) - 1:
            self.last_layer_not_optimum += 1
            return False
        return False
    
    # ---------------------------
    # Persistence
    # ---------------------------
    def save_models(self, filename: str = "sac_models.pth"):
        """Save all networks and optimizers."""
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'critic1_target_state_dict': self.critic1_target.state_dict(),
                'critic2_target_state_dict': self.critic2_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy else None,
                'log_alpha': self.log_alpha if self.auto_entropy else None,
                'alpha': self.alpha,
                'action_cache': self.action_cache,
                'training_step': self.training_step,
            }, filename)
            print(f"Models saved to {filename}")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self, filename: str = "sac_models.pth"):
        """Load all networks and optimizers."""
        try:
            if os.path.exists(filename):
                checkpoint = torch.load(filename, map_location=self.device)
                
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
                self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
                self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
                self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
                
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
                self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
                
                if self.auto_entropy and checkpoint['alpha_optimizer_state_dict'] is not None:
                    self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                    self.log_alpha = checkpoint['log_alpha']
                
                self.alpha = checkpoint['alpha']
                self.action_cache = checkpoint.get('action_cache', {})
                self.training_step = checkpoint.get('training_step', 0)
                
                print(f"Models loaded from {filename}, training step: {self.training_step}")
                return True
            else:
                print(f"Model file not found at {filename}")
                return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def notify_episode_end(self, episode_reward: float):
        """Optional: can be used for logging or adaptive hyperparameters."""
        pass