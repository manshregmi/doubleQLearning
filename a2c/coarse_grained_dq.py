import numpy as np
import random
from typing import List, Tuple
from profiling.profile import ProfilingData
import pickle
import os

class OneShotDoubleQLearningWrapper:
    """
    Wrapper for one-shot Double Q-learning that returns metrics for plotting.
    Returns: (avg_energy, avg_time, deadline_miss_count)
    """
    
    def __init__(self, profiling: ProfilingData, deadline: float):
        self.profiling = profiling
        self.deadline = deadline
        
        # State tracking
        self.initial_bandwidth = profiling.bandwidth
        self.current_bandwidth = profiling.bandwidth
        self.current_cloud_time = 0.0
        self.current_slack = 0.0  # Initial slack (completion_time - deadline)
        
        # Discretization - Now 3D state space (bw, cloud_time, slack)
        self.bw_bins = np.linspace(1, 15, 15)
        self.cloud_bins = np.linspace(0, 300, 25)
        self.slack_bins = np.linspace(-100, 100, 20)  # Slack from -100ms to +100ms
        
        self.num_bw_bins = len(self.bw_bins)
        self.num_cloud_bins = len(self.cloud_bins)
        self.num_slack_bins = len(self.slack_bins)
        
        # Action space
        self.num_actions = self._calculate_total_actions()
        self._precompute_action_vectors()
        
        # Q-tables - Now 3D (bw, cloud, slack, actions)
        self.Q1 = np.ones((self.num_bw_bins, self.num_cloud_bins, self.num_slack_bins, self.num_actions))
        self.Q2 = np.ones((self.num_bw_bins, self.num_cloud_bins, self.num_slack_bins, self.num_actions))
        
        # RL parameters
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        
        # Episode tracking
        self.episode_count = 0
    
    def save_qtables(self, filename: str = "one_shot_DQ.pkl"):
        """
        Save Double Q-learning Q-tables and training state.
        """
        try:
            with open(filename, "wb") as f:
                pickle.dump((self.Q1, self.Q2, self.epsilon, self.episode_count), f)
            print(f"Double Q-learning Q-tables saved to {filename}")
        except Exception as e:
            print(f"Error saving Double Q-learning Q-tables: {e}")

    def load_qtables(self, filename: str = "one_shot_DQ.pkl"):
        """
        Load Double Q-learning Q-tables and training state.
        """
        try:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                    
                if isinstance(data, tuple) and len(data) == 4:
                    self.Q1, self.Q2, self.epsilon, self.episode_count = data
                else:
                    # Backwards compatibility - handle old 2D Q-tables
                    self.Q1, self.Q2 = data
                    # Reshape if needed (old format to new 3D format)
                    if len(self.Q1.shape) == 3:  # Old: (bw, cloud, actions)
                        old_shape = self.Q1.shape
                        new_shape = (old_shape[0], old_shape[1], self.num_slack_bins, old_shape[2])
                        self.Q1 = np.ones(new_shape)
                        self.Q2 = np.ones(new_shape)
                    self.epsilon = 1.0
                    self.episode_count = 0
                
                print(f"Double Q-learning Q-tables loaded from {filename}")
                print(f"  Q1 shape: {self.Q1.shape}, Q2 shape: {self.Q2.shape}")
                print(f"  Episode count: {self.episode_count}, Epsilon: {self.epsilon:.4f}")
                return True
            else:
                print(f"Double Q-learning Q-table file not found at {filename}. Starting fresh.")
                return False
        except Exception as e:
            print(f"Error loading Double Q-learning Q-tables: {e}")
            return False    
        
    def _calculate_total_actions(self) -> int:
        total = 1
        for layer_idx, layer in enumerate(self.profiling.layers):
            num_nodes = len(layer)
            if layer_idx == 6:
                total *= 1
            else:
                total *= (2 ** num_nodes)
        return total
    
    def _precompute_action_vectors(self):
        self.action_vectors = []
        self._generate_combinations_recursive(0, [])
    
    def _generate_combinations_recursive(self, layer_idx, current):
        if layer_idx == 7:
            action_vector = []
            for l_idx, node_assignments in enumerate(current):
                num_nodes = len(node_assignments)
                action_matrix = np.zeros((num_nodes, 2), dtype=int)
                action_matrix[:, 0] = l_idx
                action_matrix[:, 1] = node_assignments
                action_vector.append(action_matrix)
            self.action_vectors.append(action_vector)
            return
        
        num_nodes = len(self.profiling.layers[layer_idx])
        if layer_idx == 6:
            current.append(np.zeros(num_nodes, dtype=int))
            self._generate_combinations_recursive(layer_idx + 1, current)
            current.pop()
        else:
            for pattern in range(2 ** num_nodes):
                arr = np.zeros(num_nodes, dtype=int)
                for i in range(num_nodes):
                    arr[i] = (pattern >> i) & 1
                current.append(arr)
                self._generate_combinations_recursive(layer_idx + 1, current)
                current.pop()
    
    def discretize_state(self, bandwidth: float, cloud_time: float, slack: float) -> Tuple[int, int, int]:
        """
        Discretize continuous state into 3D discrete indices.
        Returns: (bw_idx, cloud_idx, slack_idx)
        """
        # Discretize bandwidth
        bw_idx = np.digitize([bandwidth], self.bw_bins, right=True)[0] - 1
        bw_idx = max(0, min(bw_idx, self.num_bw_bins - 1))
        
        # Discretize cloud time
        cloud_idx = np.digitize([cloud_time], self.cloud_bins, right=True)[0] - 1
        cloud_idx = max(0, min(cloud_idx, self.num_cloud_bins - 1))
        
        # Discretize slack (clamp to bin range)
        slack_clamped = max(min(slack, self.slack_bins[-1]), self.slack_bins[0])
        slack_idx = np.digitize([slack_clamped], self.slack_bins, right=True)[0] - 1
        slack_idx = max(0, min(slack_idx, self.num_slack_bins - 1))
        
        return bw_idx, cloud_idx, slack_idx
    
    def get_episode_state(self) -> Tuple[float, float, float]:
        """
        Get initial state for an episode.
        Returns: (bandwidth, cloud_time, slack)
        """
        if self.episode_count == 0:
            return self.initial_bandwidth, 0.0, 0.0
        else:
            return self.current_bandwidth, self.current_cloud_time, self.current_slack
    
    def choose_action(self, bandwidth: float, cloud_time: float, slack: float) -> int:
        """
        Choose action using epsilon-greedy policy based on current state.
        """
        bw_idx, cloud_idx, slack_idx = self.discretize_state(bandwidth, cloud_time, slack)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            Q_avg = (self.Q1[bw_idx, cloud_idx, slack_idx] + self.Q2[bw_idx, cloud_idx, slack_idx]) / 2
            return np.argmax(Q_avg)
    
    def update(self, state_bw: float, state_cloud: float, state_slack: float, 
               action_id: int, reward: float, 
               next_bw: float, next_cloud: float, next_slack: float):
        """
        Update Q-tables using Double Q-learning update rule.
        """
        bw_idx, cloud_idx, slack_idx = self.discretize_state(state_bw, state_cloud, state_slack)
        next_bw_idx, next_cloud_idx, next_slack_idx = self.discretize_state(next_bw, next_cloud, next_slack)
        
        # Randomly choose which Q-table to update
        if random.random() < 0.5:
            Q_update, Q_eval = self.Q1, self.Q2
        else:
            Q_update, Q_eval = self.Q2, self.Q1
        
        current_q = Q_update[bw_idx, cloud_idx, slack_idx, action_id]
        best_next_action = np.argmax(Q_update[next_bw_idx, next_cloud_idx, next_slack_idx])
        next_q = Q_eval[next_bw_idx, next_cloud_idx, next_slack_idx, best_next_action]
        
        target = reward + self.gamma * next_q
        Q_update[bw_idx, cloud_idx, slack_idx, action_id] += self.alpha * (target - current_q)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def run_episode(self, simulator):
        """Run one episode and return metrics."""
        initial_bw, initial_cloud, initial_slack = self.get_episode_state()
        
        # Choose action
        action_id = self.choose_action(initial_bw, initial_cloud, initial_slack)
        action_vector = self.action_vectors[action_id]
        
        # Execute
        total_energy = 0.0
        total_time = 0.0
        total_cloud_waiting = 0.0
        
        current_state = (initial_bw, initial_cloud, 0, None, 0.0, 0)
        current_cloud_pending = initial_cloud
        
        for level in range(7):
            action = action_vector[level]
            
            energy, time_s = simulator.compute_energy_and_time(
                current_state, action, current_cloud_pending
            )
            
            next_cloud_pending = simulator.get_next_state_cloud_waiting_time(
                next_layer=level,
                current_action=action,
                isAllCloud=False
            )
            
            total_energy += energy
            total_time += time_s * 1000.0
            total_cloud_waiting += next_cloud_pending
            
            next_state, _, _ = simulator.get_next_state(
                current_state, action, 0.0, 0, next_cloud_pending
            )

            self.current_bandwidth = next_state[0]
            
            current_state = next_state
            current_cloud_pending = next_cloud_pending
        
        # Calculate slack: completion_time - deadline
        slack = total_time - self.deadline
        
        # Check deadline
        deadline_missed = 1 if slack > 0 else 0  # Positive slack means missed deadline
        
        # Calculate reward
        reward = -total_energy
        if deadline_missed:
            excess_ratio = slack / self.deadline
            reward -= 10000000 * excess_ratio
        
        # Update for next episode
        next_bw = current_state[0]
        next_cloud = total_cloud_waiting
        next_slack = slack  # Slack for next episode is slack from current episode
        
        # Update Q-tables
        self.update(initial_bw, initial_cloud, initial_slack, action_id, reward, 
                   next_bw, next_cloud, next_slack)
        
        # Update state for next episode
        self.current_cloud_time = next_cloud
        self.current_slack = next_slack
        self.episode_count += 1
        
        return total_energy, total_time, deadline_missed
    
    def run_simulation(self, episodes, max_steps, is_test=False):
        """
        Run simulation for multiple episodes.
        Returns: avg_energy, avg_time, deadline_miss_count
        """
        from simulator.simulator import CloudEdgeSimulator
        
        simulator = CloudEdgeSimulator(self.profiling)
        
        energies = []
        times = []
        deadline_misses = 0
        
        for episode in range(episodes):
            energy, time, missed = self.run_episode(simulator)
            
            energies.append(energy)
            times.append(time)
            deadline_misses += missed
            
            if is_test and (episode + 1) % 1000 == 0:
                avg_slack = np.mean([t - self.deadline for t in times[-1000:]])
                print(f"Episode {episode+1}: Energy={np.mean(energies[-1000:]):.2f}J, "
                      f"Time={np.mean(times[-1000:]):.1f}ms, "
                      f"Slack={avg_slack:.1f}ms, "
                      f"Miss={deadline_misses/(episode+1)*100:.1f}%")
        
        return np.mean(energies), np.mean(times), deadline_misses
    


def run_oneshot_doubleQ_simulation(profiling_data, episodes, max_steps, is_test=False):
    """
    Run one-shot Double Q-learning simulation.
    Returns: avg_energy, avg_time, deadline_miss_count
    """
    agent = OneShotDoubleQLearningWrapper(profiling_data, profiling_data.deadline)
    agent.load_qtables()
    result = agent.run_simulation(episodes, max_steps, is_test)
    agent.save_qtables()
    return result