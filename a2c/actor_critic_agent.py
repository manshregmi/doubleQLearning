import numpy as np
import os
import random
from simulator.simulator import CloudEdgeSimulator

class A2CAgent:
    def __init__(self, profiling_data, alpha_v=0.02, alpha_p=0.02, gamma=0.95, epsilon=0.05):
        self.profiling = profiling_data
        self.alpha_v = alpha_v
        self.alpha_p = alpha_p
        self.gamma = gamma
        self.epsilon = epsilon
        self.simulator = CloudEdgeSimulator(profiling_data)

        self.value_table = {}   # V(s)
        self.policy_table = {}  # π(s,a)
        self.filename_value = "value_table.npy"
        self.filename_policy = "policy_table.npy"

        # --- Discretization bins ---
        # Bandwidth (Mbps)
        self.bandwidth_bins = np.linspace(1, 15, 6)      # [1, ~4.8, ~8.6, ~12.4, ~16.2, 30]
        # Cloud pending time (ms)
        self.cloudtime_bins = np.linspace(0, 100, 20)    # step ≈5.26 ms
        # Surplus (s)
        self.surplus_bins = np.linspace(-5, 5, 21)       # step 0.5 s

    # ---------- STATE / ACTION HANDLING ----------
    def action_to_key_part(self, action_matrix):
        """Converts the action matrix (N x 2) into a unique, immutable tuple for the state key."""
        if action_matrix is None:
            # Handle initial state case where no previous action exists
            return tuple([-1, -1]) 
        # Flatten the (N x 2) matrix into a 2*N tuple of integers
        return tuple(action_matrix.flatten().tolist())

    def discretize(self, value, bins):
        """Helper to map continuous value to nearest bin center."""
        idx = np.digitize(value, bins) - 1
        idx = np.clip(idx, 0, len(bins) - 1)
        return float(bins[idx])

    def state_to_key(self, state):
        """
        Creates a unique, hashable key for tabular lookup.
        State format: (bw, ct, layer, prev_action, surplus, negative_surplus_count)
        """
        bw, ct, layer, prev_action, surplus, negative_surplus_count = state
        
        # --- Apply discretization ---
        bw_disc = self.discretize(bw, self.bandwidth_bins)
        ct_disc = self.discretize(ct, self.cloudtime_bins)
        surplus_disc = self.discretize(surplus, self.surplus_bins)

        # Convert previous action to key
        prev_action_key = self.action_to_key_part(prev_action)

        # Return composite state key
        return (bw_disc, ct_disc, int(layer), prev_action_key, surplus_disc, int(negative_surplus_count))

    def get_possible_actions(self, layer):
        nodes = self.profiling.get_num_nodes(layer)

        # ✅ If last layer → force all nodes to run on edge (action = 0)
        if layer == len(self.profiling.layers) - 1:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer  # layer index
            a[:, 1] = 0      # edge only
            return [a]

        # Otherwise compute all binary offload patterns
        actions = []
        for pattern in range(2 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            for i in range(nodes):
                a[i, 1] = (pattern >> i) & 1
            actions.append(a)

        return actions

    def get_action(self, state):
        state_key = self.state_to_key(state)
        layer = int(state[2])
        actions = self.get_possible_actions(layer)

        # ✅ Initialize policy for unseen state (uniform distribution)
        if state_key not in self.policy_table:
            self.policy_table[state_key] = np.ones(len(actions)) / len(actions)

        # ✅ Last layer: return forced action (edge only)
        if layer == len(self.profiling.layers) - 1:
            return actions[0]   # only one valid action exists

        # ✅ ε-greedy exploration (only for non-last layers)
        if random.random() < self.epsilon:
            return random.choice(actions)

        # ✅ Sample from learned policy
        probs = self.policy_table[state_key]
        probs = probs / np.sum(probs)  # normalize
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]


    # ---------- CORE TRAIN FUNCTION ----------
    def train(self, current_state, random_seed =0.0):
        state_key = self.state_to_key(current_state)
        layer = int(current_state[2])
        surplus = current_state[4]

        # select action
        action = self.get_action(current_state)


        next_state_cloud_processing = self.simulator.get_next_state_cloud_waiting_time(
            next_layer = (int(current_state[2])) if ((int(current_state[2]) + 1)  < len(self.profiling.layers)) else int(current_state[2]),
            current_action=action
        )

        # Simulator step(s)
        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state, current_action=action, cloud_pending_ms=next_state_cloud_processing
        )

        # Reward computation (simulator returns scaled reward)
        reward, surplus, negative_surplus_count, fractional_deadline = self.simulator.calculate_reward(
            int(current_state[2]), energy, completion_time_s, current_state[4], current_state[5], isA2C=False
        )
        surplus /= 1000.0  # convert to seconds

        # Next state from simulator
        next_state, terminal, _ = self.simulator.get_next_state(
            current_state, action, surplus, negative_surplus_count, new_cloud_pending=next_state_cloud_processing
        )

        

        # critic update
        next_key = self.state_to_key(next_state)
        v_s = self.value_table.get(state_key, 0.0)
        v_next = self.value_table.get(next_key, 0.0)
        delta = reward + (0 if terminal else self.gamma * v_next) - v_s
        self.value_table[state_key] = v_s + self.alpha_v * delta

        # actor update
        actions = self.get_possible_actions(layer)
        if state_key not in self.policy_table:
            self.policy_table[state_key] = np.ones(len(actions)) / len(actions)

        action_idx = next(i for i, a in enumerate(actions) if np.array_equal(a, action))
        probs = self.policy_table[state_key]
        probs[action_idx] += self.alpha_p * delta
        probs = np.maximum(probs, 1e-6)
        probs /= np.sum(probs)
        self.policy_table[state_key] = probs

        return action, reward, next_state, terminal, energy, completion_time_s

    # ---------- SAVE / LOAD ----------
    def save_tables(self):
        try:
            np.save(self.filename_value, self.value_table, allow_pickle=True)
            np.save(self.filename_policy, self.policy_table, allow_pickle=True)
            # print("Tables saved successfully.")
        except Exception as e:
            print(f"Error saving A2C tables: {e}")
            # print(f"Error saving tables: {e}")

    def load_tables(self):
        try:
            if os.path.exists(self.filename_value) and os.path.exists(self.filename_policy):
                self.value_table = np.load(self.filename_value, allow_pickle=True).item()
                self.policy_table = np.load(self.filename_policy, allow_pickle=True).item()
                print("Loaded existing A2C tables.")
            else:
                print("No A2C tables found. Starting fresh.")
        except Exception as e:
            print(f"Error loading A2C tables: {e}. Starting fresh.")
