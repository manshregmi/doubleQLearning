import numpy as np
import os
import random
from simulator.simulator import CloudEdgeSimulator

class A2CAgent:
    def __init__(
        self,
        profiling_data,
        alpha_v=0.02,
        alpha_p=0.02,
        gamma=0.95,
        epsilon=1,
        is_test=False,
        BW_bins=15,
        CT_bins=20,
        surplus_bins=25,
    ):
        self.profiling = profiling_data
        self.alpha_v = alpha_v
        self.alpha_p = alpha_p
        self.gamma = gamma
        self.epsilon = epsilon
        self.is_test = is_test
        self.simulator = CloudEdgeSimulator(profiling_data)

        # Tables
        self.value_table = {}     # V(s)
        self.policy_table = {}    # π(s,a)

        self.filename_value = "value_table.npy"
        self.filename_policy = "policy_table.npy"

        # Discretization bins
        self.bandwidth_bins = np.linspace(1, 15, BW_bins)
        self.cloudtime_bins = np.linspace(0, 100, CT_bins)
        self.surplus_bins = np.linspace(-25, 25, surplus_bins)

        # -------------------------------
        # ε-GREEDY EXPLORATION CONTROL
        # -------------------------------
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995        # slow decay per step
        self.epsilon_boost = 0.6           # boost when stuck
        self.stagnant_limit = 5000         # step threshold for stuck
        self.best_reward = -1e9            # track best reward
        self.stagnant = 0                  # stagnant step counter

    # ==========================================================
    # STATE / ACTION HELPERS
    # ==========================================================
    def action_to_key_part(self, action_matrix):
        if action_matrix is None:
            return tuple([-1, -1]) 
        return tuple(action_matrix.flatten().tolist())

    def discretize(self, value, bins):
        idx = np.digitize(value, bins) - 1
        idx = np.clip(idx, 0, len(bins) - 1)
        return float(bins[idx])

    def state_to_key(self, state):
        bw, ct, layer, prev_action, surplus, negative_surplus_count = state
        bw_disc = self.discretize(bw, self.bandwidth_bins)
        ct_disc = self.discretize(ct, self.cloudtime_bins)
        surplus_disc = self.discretize(surplus, self.surplus_bins)
        prev_action_key = self.action_to_key_part(prev_action)
        return (
            bw_disc,
            ct_disc,
            int(layer),
            prev_action_key,
            surplus_disc,
            int(negative_surplus_count),
        )

    def get_possible_actions(self, layer):
        nodes = self.profiling.get_num_nodes(layer)

        # Last layer = must execute locally
        if layer == len(self.profiling.layers) - 1:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            a[:, 1] = 0
            return [a]

        actions = []
        for pattern in range(2 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            for i in range(nodes):
                a[i, 1] = (pattern >> i) & 1
            actions.append(a)

        return actions

    # ==========================================================
    # ACTION SELECTION WITH ε-GREEDY EXPLORATION
    # ==========================================================
    def get_action(self, state):
        state_key = self.state_to_key(state)
        layer = int(state[2])
        actions = self.get_possible_actions(layer)

        # Initialize on first visit
        if state_key not in self.policy_table:
            self.policy_table[state_key] = np.ones(len(actions)) / len(actions)

        probs = self.policy_table[state_key]
        probs = probs / np.sum(probs)

        # During test → greedy
        if self.is_test:
            idx = np.argmax(probs)
            return actions[idx]

        # Training → ε-greedy
        if random.random() < self.epsilon:
            return random.choice(actions)

        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    # ==========================================================
    # CORE TRAIN FUNCTION
    # ==========================================================
    def train(self, current_state, random_seed=0.0):
        state_key = self.state_to_key(current_state)
        layer = int(current_state[2])

        # Choose action
        action = self.get_action(current_state)

        # Simulator calls
        next_state_cloud_processing = self.simulator.get_next_state_cloud_waiting_time(
            next_layer=(
                int(current_state[2])
                if ((int(current_state[2]) + 1) < len(self.profiling.layers))
                else int(current_state[2])
            ),
            current_action=action
        )

        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state,
            current_action=action,
            cloud_pending_ms=next_state_cloud_processing,
        )

        reward, surplus, negative_surplus_count, fractional_deadline = (
            self.simulator.calculate_reward(
                int(current_state[2]),
                energy,
                completion_time_s,
                current_state[4],
                current_state[5],
                isA2C=True,
            )
        )

        next_state, terminal, _ = self.simulator.get_next_state(
            current_state,
            action,
            surplus,
            negative_surplus_count,
            new_cloud_pending=next_state_cloud_processing,
        )

        # TEST MODE → skip updates
        if self.is_test:
            return action, reward, next_state, terminal, energy, completion_time_s

        # ======================================================
        # CRITIC UPDATE: update V(s)
        # ======================================================
        next_key = self.state_to_key(next_state)
        v_s = self.value_table.get(state_key, 0.0)
        v_next = self.value_table.get(next_key, 0.0)

        delta = reward + (0 if terminal else self.gamma * v_next) - v_s
        self.value_table[state_key] = v_s + self.alpha_v * delta

        # ======================================================
        # ACTOR UPDATE: update π(s,a)
        # ======================================================
        actions = self.get_possible_actions(layer)
        if state_key not in self.policy_table:
            self.policy_table[state_key] = np.ones(len(actions)) / len(actions)

        action_idx = next(i for i, a in enumerate(actions) if np.array_equal(a, action))

        probs = self.policy_table[state_key]
        probs[action_idx] += self.alpha_p * delta
        probs = np.maximum(probs, 1e-6)
        probs /= np.sum(probs)
        self.policy_table[state_key] = probs

        # ======================================================
        # ε-DECAY + BOOST WHEN STUCK
        # ======================================================
        self.stagnant += 1
        if reward > self.best_reward:
            self.best_reward = reward
            self.stagnant = 0

        # Slow decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Boost if stuck too long
        if self.stagnant > self.stagnant_limit:
            self.epsilon = min(1.0, self.epsilon + self.epsilon_boost)
            self.stagnant = 0

        return action, reward, next_state, terminal, energy, completion_time_s

    # ==========================================================
    # SAVE / LOAD
    # ==========================================================
    def save_tables(self):
        try:
            np.save(self.filename_value, self.value_table, allow_pickle=True)
            np.save(self.filename_policy, self.policy_table, allow_pickle=True)
        except Exception as e:
            print(f"Error saving A2C tables: {e}")

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
