import numpy as np
import os
import random
from simulator.simulator import CloudEdgeSimulator

class A2CAgent:
    def __init__(self, profiling_data, alpha_v=0.1, alpha_p=0.1, gamma=0.9, epsilon=0.1):
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

    # ---------- STATE / ACTION HANDLING ----------
    def state_to_key(self, state):
        bw, ct, layer, _, surplus, negative_surplus_count = state
        return (round(bw, 1), round(ct, -1), int(layer), round(surplus, 1), int(negative_surplus_count))

    def get_possible_actions(self, layer):
        nodes = self.profiling.get_num_nodes(layer)
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

        if random.random() < self.epsilon or state_key not in self.policy_table:
            return random.choice(actions)

        probs = self.policy_table[state_key]
        probs /= np.sum(probs)
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    # ---------- CORE TRAIN FUNCTION ----------
    def train(self, current_state):
        state_key = self.state_to_key(current_state)
        layer = int(current_state[2])
        surplus = current_state[4]

        # select action
        action = self.get_action(current_state)

        # simulate
        total_energy, completion_time_s = self.simulator.compute_energy_and_time(current_state, action, current_state[1])
        reward, new_surplus, negative_surplus_count = self.simulator.calculate_reward(layer, total_energy, completion_time_s, surplus, current_state[5])
        next_state, terminal, _ = self.simulator.get_next_state(current_state, action, new_surplus, negative_surplus_count)

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

        return action, reward, next_state, terminal, total_energy, completion_time_s

    # ---------- SAVE / LOAD ----------
    def save_tables(self):
        np.save(self.filename_value, self.value_table, allow_pickle=True)
        np.save(self.filename_policy, self.policy_table, allow_pickle=True)
        print("Tables saved successfully.")

    def load_tables(self):
        if os.path.exists(self.filename_value) and os.path.exists(self.filename_policy):
            self.value_table = np.load(self.filename_value, allow_pickle=True).item()
            self.policy_table = np.load(self.filename_policy, allow_pickle=True).item()
            print("Loaded existing A2C tables.")
        else:
            print("No A2C tables found. Starting fresh.")
