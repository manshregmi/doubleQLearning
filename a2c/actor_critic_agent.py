import numpy as np
import random
import pickle
import os
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


class TabularActorCriticAgent:
    def __init__(self, profiling_data, is_test=False,
                 alpha_actor=0.02, alpha_critic=0.05, gamma=0.95,
                 trace_csv_path=None,
                 timeout_threshold_ms=150.0,
                 packet_loss_prob=0.10):
        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        self.policy_table = {}
        self.value_table = {}

        self.simulator = CloudEdgeSimulator(
            profiling_data,
            trace_csv_path=trace_csv_path,
            timeout_threshold_ms=timeout_threshold_ms,
            packet_loss_prob=packet_loss_prob,
        )

        # Discretization bins
        self.bandwidth_bins = np.linspace(0.5, 30, 40)
        self.rtt_bins = np.linspace(0, 100, 20)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        self.surplus_bins = np.linspace(-100, 100, 20)
        self.chain_bins = np.linspace(0, 7, 8)
        self.timeout_bins = np.linspace(0, 400, 9)  # for timeout threshold

        # Exploration (temperature-based softmax)
        self.temperature = 1.0
        self.temperature_min = 0.25
        self.temperature_max = 2.0
        self.temperature_decay = 0.9995
        self.temperature_boost = 1.5  # multiply UP for boost

        # epsilon floor
        self.epsilon_min = 0.05

        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0
        self.stagnant_limit = 5000

        self.total_episodes = 0
        self.edge_execution_counts = {}
        self.optimistic_execution_counts = {}
        self.conservative_execution_counts = {}

    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])

    def _timeout_bin(self):
        return int(np.digitize(
            [self.simulator.timeout_threshold_ms],
            self.timeout_bins, right=True
        )[0])

    def _state_to_key(self, state):
        # state = (bw, rtt, ctime, layer, prev_action, surplus, neg_count, chain_len, timeout_bin)
        bw, rtt, ctime, layer, prev_action, surplus, neg_count, chain_len, t_bin = state
        prev_key = tuple(int(x) for x in prev_action) if prev_action is not None else (-1,)
        # Hash prev_key to small int to avoid huge table
        prev_hash = hash(prev_key) % 1000
        return (
            self._discretize(float(bw), self.bandwidth_bins),
            self._discretize(float(rtt), self.rtt_bins),
            self._discretize(float(ctime), self.cloudtime_bins),
            int(layer),
            int(prev_hash),
            self._discretize(float(surplus), self.surplus_bins),
            int(neg_count),
            int(chain_len),
            int(t_bin),
        )

    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1])

    def _get_possible_actions(self, layer_idx):
        nodes = self.profiling.get_num_nodes(layer_idx)
        # Enumerate all ternary patterns
        actions = []
        for pattern in range(3 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            temp = pattern
            for i in range(nodes):
                a[i, 1] = temp % 3
                temp //= 3
            actions.append(a)
        return actions

    def choose_action(self, state):
        layer = int(state[3])
        actions = self._get_possible_actions(layer)
        s_key = self._state_to_key(state)

        if not self.is_test and random.random() < self.epsilon_min:
            return random.choice(actions)

        prefs = np.array([
            self.policy_table.get((s_key, self._action_to_key(a)), 0.0)
            for a in actions
        ])
        prefs = prefs / max(self.temperature, 1e-6)
        prefs -= np.max(prefs)
        probs = np.exp(prefs)
        probs /= np.sum(probs)

        if self.is_test:
            return actions[int(np.argmax(probs))]
        return actions[np.random.choice(len(actions), p=probs)]

    def train(self, current_state):
        action = self.choose_action(current_state)
        current_layer = int(current_state[3])

        next_cloud = self.simulator.get_next_state_cloud_waiting_time(
            current_layer=current_layer,
            current_action=action,
            isAllCloud=False,
        )

        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state,
            current_action=action,
            cloud_pending_ms=next_cloud,
        )

        reward, surplus, neg_count, fractional_deadline = self.simulator.calculate_reward(
            current_layer,
            energy,
            completion_time_s,
            current_state[5],
            current_state[6],
            isA2C=True,
        )

        next_state, terminal = self.simulator.get_next_state(
            current_state,
            action,
            new_cloud_pending=next_cloud,
            surplus=surplus,
            neg_count=neg_count,
        )

        self.track_action_execution(action, current_layer)

        return (action, reward, next_state, terminal, energy,
                completion_time_s, next_state[0], surplus,
                fractional_deadline, neg_count)

    def update_trajectory(self, trajectory):
        G = 0.0
        for step in reversed(trajectory):
            G = step["reward"] + self.gamma * G
            G = float(np.clip(G, -1000.0, 1000.0))

            s = step["state_key"]
            a = step["action_key"]

            V = self.value_table.get(s, 0.0)
            advantage = G - V

            self.value_table[s] = V + self.alpha_critic * advantage
            self.policy_table[(s, a)] = float(np.clip(
                self.policy_table.get((s, a), 0.0) + self.alpha_actor * advantage,
                -50.0, 50.0,
            ))

    def track_action_execution(self, action, layer):
        for node_idx, (_, location) in enumerate(action):
            key = (layer, node_idx)
            if location == 0:
                self.edge_execution_counts[key] = self.edge_execution_counts.get(key, 0) + 1
            elif location == 1:
                self.optimistic_execution_counts[key] = self.optimistic_execution_counts.get(key, 0) + 1
            else:
                self.conservative_execution_counts[key] = self.conservative_execution_counts.get(key, 0) + 1

    def get_execution_stats(self):
        return {
            'edge_counts': self.edge_execution_counts,
            'optimistic_counts': self.optimistic_execution_counts,
            'conservative_counts': self.conservative_execution_counts,
            'total_episodes': self.total_episodes,
        }

    def notify_episode_end(self, episode_reward):
        self.total_episodes += 1
        if episode_reward > self.best_episode_reward + 1e-6:
            self.best_episode_reward = episode_reward
            self.episodes_since_improvement = 0
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
        else:
            self.episodes_since_improvement += 1
            if self.episodes_since_improvement >= self.stagnant_limit:
                self.temperature = min(self.temperature_max,
                                       self.temperature * self.temperature_boost)
                self.episodes_since_improvement = 0
                print(f"  [Boost] Temperature -> {self.temperature:.2f}")
            else:
                self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)

    def save(self, file="a2c_tables.pkl"):
        with open(file, "wb") as f:
            pickle.dump((self.policy_table, self.value_table), f)

    def load(self, file="a2c_tables.pkl"):
        if os.path.exists(file):
            with open(file, "rb") as f:
                self.policy_table, self.value_table = pickle.load(f)