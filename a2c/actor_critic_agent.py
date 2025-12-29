import numpy as np
import random
import pickle
import os
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


class TabularActorCriticAgent:
    """
    Tabular Actor–Critic with:
    - Double-Q–equivalent logic
    - Stable state abstraction
    - ε-style exploration
    - Temperature-based policy smoothing
    """

    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test=False,
        alpha_actor=0.02,
        alpha_critic=0.05,
        gamma=0.95,
    ):
        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma

        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        self.policy_table = {}   # (state_key, action_key) → preference
        self.value_table = {}    # state_key → V(s)

        self.simulator = CloudEdgeSimulator(profiling_data)

        # SAME discretization scale as Double-Q
        self.bandwidth_bins = np.linspace(1, 15, 60)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        self.surplus_bins = np.linspace(-25, 25, 25)

        # Exploration control (ε-equivalent)
        self.temperature = 1.0
        self.temperature_min = 0.25
        self.temperature_decay = 0.999
        self.temperature_boost = 1.4

        self.epsilon_min = 0.05   # hard ε-floor

        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0
        self.stagnant_limit = 2500

    # ======================================================
    # Discretization
    # ======================================================
    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])

    # ======================================================
    # ✅ FIXED STATE (NO prev_action)
    # ======================================================
    def _state_to_key(self, state):
        bw, ctime, layer, _, surplus, neg_count = state
        return (
            self._discretize(float(bw), self.bandwidth_bins),
            self._discretize(float(ctime), self.cloudtime_bins),
            int(layer),
            self._discretize(float(surplus), self.surplus_bins),
            int(neg_count),
        )

    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1])

    # ======================================================
    # Action Space (IDENTICAL to Double-Q)
    # ======================================================
    def _get_possible_actions(self, layer_idx):
        nodes = self.profiling.get_num_nodes(layer_idx)

        if layer_idx == len(self.profiling.layers) - 1:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            return [a]

        max_patterns = 64
        patterns = list(range(2 ** nodes))
        if len(patterns) > max_patterns:
            patterns = random.sample(patterns, max_patterns)

        actions = []
        for p in patterns:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            for i in range(nodes):
                a[i, 1] = (p >> i) & 1
            actions.append(a)
        return actions

    # ======================================================
    # ✅ ACTION SELECTION (ε + softmax)
    # ======================================================
    def choose_action(self, state):
        layer = int(state[2])
        actions = self._get_possible_actions(layer)
        s_key = self._state_to_key(state)

        # ε-style forced exploration
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

    # ======================================================
    # Environment step (IDENTICAL)
    # ======================================================
    def train(self, current_state):

        action = self.choose_action(current_state)

        next_cloud = self.simulator.get_next_state_cloud_waiting_time(
            next_layer=min(int(current_state[2]) + 1, len(self.profiling.layers) - 1),
            current_action=action,
            isAllCloud=False,
        )

        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state,
            current_action=action,
            cloud_pending_ms=next_cloud,
        )

        reward, surplus, neg_count, fractional_deadline = \
            self.simulator.calculate_reward(
                int(current_state[2]),
                energy,
                completion_time_s,
                current_state[4],
                current_state[5],
                isA2C=True,
            )

        next_state, terminal, _ = self.simulator.get_next_state(
            current_state,
            action,
            surplus,
            neg_count,
            new_cloud_pending=next_cloud,
        )

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
    # Monte-Carlo Actor–Critic Update
    # ======================================================
    def update_trajectory(self, trajectory):
        G = 0.0
        for step in reversed(trajectory):
            G = step["reward"] + self.gamma * G
            G = np.clip(G, -1000.0, 1000.0)

            s = step["state_key"]
            a = step["action_key"]

            V = self.value_table.get(s, 0.0)
            advantage = G - V

            self.value_table[s] = V + self.alpha_critic * advantage
            self.policy_table[(s, a)] = np.clip(
                self.policy_table.get((s, a), 0.0)
                + self.alpha_actor * advantage,
                -50.0, 50.0,
            )

    # ======================================================
    # Episode-level exploration control
    # ======================================================
    def notify_episode_end(self, episode_reward):
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

    # ======================================================
    # Persistence
    # ======================================================
    def save(self, file="a2c_tables.pkl"):
        with open(file, "wb") as f:
            pickle.dump((self.policy_table, self.value_table), f)

    def load(self, file="a2c_tables.pkl"):
        if os.path.exists(file):
            with open(file, "rb") as f:
                self.policy_table, self.value_table = pickle.load(f)
