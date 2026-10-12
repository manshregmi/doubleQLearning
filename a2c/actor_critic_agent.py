import numpy as np
import random
import pickle
import os
import itertools
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


ACTION_SPACES = {
    "ternary":      (0, 1, 2),
    "optimistic":   (0, 1),
    "conservative": (0, 2),
}


class TabularActorCriticAgent:
    """Tabular A2C over a per-node mode action space.

    Reward (paper):
        R_imm   = -eta * E                              (Eq. 10, eta = 150)
        R_final = R_imm - phi * max(0, delta - tau)/tau   (Eq. 11, runner)

    State (10-tuple):
        (bw, rtt, ctime, layer, prev_action, surplus,
         neg_count, chain_len, d_idx, recent_ack_loss)
    """

    def __init__(self, profiling_data, is_test=False,
                 alpha_actor=0.01, alpha_critic=0.01, gamma=0.95,
                 trace_csv_path=None,
                 action_modes=(0, 1, 2),
                 deadline_levels=(500.0,),
                 verbose=False):
        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        self.deadline_levels = np.asarray(sorted(deadline_levels), dtype=float)

        self.action_modes = tuple(int(m) for m in action_modes)
        if 0 not in self.action_modes:
            raise ValueError("action_modes must include 0 (local execution).")
        self._action_cache = {}

        self.policy_table = {}
        self.value_table = {}

        self.eval_decisions = 0
        self.eval_unseen_states = 0

        self.simulator = CloudEdgeSimulator(
            profiling_data,
            trace_csv_path=trace_csv_path,
        )
        self.simulator.verbose = verbose

        self.bandwidth_bins = np.linspace(0.5, 30, 40)
        self.rtt_bins = np.linspace(1.0, 6.0, 20)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        self.surplus_frac_bins = np.linspace(-0.3, 0.3, 24)

        self.temperature = 1.0
        self.temperature_min = 0.25
        self.temperature_max = 2.0
        self.temperature_decay = 0.9995
        self.temperature_boost = 1.5

        self.epsilon_min = 0.05

        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0
        self.stagnant_limit = 5000

        self.total_episodes = 0
        self.edge_execution_counts = {}
        self.optimistic_execution_counts = {}
        self.conservative_execution_counts = {}

    def context_bin(self, deadline_ms):
        return int(np.argmin(np.abs(self.deadline_levels - float(deadline_ms))))

    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])

    def _state_to_key(self, state):
        (bw, rtt, ctime, layer, prev_action, surplus,
         neg_count, chain_len, d_idx, recent_ack_loss) = state

        prev_key = tuple(int(x) for x in prev_action) if prev_action is not None else (-1,)
        prev_hash = hash(prev_key) % 1000

        deadline = float(self.deadline_levels[int(d_idx)])
        surplus_frac = float(surplus) / max(deadline, 1e-9)

        key = (
            self._discretize(float(bw), self.bandwidth_bins),
            self._discretize(float(rtt), self.rtt_bins),
            self._discretize(float(ctime), self.cloudtime_bins),
            int(layer),
            int(prev_hash),
            self._discretize(surplus_frac, self.surplus_frac_bins),
            int(neg_count),
            int(chain_len),
            int(d_idx),
            int(recent_ack_loss),
        )
        return key

    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1])

    def _get_possible_actions(self, layer_idx):
        if layer_idx in self._action_cache:
            return self._action_cache[layer_idx]
        nodes = self.profiling.get_num_nodes(layer_idx)
        actions = []
        for pattern in itertools.product(self.action_modes, repeat=nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            a[:, 1] = pattern
            actions.append(a)
        self._action_cache[layer_idx] = actions
        return actions

    def choose_action(self, state):
        layer = int(state[3])
        actions = self._get_possible_actions(layer)
        s_key = self._state_to_key(state)

        if not self.is_test and random.random() < self.epsilon_min:
            return random.choice(actions)

        action_keys = [self._action_to_key(a) for a in actions]

        if self.is_test:
            self.eval_decisions += 1
            if not any((s_key, ak) in self.policy_table for ak in action_keys):
                self.eval_unseen_states += 1

        prefs = np.array([self.policy_table.get((s_key, ak), 0.0) for ak in action_keys])
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
            G = float(np.clip(G, -3000.0, 500.0))
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
            pickle.dump({"policy": self.policy_table, "value": self.value_table}, f)

    def load(self, file="a2c_tables.pkl"):
        if os.path.exists(file):
            with open(file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.policy_table = data.get("policy", {})
                self.value_table = data.get("value", {})
            else:
                self.policy_table, self.value_table = data