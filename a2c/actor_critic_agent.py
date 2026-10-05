import numpy as np
import random
import pickle
import os
import itertools
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


# Predefined action spaces (per-node modes)
#   0 = local (user/edge device)
#   1 = optimistic cloudlet  (result download deferred)
#   2 = conservative cloudlet (result download immediate)
ACTION_SPACES = {
    "ternary":      (0, 1, 2),
    "optimistic":   (0, 1),
    "conservative": (0, 2),
}


class TabularActorCriticAgent:
    """Tabular A2C over a per-node mode action space.

    Reward follows the paper's constrained formulation:
        R_imm  = -eta * E                      (Eq. 10, eta = 150)
        R_final = R_imm - phi * max(0, delta - tau) / tau  on violation (Eq. 11)
    The terminal penalty is applied only in the runner, so this agent just
    consumes rewards exactly as they arrive in the trajectory.

    The last slot of the environment state is a context tuple
    (deadline_idx, ratio_idx): the index of the episode's deadline in
    `deadline_levels` and of its timeout ratio (timeout = ratio * deadline)
    in `threshold_ratios`.

    Every agent observes the deadline. Only a threshold-aware agent also
    observes the timeout ratio.
    """

    def __init__(self, profiling_data, is_test=False,
                 alpha_actor=0.01, alpha_critic=0.01, gamma=0.95,
                 trace_csv_path=None,
                 timeout_threshold_ms=150.0,
                 packet_loss_prob=0.10,
                 action_modes=(0, 1, 2),
                 threshold_aware=True,
                 deadline_levels=(500.0,),
                 threshold_ratios=(0.10,),
                 verbose=False):
        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        self.threshold_aware = bool(threshold_aware)
        self.deadline_levels = np.asarray(sorted(deadline_levels), dtype=float)
        self.threshold_ratios = np.asarray(sorted(threshold_ratios), dtype=float)

        # Allowed per-node modes for this agent
        self.action_modes = tuple(int(m) for m in action_modes)
        if 0 not in self.action_modes:
            raise ValueError("action_modes must include 0 (local execution).")
        self._action_cache = {}

        self.policy_table = {}
        self.value_table = {}

        # Evaluation diagnostics
        self.eval_decisions = 0
        self.eval_unseen_states = 0

        self.simulator = CloudEdgeSimulator(
            profiling_data,
            trace_csv_path=trace_csv_path,
            timeout_threshold_ms=timeout_threshold_ms,
            packet_loss_prob=packet_loss_prob,
        )
        self.simulator.verbose = verbose

        # Discretization bins
        self.bandwidth_bins = np.linspace(0.5, 30, 40)
        self.rtt_bins = np.linspace(0, 100, 20)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        # Surplus as a FRACTION of the deadline (comparable across deadlines)
        self.surplus_frac_bins = np.linspace(-0.3, 0.3, 24)

        # Exploration (temperature-based softmax)
        self.temperature = 1.0
        self.temperature_min = 0.25
        self.temperature_max = 2.0
        self.temperature_decay = 0.9995
        self.temperature_boost = 1.5

        # epsilon floor
        self.epsilon_min = 0.05

        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0
        self.stagnant_limit = 5000

        self.total_episodes = 0
        self.edge_execution_counts = {}
        self.optimistic_execution_counts = {}
        self.conservative_execution_counts = {}

    # ------------------------------------------------------------------
    # Context / state keys
    # ------------------------------------------------------------------
    def context_bin(self, deadline_ms, ratio):
        """(deadline_idx, ratio_idx) using the nearest configured level."""
        d_idx = int(np.argmin(np.abs(self.deadline_levels - float(deadline_ms))))
        r_idx = int(np.argmin(np.abs(self.threshold_ratios - float(ratio))))
        return (d_idx, r_idx)

    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])

    def _state_to_key(self, state):
        # state = (bw, rtt, ctime, layer, prev_action, surplus, neg_count, chain_len, context)
        bw, rtt, ctime, layer, prev_action, surplus, neg_count, chain_len, context = state
        d_idx, r_idx = context

        prev_key = tuple(int(x) for x in prev_action) if prev_action is not None else (-1,)
        prev_hash = hash(prev_key) % 1000

        deadline = float(self.deadline_levels[d_idx])
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
            int(d_idx),                      # every agent knows the deadline
        )
        if self.threshold_aware:
            key = key + (int(r_idx),)        # only threshold-aware agents see the ratio
        return key

    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1])

    def _get_possible_actions(self, layer_idx):
        """All per-node assignments using only this agent's allowed modes."""
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

    # ------------------------------------------------------------------
    # Acting / learning
    # ------------------------------------------------------------------
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
                # Never trained on this state -> argmax of all-zero prefs = all-local
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

        # Must run BEFORE get_next_state (it sets last_timeout_occurred).
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
            # Room for a few hundred J at eta=150 plus a phi=1000 terminal penalty
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
            else:  # legacy (policy, value) tuple
                self.policy_table, self.value_table = data