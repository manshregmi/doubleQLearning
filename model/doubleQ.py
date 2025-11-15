import numpy as np
import random
import pickle
import os
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


class DoubleQLearningAgent:
    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test: bool,
        alpha: float = 0.02,
        gamma: float = 0.95,
        epsilon: float = 0.3,
    ):
        """
        Double Q-learning agent for layer-by-layer offloading decisions.
        """
        self.profiling = profiling_data
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.is_test = is_test

        # Q-tables
        self.Q1 = {}
        self.Q2 = {}

        # Simulator
        self.simulator = CloudEdgeSimulator(profiling_data)

        # --- Discretization bins ---
        self.bandwidth_bins = np.linspace(1, 15, 6)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        self.surplus_bins = np.linspace(-300, 300, 100)

        # --- Exploration & visit-counts ---
        self.visit_counts = {}              # (s_key, a_key) -> visits
        self.beta = 0.5                     # exploration-bonus coefficient

        # --- Adaptive epsilon scheduling ---
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995         # slow decay per episode
        self.epsilon_boost = 0.6            # added when stuck
        self.stagnant_limit = 200           # episodes without improvement

        # Episode performance tracking
        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0

        # Optimistic initialization
        self.optimistic_init_value = 1.0

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])

    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1].tolist())

    def _state_to_key(self, state):
        bw, ctime, layer, prev_action, surplus, neg_surplus_count = state
        bw_d = self._discretize(float(bw), self.bandwidth_bins)
        ct_d = self._discretize(float(ctime), self.cloudtime_bins)
        sp_d = self._discretize(float(surplus), self.surplus_bins)
        layer_i = int(layer)
        neg_i = int(neg_surplus_count)
        prev_key = None if prev_action is None else self._action_to_key(prev_action)
        return (bw_d, ct_d, layer_i, sp_d, neg_i, prev_key)

    def _argmax_with_tiebreak(self, values):
        max_val = max(values)
        idxs = [i for i, v in enumerate(values) if np.isclose(v, max_val)]
        return random.choice(idxs)

    # ------------------------------------------------------------
    # Possible full actions
    # ------------------------------------------------------------
    def _get_possible_full_actions(self):
        layer_node_counts = [self.profiling.get_num_nodes(i)
                             for i in range(len(self.profiling.layers))]
        total_nodes = sum(layer_node_counts)

        all_patterns = []
        max_combos = 2 ** total_nodes

        if max_combos > 5000:  # sample when too large
            for _ in range(1000):
                plan, offset = [], 0
                for layer_idx, n in enumerate(layer_node_counts):
                    a = np.zeros((n, 2), dtype=int)
                    a[:, 0] = layer_idx
                    a[:, 1] = np.random.randint(0, 2, size=n)
                    plan.append(a)
                all_patterns.append(plan)
        else:
            for pattern in range(max_combos):
                bits = list(map(int, format(pattern, f"0{total_nodes}b")))
                plan = []
                offset = 0
                for layer_idx, n in enumerate(layer_node_counts):
                    a = np.zeros((n, 2), dtype=int)
                    a[:, 0] = layer_idx
                    a[:, 1] = bits[offset: offset + n]
                    offset += n
                    plan.append(a)
                all_patterns.append(plan)

        return all_patterns

    # ------------------------------------------------------------
    # Action selection (full-task)
    # ------------------------------------------------------------
    def choose_action_all(self, state):
        actions = self._get_possible_full_actions()
        s_key = self._state_to_key(state)

        # ε-greedy exploration
        if (not self.is_test) and (random.random() < self.epsilon):
            return random.choice(actions)

        q_vals = []
        for plan in actions:
            a_key = tuple([tuple(x[:, 1].tolist()) for x in plan])
            full_key = (s_key, a_key)

            # base Q-values (with optimistic init)
            q1 = self.Q1.get(full_key, self.optimistic_init_value)
            q2 = self.Q2.get(full_key, self.optimistic_init_value)
            base_q = q1 + q2

            # visit-count bonus
            visits = self.visit_counts.get(full_key, 0)
            bonus = self.beta / np.sqrt(visits + 1)

            q_vals.append(base_q + bonus)

        if self.is_test:
            idx = int(np.argmax(q_vals))
        else:
            idx = self._argmax_with_tiebreak(q_vals)

        return actions[idx]

    # ------------------------------------------------------------
    # Training (full-task)
    # ------------------------------------------------------------
    def handle_ndim_action(self, current_action):
        if current_action is None:
            return np.array([])
        if isinstance(current_action, (int, np.integer)):
            return np.array([current_action])
        if isinstance(current_action, np.ndarray):
            if current_action.ndim == 1:
                return current_action
            return current_action[:, 1]
        raise ValueError(f"Unexpected action type {type(current_action)}")

    def train_all(self, state):
        # choose full-task plan
        action_plan = self.choose_action_all(state)

        # run full simulation
        initial_bw = state[0]
        total_energy, total_time, reward, bw = self.simulator.run_full_task(
            action_plan, initial_bw
        )

        # terminal episode (single-step)
        cur_key = (
            self._state_to_key(state),
            tuple([tuple(self.handle_ndim_action(x).tolist()) for x in action_plan])
        )

        # --- visit count increment ---
        self.visit_counts[cur_key] = self.visit_counts.get(cur_key, 0) + 1

        # --- optimistic init ---
        if cur_key not in self.Q1:
            self.Q1[cur_key] = self.optimistic_init_value
        if cur_key not in self.Q2:
            self.Q2[cur_key] = self.optimistic_init_value

        # --- Double Q update ---
        if random.random() < 0.5:
            q_table, q_eval = self.Q1, self.Q2
        else:
            q_table, q_eval = self.Q2, self.Q1

        old = q_table.get(cur_key, self.optimistic_init_value)
        q_table[cur_key] = old + self.alpha * (reward - old)

        # --- epsilon decay ---
        if not self.is_test:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return reward, total_energy, total_time, bw

    # ------------------------------------------------------------
    # Episode end (ε-boost logic)
    # ------------------------------------------------------------
    def notify_episode_end(self, episode_reward):
        # improvement?
        if episode_reward > self.best_episode_reward:
            self.best_episode_reward = episode_reward
            self.episodes_since_improvement = 0
        else:
            self.episodes_since_improvement += 1

        # if stuck -> boost epsilon
        if self.episodes_since_improvement >= self.stagnant_limit:
            self.epsilon = min(1.0, self.epsilon + self.epsilon_boost)
            self.episodes_since_improvement = 0

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------
    def save_qtables(self, filename="q_tables.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump((self.Q1, self.Q2), f)
        except Exception as e:
            print("Error saving:", e)

    def load_qtables(self, filename="q_tables.pkl"):
        try:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    self.Q1, self.Q2 = pickle.load(f)
                print(f"Loaded Q-tables (Q1={len(self.Q1)}, Q2={len(self.Q2)})")
                return True
            print("No Q-table found, starting empty.")
            return False
        except Exception as e:
            print("Error loading:", e)
            return False
