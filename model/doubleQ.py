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
        self.profiling = profiling_data
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.is_test = is_test

        # Q tables
        self.Q1 = {}
        self.Q2 = {}

        # Simulator
        self.simulator = CloudEdgeSimulator(profiling_data)

        # Discretization
        self.bandwidth_bins = np.linspace(1, 15, 6)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        self.surplus_bins = np.linspace(-300, 300, 100)

        # Visit counts & exploration bonus
        self.visit_counts = {}
        self.beta = 0.5

        # Epsilon scheduling
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.epsilon_boost = 0.6
        self.stagnant_limit = 200

        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0

        # Optimistic init
        self.optimistic_init_value = 1.0

    # ----------------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------------
    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins)-1))])

    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1].tolist())

    def handle_ndim_action(self, current_action):
        if current_action is None:
            return np.array([])
        if isinstance(current_action, (int, np.integer)):
            return np.array([int(current_action)])
        if isinstance(current_action, np.ndarray):
            if current_action.ndim == 1:
                return current_action.astype(int)
            return current_action[:, 1].astype(int)
        raise ValueError(f"Unexpected action type {type(current_action)}")

    def _state_to_key(self, state):
        bw, ctime = state
        bw_d = self._discretize(float(bw), self.bandwidth_bins)
        ct_d = self._discretize(float(ctime), self.cloudtime_bins)
        # sp_d = self._discretize(float(surplus), self.surplus_bins)
        # layer_i = int(layer)
        # neg_i = int(neg_surplus_count)
        # prev_key = None if prev_action is None else self._action_to_key(prev_action)
        return (bw_d, ct_d)

    def _make_full_key(self, state, action_plan):
        s_key = self._state_to_key(state)
        a_key = tuple([
            tuple(self.handle_ndim_action(seg).tolist())
            for seg in action_plan
        ])
        return (s_key, a_key)

    def _argmax_with_tiebreak(self, values):
        m = max(values)
        idxs = [i for i,v in enumerate(values) if np.isclose(v, m)]
        return random.choice(idxs)

    # ----------------------------------------------------------------------
    # Generate all possible complete action plans
    # ----------------------------------------------------------------------
    def _get_possible_full_actions(self):
        layer_node_counts = [self.profiling.get_num_nodes(i)
                             for i in range(len(self.profiling.layers))]
        total_nodes = sum(layer_node_counts)

        all_patterns = []
        max_combos = 2 ** total_nodes

        if max_combos > 5000:
            for _ in range(1000):
                plan = []
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
                    a[:, 1] = bits[offset: offset+n]
                    offset += n
                    plan.append(a)
                all_patterns.append(plan)

        return all_patterns

    # ----------------------------------------------------------------------
    # Choose action (whole plan)
    # ----------------------------------------------------------------------
    def choose_action_all(self, state):
        actions = self._get_possible_full_actions()
        s_key = self._state_to_key(state)

        # epsilon-greedy
        if not self.is_test and random.random() < self.epsilon:
            return random.choice(actions)

        q_vals = []
        for plan in actions:
            a_key = tuple([tuple(self.handle_ndim_action(x).tolist()) for x in plan])
            full_key = (s_key, a_key)

            q1 = self.Q1.get(full_key, self.optimistic_init_value)
            q2 = self.Q2.get(full_key, self.optimistic_init_value)

            visits = self.visit_counts.get(full_key, 0)
            bonus = self.beta / np.sqrt(visits + 1)

            q_vals.append(q1 + q2 + bonus)

        if self.is_test:
            idx = int(np.argmax(q_vals))
        else:
            idx = self._argmax_with_tiebreak(q_vals)

        return actions[idx]

    # ----------------------------------------------------------------------
    # FULL TASK SIMULATION (inside agent)
    # ----------------------------------------------------------------------
    def run_full_task(self, action_plan, initial_bandwidth):
        bandwidth = initial_bandwidth
        cloud_pending_ms = 0.0
        surplus = 0.0
        negative_surplus_count = 0
        total_energy = 0.0
        total_time_ms = 0.0
        total_cloud_pending = 0.0

        for layer_idx, action in enumerate(action_plan):
            current_state = (
                bandwidth,
                cloud_pending_ms,
                layer_idx,
                None if layer_idx == 0 else action_plan[layer_idx - 1],
                surplus,
                negative_surplus_count,
            )

            next_wait = self.simulator.get_next_state_cloud_waiting_time(
                next_layer=layer_idx if (layer_idx+1) < len(self.profiling.layers) else layer_idx,
                current_action=action,
                isAllCloud=False,
            )

            energy, completion_time_s = self.simulator.compute_energy_and_time(
                current_state=current_state,
                current_action=action,
                cloud_pending_ms=next_wait,
            )

            total_energy += energy
            total_time_ms += completion_time_s * 1000.0

            _, surplus, __, _ = self.simulator.calculate_reward(
                layer_idx, energy, completion_time_s, surplus, negative_surplus_count, False
            )

            next_state, terminal, _ = self.simulator.get_next_state(
                current_state, action, 0, 0, new_cloud_pending=next_wait
            )
            total_cloud_pending += next_wait

            if terminal:
                break


        total_reward = self.simulator.compute_whole_action_reward(total_energy, total_time_ms)
        return total_energy, total_time_ms, total_reward, bandwidth, total_cloud_pending

    # ----------------------------------------------------------------------
    # TRAIN ALL
    # ----------------------------------------------------------------------
    def train_all(self, state):
        action_plan = self.choose_action_all(state)
        initial_bw = float(state[0])

        total_energy, total_time, reward, new_bw, total_cloud_time = self.run_full_task(
            action_plan, initial_bw
        )

        full_key = self._make_full_key(state, action_plan)

        # visits
        self.visit_counts[full_key] = self.visit_counts.get(full_key, 0) + 1

        # optimistic init
        if full_key not in self.Q1:
            self.Q1[full_key] = self.optimistic_init_value
        if full_key not in self.Q2:
            self.Q2[full_key] = self.optimistic_init_value

        # select Q-table
        if random.random() < 0.5:
            q_table = self.Q1
        else:
            q_table = self.Q2

        old_q = q_table[full_key]
        target = float(reward)

        q_table[full_key] = old_q + self.alpha * (target - old_q)

        # if not self.is_test:
            # print(f"[UPDATE] key={full_key}")
            # print(f" visits={self.visit_counts[full_key]}")
            # print(f" old={old_q:.4f} new={q_table[full_key]:.4f} reward={reward:.4f}")
            # print(f" Q1={len(self.Q1)} Q2={len(self.Q2)}")

        if not self.is_test:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.notify_episode_end(reward)

        return reward, total_energy, total_time, new_bw, total_cloud_time

    # ----------------------------------------------------------------------
    # EPISODE END (epsilon boost)
    # ----------------------------------------------------------------------
    def notify_episode_end(self, episode_reward):
        if episode_reward > self.best_episode_reward:
            self.best_episode_reward = episode_reward
            self.episodes_since_improvement = 0
        else:
            self.episodes_since_improvement += 1

        if self.episodes_since_improvement >= self.stagnant_limit:
            self.epsilon = min(1.0, self.epsilon + self.epsilon_boost)
            self.episodes_since_improvement = 0

    # ----------------------------------------------------------------------
    # SAVE / LOAD
    # ----------------------------------------------------------------------
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
                print(f"Loaded Q-tables Q1={len(self.Q1)} Q2={len(self.Q2)}")
                return True
            print("No Q-table file found.")
            return False
        except Exception as e:
            print("Error loading:", e)
            return False
