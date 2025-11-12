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

        # --- Discretization bins (coarser than original to avoid state explosion) ---
        self.bandwidth_bins = np.linspace(1, 15, 6)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        self.surplus_bins = np.linspace(-300, 300, 100)

    # ---------------------------
    # Utility & discretization
    # ---------------------------
    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        idx = max(0, min(idx, len(bins) - 1))
        return float(bins[idx])

    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1].tolist())

    def _state_to_key(self, state):
        bw, ctime, layer, prev_action, surplus, negative_surplus_count = state
        bw_disc = self._discretize(float(bw), self.bandwidth_bins)
        ctime_disc = self._discretize(float(ctime), self.cloudtime_bins)
        surplus_disc = self._discretize(float(surplus), self.surplus_bins)
        layer_i = int(layer)
        neg_count_i = int(negative_surplus_count)
        if prev_action is None:
            prev_key = None
        else:
            prev_key = self._action_to_key(prev_action)
        return (bw_disc, ctime_disc, layer_i, surplus_disc, neg_count_i, prev_key)

    # ---------------------------
    # Actions
    # ---------------------------
    def _get_possible_actions(self, layer_idx):
        nodes = self.profiling.get_num_nodes(layer_idx)
        if layer_idx == (len(self.profiling.layers) - 1):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            a[:, 1] = 0
            return [a]

        actions = []
        for pattern in range(2 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            for i in range(nodes):
                a[i, 1] = (pattern >> i) & 1
            actions.append(a)
        return actions

    def _argmax_with_tiebreak(self, values):
        """Return index of max with random tie-breaking."""
        max_val = max(values)
        candidates = [i for i, v in enumerate(values) if np.isclose(v, max_val)]
        return random.choice(candidates)

    # ---------------------------
    # Action selection
    # ---------------------------
    def choose_action(self, state):
        """
        Epsilon-greedy selection based on sum of Q1 + Q2.
        When is_test=True: deterministic choice based purely on Q1+Q2 (no epsilon, no random tie-break).
        """
        layer = int(state[2])
        actions = self._get_possible_actions(layer)
        s_key = self._state_to_key(state)

        # Exploration only allowed when NOT testing and on internal layers
        if (not self.is_test) and (random.random() < self.epsilon) and (layer > 0 and layer < (len(self.profiling.layers) - 1)):
            return random.choice(actions)

        # Compute Q-values (Q1+Q2)
        q_vals = []
        for a in actions:
            a_key = self._action_to_key(a)
            full_key = (s_key, a_key)
            q1 = self.Q1.get(full_key, 0.0)
            q2 = self.Q2.get(full_key, 0.0)
            q_vals.append(q1 + q2)

        # Deterministic when testing: take argmax (np.argmax returns first max)
        if self.is_test:
            chosen_idx = int(np.argmax(q_vals))
        else:
            # Training: break ties randomly for exploration stability
            chosen_idx = self._argmax_with_tiebreak(q_vals)

        return actions[chosen_idx]

    # ---------------------------
    # Training (single step)
    # ---------------------------
    def train(self, current_state):
        """
        Perform one environment step.
        If is_test=True: do NOT update Q-tables, do NOT apply discounting for updates (no updates done).
        Returns:
            (action, reward, next_state, terminal, energy, completion_time, next_bandwidth, surplus, fractional_deadline)
        """
        # Choose and apply action (deterministic in test)
        action = self.choose_action(current_state)

        next_state_cloud_processing = self.simulator.get_next_state_cloud_waiting_time(
            next_layer = (int(current_state[2])) if ((int(current_state[2]) + 1)  < len(self.profiling.layers)) else int(current_state[2]),
            current_action=action, isAllCloud=False
        )

        # Simulator step(s)
        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state, current_action=action, cloud_pending_ms=next_state_cloud_processing
        )

        # Reward computation (simulator returns scaled reward)
        reward, surplus, negative_surplus_count, fractional_deadline = self.simulator.calculate_reward(
            int(current_state[2]), energy, completion_time_s, current_state[4], current_state[5], isA2C=False
        )

        # Next state from simulator
        next_state, terminal, _ = self.simulator.get_next_state(
            current_state, action, surplus, negative_surplus_count, new_cloud_pending=next_state_cloud_processing,
        )

        # If we're in test mode -> DO NOT update Q-tables and DO NOT use gamma/epsilon for anything.
        if self.is_test:
            # Return observed quantities; Q1/Q2 left unchanged.
            return action, reward, next_state, terminal, energy, completion_time_s, next_state[0], surplus, fractional_deadline

        # --- Training mode: perform Double-Q update (unchanged behaviour) ---
        cur_key = (self._state_to_key(current_state), self._action_to_key(action))

        if random.random() < 0.5:
            # Update Q1: select best next action according to Q1, evaluate with Q2
            q_table = self.Q1
            q_eval = self.Q2
            if terminal:
                target = reward
            else:
                next_actions = self._get_possible_actions(int(next_state[2]))
                best_next_action = max(
                    next_actions,
                    key=lambda a: q_table.get((self._state_to_key(next_state), self._action_to_key(a)), 0.0),
                )
                eval_value = q_eval.get((self._state_to_key(next_state), self._action_to_key(best_next_action)), 0.0)
                target = reward + self.gamma * eval_value
        else:
            # Update Q2: select best next action according to Q2, evaluate with Q1
            q_table = self.Q2
            q_eval = self.Q1
            if terminal:
                target = reward
            else:
                next_actions = self._get_possible_actions(int(next_state[2]))
                best_next_action = max(
                    next_actions,
                    key=lambda a: q_table.get((self._state_to_key(next_state), self._action_to_key(a)), 0.0),
                )
                eval_value = q_eval.get((self._state_to_key(next_state), self._action_to_key(best_next_action)), 0.0)
                target = reward + self.gamma * eval_value

        old_value = q_table.get(cur_key, 0.0)
        q_table[cur_key] = old_value + self.alpha * (target - old_value)

        return action, reward, next_state, terminal, energy, completion_time_s, next_state[0], surplus, fractional_deadline

    # ---------------------------
    # Persistence
    # ---------------------------
    def save_qtables(self, filename="q_tables.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump((self.Q1, self.Q2), f)
        except Exception as e:
            print(f"Error saving Q-tables: {e}")

    def load_qtables(self, filename="q_tables.pkl"):
        try:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    self.Q1, self.Q2 = pickle.load(f)
                print(f"Q-tables found and loaded from {filename}, q1 size: {len(self.Q1)}, q2 size: {len(self.Q2)}")
                return True
            else:
                print(f"Q-table file not found at {filename}. Starting fresh.")
                return False
        except Exception as e:
            print(f"Error loading Q-tables: {e}")
            return False
