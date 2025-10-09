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
        alpha: float = 0.001,
        gamma: float = 0.85,
        epsilon: float = 0.125,
    ):
        """
        Double Q-learning agent for layer-by-layer offloading decisions.

        Args:
            profiling_data: ProfilingData object (must provide layer/node counts, timings, powers, etc.)
            alpha: learning rate
            gamma: discount factor
            epsilon: initial epsilon for epsilon-greedy
            min_epsilon: lower bound for epsilon when decaying
            epsilon_decay: multiplicative decay applied when decay_epsilon() called
        """
        self.profiling = profiling_data
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-tables
        self.Q1 = {}
        self.Q2 = {}

        # Simulator
        self.simulator = CloudEdgeSimulator(profiling_data)

        # --- Discretization bins (coarser than original to avoid state explosion) ---
        # bandwidth (Mbps) — cap matches simulator cap (1..30)
        self.bandwidth_bins = np.linspace(1, 30, 6)  # e.g., [1,~6,~11,~16,~21,30]
        # cloud pending time (ms)
        self.cloudtime_bins = np.linspace(0, 500, 6)
        # surplus bins (coarser; avoid extremely fine fragmentation)
        self.surplus_bins = np.linspace(-5, 5, 21)  # step 0.5 s

    # ---------------------------
    # Utility & discretization
    # ---------------------------
    def _discretize(self, value, bins):
        """Return the bin representative for `value`. Uses nearest-bin (index) and returns bin value."""
        # If bins is a 1D array of representative values, map to the nearest bin index.
        idx = np.digitize([value], bins, right=True)[0] - 1
        idx = max(0, min(idx, len(bins) - 1))
        return float(bins[idx])

    def _action_to_key(self, action):
        """
        Convert action array (n x 2) to an immutable tuple representing assignments (0 edge / 1 cloud).
        Example: array([[layer,0],[layer,1]]) -> (0, 1)
        """
        return tuple(int(x) for x in action[:, 1].tolist())

    def _state_to_key(self, state):
        """
        Map continuous/structured state to a stable key for Q-tables.

        State expected format:
            (bandwidth, cloud_time_ms, layer, prev_action_array_or_None, surplus, negative_surplus_count)
        """
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

        # Form a compact tuple key
        return (bw_disc, ctime_disc, layer_i, surplus_disc, neg_count_i, prev_key)

    # ---------------------------
    # Actions
    # ---------------------------
    def _get_possible_actions(self, layer_idx):
        """
        Generate all possible assignments for a given layer.
        Each action is an array shape (nodes, 2) where column 0 is layer index, column 1 is assignment (0=edge,1=cloud).
        First and last layers are forced to edge (0) like your original design.
        """
        nodes = self.profiling.get_num_nodes(layer_idx)

        # First or last layer -> edge only
        if layer_idx == 0 or layer_idx == (len(self.profiling.layers) - 1):
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
        Epsilon-greedy selection based on sum of Q1 + Q2 (standard Double Q greedy policy uses combined estimates).
        Note: we still update Q1 or Q2 separately during training.
        """
        layer = int(state[2])
        actions = self._get_possible_actions(layer)
        s_key = self._state_to_key(state)

        # Exploration: only on non-terminal internal layers
        if (random.random() < self.epsilon) and (layer > 0 and layer < (len(self.profiling.layers) - 1)):
            return random.choice(actions)

        # Greedy selection based on Q1+Q2
        q_vals = []
        for a in actions:
            a_key = self._action_to_key(a)
            full_key = (s_key, a_key)
            q1 = self.Q1.get(full_key, 0.0)
            q2 = self.Q2.get(full_key, 0.0)
            q_vals.append(q1 + q2)

        chosen_idx = self._argmax_with_tiebreak(q_vals)
        return actions[chosen_idx]

    # ---------------------------
    # Training (single step)
    # ---------------------------
    def train(self, current_state):
        """
        Perform one step of interaction: choose action, query simulator (energy/time), get reward & next state,
        then update either Q1 or Q2 according to Double Q-learning rules.

        Args:
            current_state: tuple as expected by the simulator
            decay_epsilon: whether to apply multiplicative epsilon decay after the update
        Returns:
            (action, reward, next_state, terminal, energy, completion_time, next_bandwidth)
        """
        # Choose and apply action
        action = self.choose_action(current_state)

        # Simulator step(s)
        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state, current_action=action, cloud_pending_ms=current_state[1]
        )

        # Reward computation (simulator returns scaled reward)
        reward, surplus, negative_surplus_count, fractional_deadline = self.simulator.calculate_reward(
            int(current_state[2]), energy, completion_time_s, current_state[4], current_state[5]
        )
        surplus /= 1000.0  # convert to seconds

        # Next state from simulator
        next_state, terminal, _ = self.simulator.get_next_state(
            current_state, action, surplus, negative_surplus_count
        )

        # Compose keys
        cur_key = (self._state_to_key(current_state), self._action_to_key(action))

        # Choose which Q-table to update and perform Double-Q logic properly:
        if random.random() < 0.5:
            # Update Q1: select best next action according to Q1, evaluate with Q2
            q_table = self.Q1
            q_eval = self.Q2
            # If terminal -> no next action
            if terminal:
                target = reward
            else:
                next_actions = self._get_possible_actions(int(next_state[2]))
                # Choose best next action using Q1
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

        return action, reward, next_state, terminal, energy, completion_time_s, next_state[0], surplus , fractional_deadline

    # ---------------------------
    # Epsilon utils & persistence
    # ---------------------------
    # def decay_epsilon(self):
    #     """Decay epsilon multiplicatively but keep above min_epsilon."""
    #     self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    #     return self.epsilon

    def save_qtables(self, filename="q_tables.pkl"):
        """Save Q1 and Q2 tables to disk."""
        with open(filename, "wb") as f:
            pickle.dump((self.Q1, self.Q2), f)
        print(f"Q-tables saved to {filename}")

    def load_qtables(self, filename="q_tables.pkl"):
        """Load Q1 and Q2 tables if file exists, else skip."""
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.Q1, self.Q2 = pickle.load(f)
            print(f"Q-tables found and loaded from {filename}, q1 size: {len(self.Q1)}, q2 size: {len(self.Q2)}")
            return True
        else:
            print(f"Q-table file not found at {filename}. Starting fresh.")
            return False
