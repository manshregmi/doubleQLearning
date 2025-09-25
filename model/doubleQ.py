import numpy as np
import random
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


class DoubleQLearningAgent:
    def __init__(self, profiling_data: ProfilingData, alpha=0.1, gamma=0.9, epsilon=0.05):
        self.profiling = profiling_data
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q1 = {}
        self.Q2 = {}
        self.simulator = CloudEdgeSimulator(profiling_data)

        # ---- Discretization bins ----
        # Bandwidth in Mbps (range 1–100 Mbps, 20 bins)
        self.bandwidth_bins = np.linspace(1, 100, 5)
        # Cloud congestion time in ms (range 0–500 ms, 25 bins)
        self.cloudtime_bins = np.linspace(0, 500, 5)

    # ----- Helpers -----
    def _discretize(self, value, bins):
        """Map continuous value to nearest bin center."""
        idx = np.digitize(value, bins) - 1
        idx = max(0, min(idx, len(bins) - 1))  # clamp
        return bins[idx]

    def _state_to_key(self, state):
        """Discretize continuous values for Q-table stability, include prev_action as tuple."""
        bw, ctime, layer, prev_action = state

        # Discretize bandwidth (Mbps) and cloud congestion (ms)
        bw_disc = self._discretize(float(bw), self.bandwidth_bins)
        ctime_disc = self._discretize(float(ctime), self.cloudtime_bins)
        s_key = (bw_disc, ctime_disc, int(layer))

        if prev_action is not None:
            prev_action_key = self._action_to_key(prev_action)
            return s_key + (prev_action_key,)
        return s_key + (None,)

    def _action_to_key(self, action):
        """Convert action array into immutable tuple (only assignment col)."""
        return tuple(int(x) for x in action[:, 1].tolist())

    def _get_possible_actions(self, layer_idx):
        """Generate all possible action patterns for given layer."""
        nodes = self.profiling.get_num_nodes(layer_idx)

        # First layer → EDGE only
        if (layer_idx == 0 or layer_idx == (len(self.profiling.layers) - 1)):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            a[:, 1] = 0   # 0 = edge
            return [a]


        # Middle layers → all patterns (edge=0, cloud=1)
        actions = []
        for pattern in range(2 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            for i in range(nodes):
                a[i, 1] = (pattern >> i) & 1
            actions.append(a)
        return actions


    # ----- Action selection -----
    def choose_action(self, state):
        layer = int(state[2])
        actions = self._get_possible_actions(layer)
        s_key = self._state_to_key(state)

        # ε-greedy
        if (random.random() < self.epsilon) and layer > 0 and layer < (len(self.profiling.layers) - 1) :
            return random.choice(actions)

        q_values = []
        for a in actions:
            key = (s_key, self._action_to_key(a))
            q_values.append(self.Q1.get(key, 0.0) + self.Q2.get(key, 0.0))

        return actions[int(np.argmax(q_values))]


    # ----- Training update -----
    def train(self, current_state):
        # Choose action
        action = self.choose_action(current_state)
        # Environment transition
        next_state, terminal, _ = self.simulator.get_next_state(current_state, action)
        energy, completion_time = self.simulator.compute_energy_and_time(current_state=current_state, current_action=action, cloud_pending_ms= next_state[1])
        # Reward calculation
        reward = self.simulator.calculate_reward(int(current_state[2]), energy, completion_time)

        # Decide which Q-table to update
        if random.random() < 0.5:
            q_table, q_other = self.Q1, self.Q2
        else:
            q_table, q_other = self.Q2, self.Q1

        # Current key
        key = (self._state_to_key(current_state), self._action_to_key(action))
        old_value = q_table.get(key, 0.0)

        if terminal:
            target = reward
        else:
            layer_next = int(next_state[2])
            next_actions = self._get_possible_actions(layer_next)

            # Best next action
            best_next_action = max(
                next_actions,
                key=lambda a: q_table.get(
                    (self._state_to_key(next_state), self._action_to_key(a)), 0.0
                ),
            )
            target = reward + self.gamma * q_other.get(
                (self._state_to_key(next_state), self._action_to_key(best_next_action)), 0.0
            )

        # Update Q-value
        q_table[key] = old_value + self.alpha * (target - old_value)

        return action, reward, next_state, terminal, energy, completion_time , next_state[0]
