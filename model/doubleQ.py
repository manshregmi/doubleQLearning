import numpy as np
import random
from profile.profile import ProfilingData


class DoubleQLearningAgent:
    def __init__(self, profiling_data: ProfilingData, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Double Q-Learning agent for cloudlet-edge assignment scenario.
        Uses ProfilingData to fetch layer-node specific metrics.
        """
        self.profiling = profiling_data
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q1 = {}
        self.Q2 = {}

    # ----- Helpers -----
    def _state_to_key(self, state):
        """Discretize continuous state values for Q-table stability."""
        bw, ctime, layer, nodes = state
        return (round(bw, 1), round(ctime, 1), int(layer), int(nodes))

    def _action_to_key(self, action):
        return tuple(map(tuple, action))

    def _get_possible_actions(self, layer_idx):
        """
        Generate all possible assignment vectors for nodes in a layer
        Column 0: layer number
        Column 1: assignment (0=edge, 1=cloudlet)
        """
        nodes = self.profiling.get_num_nodes(layer_idx)
        actions = []
        for pattern in range(2 ** nodes):
            action = np.zeros((nodes, 2), dtype=int)
            action[:, 0] = layer_idx
            for i in range(nodes):
                action[i, 1] = (pattern >> i) & 1
            actions.append(action)
        return actions

    # ----- Action selection -----
    def choose_action(self, state):
        layer = int(state[2])
        actions = self._get_possible_actions(layer)

        if random.random() < self.epsilon:
            return random.choice(actions)

        q_values = [
            self.Q1.get((self._state_to_key(state), self._action_to_key(a)), 0.0)
            + self.Q2.get((self._state_to_key(state), self._action_to_key(a)), 0.0)
            for a in actions
        ]
        return actions[np.argmax(q_values)]

    # ----- Next state calculation -----
    def get_next_state(self, current_state, action):
        bandwidth, comp_time, layer, _ = current_state
        layer = int(layer)
        nodes = self.profiling.get_num_nodes(layer)

        total_comp_time = 0
        for i in range(nodes):
            if action[i, 1] == 1:  # cloudlet
                node_ctime = self.profiling.get_node_cloud_time(layer, i)
                total_comp_time += node_ctime

        # Bandwidth reduction
        cloudlet_nodes = np.sum(action[:, 1] == 1) * self.profiling.numberOfEdgeDevice
        network_latency = self.profiling.rtt + (self.profiling.output_size / max(bandwidth, 1e-6))
        bandwidth_drop = 0.05 * cloudlet_nodes + 0.1 * network_latency
        next_bandwidth = max(bandwidth - bandwidth_drop, 1.0)

        # Advance layer or end episode if last layer
        terminal = False
        if layer + 1 < len(self.profiling.layers):
            layer += 1
        else:
            terminal = True  # stop when last layer is done

        next_nodes = self.profiling.get_num_nodes(layer) if not terminal else 0

        next_state = [
            next_bandwidth,
            (total_comp_time * self.profiling.numberOfEdgeDevice * 0.4) + network_latency,
            layer,
            next_nodes,
        ]
        return next_state, terminal

    # ----- Reward calculation -----
    def calculate_reward(self, next_state, action):
        _, _, layer, _ = next_state
        layer = int(layer)
        nodes = action.shape[0]

        total_energy = 0
        total_time = 0
        rtt = 1  # could be self.profiling.rtt too

        for i in range(nodes):
            if action[i, 1] == 1:  # cloudlet
                total_time += (next_state[1] + rtt)
                total_energy += self.profiling.edge_idle_power * (next_state[1] + rtt)
            else:  # edge
                node_power = self.profiling.get_node_edge_power(layer, i)
                node_time = self.profiling.get_node_edge_time(layer, i)
                total_energy += node_power * node_time

        fractional_deadline = (
            self.profiling.deadline
            * self.profiling.get_num_nodes(layer)
            / self.profiling.get_total_nodes()
        )

        # Normalize reward: [-1, 1]
        if total_time > fractional_deadline:
            return -1.0  # deadline miss
        else:
            # reward higher if energy is low
            max_energy = self.profiling.get_total_nodes() * self.profiling.edge_idle_power * self.profiling.deadline
            return 1.0 - (total_energy / (max_energy + 1e-6))

    # ----- Training update -----
    def train(self, current_state):
        action = self.choose_action(current_state)
        next_state, terminal = self.get_next_state(current_state, action)
        reward = self.calculate_reward(next_state, action)

        # Randomly choose which Q-table to update
        if random.random() < 0.5:
            q_table, q_other = self.Q1, self.Q2
        else:
            q_table, q_other = self.Q2, self.Q1

        key = (self._state_to_key(current_state), self._action_to_key(action))
        old_value = q_table.get(key, 0.0)

        if terminal:
            target = reward
        else:
            # Best next action according to chosen Q-table
            layer_next = int(next_state[2])
            next_actions = self._get_possible_actions(layer_next)
            best_next_action = max(
                next_actions,
                key=lambda a: q_table.get((self._state_to_key(next_state), self._action_to_key(a)), 0.0),
            )
            target = reward + self.gamma * q_other.get(
                (self._state_to_key(next_state), self._action_to_key(best_next_action)), 0.0
            )

        q_table[key] = old_value + self.alpha * (target - old_value)
        return action, reward, next_state, terminal
