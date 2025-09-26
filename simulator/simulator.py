import numpy as np
import random
from profiling.profile import ProfilingData

class CloudEdgeSimulator:
    def __init__(self, profiling_data: ProfilingData):
        """
        Simulator for predicting next state given current state and action.
        Args:
            profiling_data: ProfilingData object
        """
        self.profiling = profiling_data

    def get_next_state(self, current_state, action, surplus):
        """
        Compute next state given current state and action.
        State = (bandwidth [Mbps], cloud_time [ms], layer, prev_action_array)
        """
        bandwidth, cloud_time, layer, previous_action, ___ = current_state
        layer = int(layer)

        # --- Cloud processing update ---
        cloud_nodes = np.where(action[:, 1] == 1)[0]
        previous_cloud_nodes = np.where(previous_action[:, 1] == 1)[0] if previous_action is not None else []
        # If some tasks were on cloud previously and now new tasks are added to cloud,
        if len(cloud_nodes) > 0:
            cloud_proc = max(
                self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes
            )  # ms

            congestion = random.uniform(0, 100)  # ms
            cloud_time =  cloud_proc + congestion
        elif (previous_action is not None) and (len(previous_cloud_nodes) > 0):
            # SOME OF THE PREVIOUS OPERATIONS IN CLOUD IS ASSUMED TO BE DONE IN THIS FRAME
            cloud_time -= random.uniform(10, 20)

        else:
            # no new tasks → cloud_time decreases
            cloud_time = max(0.0, (cloud_time - random.uniform(0, 10)))

        # --- Bandwidth update (stochastic change) ---
        bw_change = random.uniform(-5, 5)  # Mbps fluctuation
        new_bandwidth = max(1.0, bandwidth + bw_change)
        new_bandwidth = min(new_bandwidth, 30.0)  # cap max bandwidth

        # --- Advance to next layer ---
        terminal = False
        if layer + 1 < len(self.profiling.layers):
            next_layer = layer + 1
        else:
            terminal = True
            next_layer = layer

        # --- Next state carries current action as prev_action ---
        next_state = (new_bandwidth, cloud_time, next_layer, action.copy(), surplus)
        return next_state, terminal, cloud_time


    def compute_energy_and_time(self, current_state, current_action, cloud_pending_ms):
        """
        Compute energy consumption and completion time for a given action.

        Returns:
            total_energy (float): total energy (Joules)
            completion_time_s (float): completion time (seconds)
        """
        bandwidth, _, layer, prev_action, _ = current_state
        layer = int(layer)

        total_energy = 0.0
        transmission_time_s = []

        # --- Transmission time calculation ---
        if prev_action is not None:  # not the first layer
            prev_assignments = prev_action[:, 1]
            curr_assignments = current_action[:, 1]
            for i in range(len(prev_assignments)):
                for j in range(len(curr_assignments)):
                    if prev_assignments[i] != curr_assignments[j]:
                        # convert KB → bits, Mbps → bits/s
                        transmission_time = (
                            (self.profiling.output_size * 8 * 1024)
                            / (max(bandwidth, 1e-6) * 10**6)
                        )
                        transmission_time_s.append(transmission_time)

        if len(transmission_time_s) > 0:
            transmission_time = max(transmission_time_s)
            total_energy += self.profiling.edge_communication_power * transmission_time  # J

        # --- Edge tasks energy ---
        edge_total_time_s = 0.0
        for i in range(len(current_action)):
            if current_action[i, 1] == 0:  # edge
                node_p = self.profiling.get_node_edge_power(layer, i)  # W
                node_t_s = self.profiling.get_node_edge_time(layer, i) / 1000.0  # ms → s
                edge_total_time_s += node_t_s
                total_energy += (node_p * node_t_s)  # J

        # --- Cloud energy ---
        cloud_pending_s = cloud_pending_ms / 1000.0
        actual_idle_time_s = 0.0
        if np.any(current_action[:, 1] == 1):  # some tasks on cloud
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)
            total_energy += self.profiling.edge_idle_power * actual_idle_time_s  # J

        # --- Completion time (s) ---
        completion_time_s = actual_idle_time_s + edge_total_time_s + max(transmission_time_s, default=0.0)

        return total_energy, completion_time_s


    def calculate_reward(self, layer, total_energy, completion_time_s, previous_surplus):
        """
        Compute reward from energy and completion time.

        Returns:
            reward (float)
        """
        # fractional deadline scaling
        fractional_deadline_s = (
            self.profiling.get_edge_time_for_layer(layer)/self.profiling.get_total_edge_time()
        ) * (self.profiling.deadline / 1000.0)  # ms → s

        constrained_completion_time_s = min(0, completion_time_s - previous_surplus)
        surplus = constrained_completion_time_s - fractional_deadline_s

        if constrained_completion_time_s > fractional_deadline_s:
            # Penalize proportional to delay
            delay = completion_time_s - fractional_deadline_s
            reward = -(total_energy + (delay*100))*1000000
        else:
            reward = - total_energy

        return reward , surplus

