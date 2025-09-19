import numpy as np
import random

class CloudEdgeSimulator:
    def __init__(self, profiling_data):
        """
        Simulator for predicting next state given current state and action.
        Args:
            profiling_data: ProfilingData object
        """
        self.profiling = profiling_data

    def get_next_state(self, current_state, action):
        """
        Compute next state given current state and action.
        State = (bandwidth [Mbps], cloud_time [ms], layer, prev_action_array)
        """
        bandwidth, cloud_time, layer, _ = current_state
        layer = int(layer)

        # --- Cloud processing update ---
        cloud_nodes = np.where(action[:, 1] == 1)[0]
        if len(cloud_nodes) > 0:
            cloud_proc = max(
                self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes
            )  # ms

            congestion = random.uniform(0, 5)  # ms
            cloud_time = (
                max(0.0, cloud_time - random.uniform(0, 5))
                + cloud_proc
                + congestion
            )
        else:
            # no new tasks → cloud_time decreases
            cloud_time = max(0.0, cloud_time - random.uniform(0, 5))

        # --- Bandwidth update (stochastic change) ---
        bw_change = random.uniform(-5, 5)  # Mbps fluctuation
        new_bandwidth = max(1.0, bandwidth + bw_change)

        # --- Advance to next layer ---
        terminal = False
        if layer + 1 < len(self.profiling.layers):
            next_layer = layer + 1
        else:
            terminal = True
            next_layer = layer

        # --- Next state carries current action as prev_action ---
        next_state = (new_bandwidth, cloud_time, next_layer, action.copy())
        return next_state, terminal, cloud_time

    def calculate_reward(self, next_state, action):
        """
        Reward using next_state:
        Converts units here:
          - ms → s for time
          - KB → bits, Mbps → bits/s for transmission
          - Energy = Watts × seconds
        """
        bandwidth, cloud_pending_ms, layer, prev_action = next_state
        layer = int(layer)

        total_energy = 0.0
        transmission_time_s = []

        # --- Transmission time calculation ---
        if prev_action is not None:  # not the first layer
            prev_assignments = prev_action[:, 1]
            curr_assignments = action[:, 1]
            transmission_time = 0.0
            for i in range(len(prev_assignments)):
                for j in range(len(curr_assignments)):
                    if prev_assignments[i] != curr_assignments[j]:
                        # convert KB → bits, Mbps → bits/s
                        transmission_time += (
                            (self.profiling.output_size * 8 * 1024)
                            / (max(bandwidth, 1e-6) * 1e6)
                        )
                        transmission_time_s.append(transmission_time)

        if len(transmission_time_s) > 0:
            transmission_time = max(transmission_time_s)+( self.profiling.rtt / 1000.0)  # RTT ms → s
            total_energy += self.profiling.edge_communication_power * transmission_time  # J

        # --- Edge tasks energy ---
        edge_total_time_s = 0.0
        for i in range(len(action)):
            if action[i, 1] == 0:  # edge
                node_p = self.profiling.get_node_edge_power(layer, i)  # W
                node_t_s = self.profiling.get_node_edge_time(layer, i) / 1000.0  # ms → s
                edge_total_time_s += node_t_s
                total_energy += node_p * node_t_s  # J

        # --- Cloud energy ---
        cloud_pending_s = cloud_pending_ms / 1000.0
        actual_idle_time_s = 0.0
        final_transmission_time_s = 0.0
        if np.any(action[:, 1] == 1):  # some tasks on cloud
        
            actual_idle_time_s = max(
                0.0, cloud_pending_s - edge_total_time_s
            )
            total_energy += self.profiling.edge_idle_power * actual_idle_time_s  # J
            # At the end of last layer, if cloud was used
            if layer == len(self.profiling.layers) - 1 and np.any(action[:, 1] == 1):
                # Convert KB → Mb
                size_Mb = (self.profiling.output_size * 8) / 1000.0  

                # Transmission time in ms
                final_transmission_time_s = (size_Mb / max(bandwidth, 1e-6)) * 1000.0  

                # Add communication energy (convert ms → s)
                total_energy += (
                    self.profiling.edge_communication_power * (final_transmission_time_s * 1e-3)
                )


        # --- Completion time (s) ---
        completion_time_s = actual_idle_time_s + edge_total_time_s + max(transmission_time_s, default=0.0) + final_transmission_time_s 

        # --- Deadline check ---
        fractional_deadline_s = (
            self.profiling.deadline / 1000.0  # ms → s
            * self.profiling.get_num_nodes(layer)
            / max(1, self.profiling.get_total_nodes())
        )

        if completion_time_s > fractional_deadline_s:
            reward = -10000
        else:
            # Calculate max possible edge energy for this layer
            max_energy = sum(
                self.profiling.get_node_edge_power(layer, i)
                * (self.profiling.get_node_edge_time(layer, i) / 1000.0)
                for i in range(self.profiling.get_num_nodes(layer))
            )
            norm_energy = total_energy / max_energy if max_energy > 0 else 1.0

            if np.isclose(max_energy, total_energy):
                reward = -12000
            else:
                reward = (1.0 - norm_energy) * 10000

        return reward, total_energy, completion_time_s
