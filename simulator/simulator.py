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

    def get_next_state(self, current_state, action, surplus, negative_surplus_count):
        bandwidth, cloud_time_pending_ms, layer, previous_action, _, negative_surplus_count = current_state
        layer = int(layer)

        # Determine which nodes are on the cloud for this action
        cloud_nodes = np.where(action[:, 1] == 1)[0]
        congestion = random.uniform(0, 100)  # stochastic congestion ms
        new_cloud_pending = 0.0
        new_cloud_pending += congestion


        # If some tasks are assigned to cloud this layer, compute new cloud processing added
        if len(cloud_nodes) > 0:
            # cloud_proc is the maximum cloud processing time among tasks assigned this layer
            cloud_proc_ms = max(self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes)
            # New pending = previous pending (what remains) + new cloud work + congestion
            new_cloud_pending += max(0.0,  cloud_proc_ms)

        # Bandwidth update (stochastic)
        bw_change = random.uniform(-5, 5)  # Mbps fluctuation
        new_bandwidth = max(1.0, min(bandwidth + bw_change, 30.0))

        # Next layer / terminal flag
        terminal = False
        if layer + 1 < len(self.profiling.layers):
            next_layer = layer + 1
        else:
            terminal = True
            next_layer = layer

        next_state = (new_bandwidth, new_cloud_pending, next_layer, action.copy(), surplus, negative_surplus_count)
        return next_state, terminal, new_cloud_pending

    def compute_energy_and_time(self, current_state, current_action, cloud_pending_ms):
        bandwidth, _, layer, prev_action, _, negative_surplus_count = current_state
        layer = int(layer)

        total_energy = 0.0
        transmission_times = []

        # --- Transmission time calculation ---
        # For each node, if its assignment changed from previous layer -> transmission needed.
        if prev_action is not None:
            prev_assignments = np.asarray(prev_action[:, 1], dtype=int)
            curr_assignments = np.asarray(current_action[:, 1], dtype=int)

            # Different node counts: compare via cross-layer transfers
            # Transmission only occurs between layers when assignments differ
            # (Edge → Cloud or Cloud → Edge)
            for prev_node in prev_assignments:
                for curr_node in curr_assignments:
                    if prev_node != curr_node:
                        transmission_time = (
                            (self.profiling.output_size * 8 * 1024)
                            / (max(bandwidth, 1e-6) * 1e6)
                        )
                        transmission_times.append(transmission_time)

        # If any transmissions, the bottleneck is the longest one (assuming pipelined/parallel transfers)
        max_transmission_time = max(transmission_times) if transmission_times else 0.0
        if max_transmission_time > 0:
            total_energy += self.profiling.edge_communication_power * max_transmission_time

        # --- Edge tasks energy ---
        edge_total_time_s = 0.0
        for i in range(len(current_action)):
            if current_action[i, 1] == 0:  # edge
                node_p = self.profiling.get_node_edge_power(layer, i)  # W
                node_t_s = self.profiling.get_node_edge_time(layer, i) / 1000.0  # ms → s
                edge_total_time_s += node_t_s
                total_energy += (node_p * node_t_s)  # J

        # --- Cloud idle/busy energy ---
        cloud_pending_s = cloud_pending_ms / 1000.0
        actual_idle_time_s = 0.0
        if np.any(current_action[:, 1] == 1):  # some tasks on cloud
            # If cloud has pending work, edge processing time may be overlapped with cloud pending time.
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)
            total_energy += self.profiling.edge_idle_power * actual_idle_time_s  # J

        # --- Completion time (s) ---
        # total time is edge processing + any necessary transmission + any waiting for cloud results
        completion_time_s = edge_total_time_s + max_transmission_time + actual_idle_time_s

        return total_energy, completion_time_s



    def calculate_reward(
        self,
        layer,
        total_energy,
        completion_time_s,
        previous_surplus,
        negative_surplus_count,
        isA2C=False,
    ):
        """
        Reward = High when energy is low and the overall computation 
        remains within the global deadline (via fractional deadlines).

        Uses 'surplus' to propagate time savings or overruns across layers.
        """

        # --- 1. Compute fractional deadline for this layer ---
        fractional_deadline_s = (
            self.profiling.get_edge_time_for_layer(layer)
            / self.profiling.get_total_edge_time()
        ) * (self.profiling.deadline / 1000.0)

        # --- 2. Compute surplus (carry-over time budget) ---
        # Positive surplus => finished early (time saved)
        # Negative surplus => exceeded time (deadline pressure)
        layer_surplus = fractional_deadline_s + previous_surplus - completion_time_s

        # --- 3. Define a dynamic λ (penalty balancing factor) ---
        # Higher power or time makes delay more critical.
        lambda_param = 5.0

        # --- 4. Compute penalties ---
        # Energy is always minimized
        energy_penalty = total_energy

        # Delay penalty only applies if we are late
        delay_penalty = 0.0
        if layer_surplus < 0:
            delay_penalty = abs(layer_surplus) * lambda_param
            negative_surplus_count += 1

        # --- 5. Combine penalties ---
        total_penalty = energy_penalty + delay_penalty

        # --- 6. Compute reward ---
        # Negative because we want to minimize both
        reward = -total_penalty

        # --- 7. Add bonuses/penalties based on surplus ---
        if layer_surplus > 0:
            # Finished early — small positive incentive
            reward += 5.0  * layer_surplus
        else:
            # Late — small additional penalty for missing local deadline
            reward -= 5.0 * abs(layer_surplus)

        # --- 8. If this is the final layer ---
        # Add a global deadline check
        if layer == len(self.profiling.layers) - 1:
            if layer_surplus >= 0:
                # All layers collectively finished within deadline
                reward += 10.0
            else:
                # Missed global deadline → penalty grows with delay count
                reward -= 5.0 * negative_surplus_count

        return reward, layer_surplus, negative_surplus_count


