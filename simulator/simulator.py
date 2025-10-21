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

    def get_possible_actions(self, layer):
        """Generates all possible discrete action matrices for a layer (A2C/Discrete SAC logic)."""
        if layer >= len(self.profiling.layers):
            return []
        nodes = self.profiling.get_num_nodes(layer)
        actions = []
        for pattern in range(2 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            for i in range(nodes):
                a[i, 1] = (pattern >> i) & 1
            actions.append(a)
        return actions        

    def get_next_state(self, current_state, action, surplus, negative_surplus_count, isAllCloud= False):
        bandwidth, cloud_time_pending_ms, layer, previous_action, _, negative_surplus_count = current_state
        layer = int(layer)

        # Determine which nodes are on the cloud for this action
        cloud_nodes = np.where(action[:, 1] == 1)[0]
        congestion = random.uniform(25, 60)  # stochastic congestion ms
        new_cloud_pending = 0.0
        new_cloud_pending += congestion

        if isAllCloud:
            new_cloud_pending*=self.profiling.numberOfEdgeDevice*0.25


        # If some tasks are assigned to cloud this layer, compute new cloud processing added
        if len(cloud_nodes) > 0:
            # cloud_proc is the maximum cloud processing time among tasks assigned this layer
            cloud_proc_ms = max(self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes)
            # New pending = previous pending (what remains) + new cloud work + congestion
            new_cloud_pending += max(0.0,  cloud_proc_ms)

        # Bandwidth update (stochastic)
        bw_change = random.uniform(-1, 1)  # Mbps fluctuation
        new_bandwidth = max(1.0, min(bandwidth + bw_change, 10.0))

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

        Uses 'surplus' (ms) to propagate time savings or overruns across layers.
        """

        # --- 1. Compute fractional deadline for this layer (in ms) ---
        fractional_deadline_ms = (
            self.profiling.get_edge_time_for_layer(layer)
            / self.profiling.get_total_edge_time()
        ) * self.profiling.deadline

        # --- 2. Convert completion time to ms ---
        completion_time_ms = completion_time_s * 1000.0

        # --- 3. Compute surplus in ms ---
        # Positive surplus => finished early
        # Negative surplus => exceeded local deadline
        layer_surplus_ms = fractional_deadline_ms + previous_surplus - completion_time_ms

        # --- 4. λ (trade-off between energy and delay) ---
        if not isA2C:
            lambda_param = 5.0
        else:
            lambda_param = 1.0
        # --- 5. Penalties ---
        if layer_surplus_ms < 0:
            delay_penalty = abs(layer_surplus_ms) * lambda_param
        else:
            delay_penalty = 0.0

        # --- 6. Combine penalties smoothly ---
        # Normalize surplus to seconds scale for sigmoid stability
        norm_surplus = layer_surplus_ms / 100.0  # scaling prevents overflow
        sigmoid_weight = 1 / (1 + np.exp(-norm_surplus))

        energy_weight = lambda_param * sigmoid_weight
        delay_weight = lambda_param * (1 - sigmoid_weight)
        total_penalty = energy_weight * (total_energy * 100) + delay_weight * delay_penalty

        # --- 7. Reward computation ---
        reward = -total_penalty

        # --- 8. Local bonuses/penalties ---
        if layer_surplus_ms > 0:
            if (not isA2C):
                reward += 5
            else:
                reward += 0.5
        else:
            if (not isA2C):
                reward -= 15 * (abs(layer_surplus_ms) / fractional_deadline_ms)
            else:
                reward -= 0.75

        # --- 9. Return ---
        return reward, layer_surplus_ms, negative_surplus_count, fractional_deadline_ms



