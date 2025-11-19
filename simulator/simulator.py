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


    def get_next_state_cloud_waiting_time(self, next_layer, current_action, isAllCloud= False):
        layer = int(next_layer)

        # Determine which nodes are on the cloud for this action
        cloud_nodes = np.where(current_action[:, 1] == 1)[0]
        congestion = abs(self.profiling.get_max_layer_cloud_time(layer) * (self.profiling.numberOfEdgeDevice - 1) * np.random.uniform(0,1))
        # congestion = 0.0
        new_cloud_pending = 0.0
        new_cloud_pending += congestion

        # If some tasks are assigned to cloud this layer, compute new cloud processing added
        if len(cloud_nodes) > 0:
            # cloud_proc is the maximum cloud processing time among tasks assigned this layer
            cloud_proc_ms = max(self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes)
            # New pending =  new cloud work + congestion
            new_cloud_pending += max(0.0,  cloud_proc_ms)
        
        # Determine which nodes are on the cloud for this action
        cloud_nodes = np.where(current_action[:, 1] == 1)[0]
        # congestion = random.uniform(15, 50)  # stochastic congestion ms

        # If some tasks are assigned to cloud this layer, compute new cloud processing added
        if isAllCloud and len(cloud_nodes) > 0:
            # cloud_proc is the maximum cloud processing time among tasks assigned this layer
            cloud_proc_ms = max(self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes)
            # New pending = previous pending (what remains) + new cloud work + congestion
            new_cloud_pending = max(0.0,  cloud_proc_ms)*self.profiling.numberOfEdgeDevice


        return  new_cloud_pending

    def get_next_state(self, current_state, action, surplus, negative_surplus_count, new_cloud_pending):
        bandwidth, _, layer, _, _, negative_surplus_count = current_state
        layer = int(layer)

        
        # Bandwidth update (stochastic)
        bw_change = np.random.normal(-0.25, 1)
        # bw_change = 0
        new_bandwidth = max(1.0, min(bandwidth + bw_change, 15.0))

        # Next layer / terminal flag
        terminal = False
        if layer + 1  < len(self.profiling.layers):
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
        profiling = self.profiling
        deps = profiling.dependencies

        # --- Transmission Time (Dependency-based) ---
        transmission_times = []

        if prev_action is not None and layer > 0:
            prev_assignments = np.asarray(prev_action[:, 1], dtype=int)
            curr_assignments = np.asarray(current_action[:, 1], dtype=int)

            for curr_node in range(len(curr_assignments)):
                parent_nodes = deps.get((layer, curr_node), [])
                for (p_layer, p_node) in parent_nodes:
                    parent_loc = prev_assignments[p_node] if p_layer == layer - 1 else 0
                    curr_loc = curr_assignments[curr_node]

                    # Transmission only if parent and child are on different locations
                    if parent_loc != curr_loc:
                        output_size = profiling.get_output_size(layer, curr_node)
                        transmission_time = max(
                            (output_size / 1024.0) / max(bandwidth, 1e-6),
                            profiling.rtt / 1000.0
                        )
                        # print("bandwidth:", bandwidth, "output size:", output_size, "transmission time (s):", transmission_time)
                        transmission_times.append(transmission_time)

        else:
            # First layer → upload input to cloud if cloud execution
            for i in range(len(current_action)):
                if current_action[i, 1] == 1:  # cloud
                    transmission_time = max(
                        (profiling.get_input_size() / 1024.0) / max(bandwidth, 1e-6),
                        profiling.rtt / 1000.0
                    )
                    transmission_times.append(transmission_time)

        # Bottleneck = longest dependent transmission
        max_transmission_time = max(transmission_times) if transmission_times else 0.0
        if max_transmission_time > 0:
            total_energy += profiling.edge_communication_power * max_transmission_time

        # --- Edge Processing Energy ---
        edge_times = []
        edge_energy = []
        for i in range(len(current_action)):
            if current_action[i, 1] == 0:  # edge node
                node_p = profiling.get_node_edge_power(layer, i)
                node_t_s = profiling.get_node_edge_time(layer, i) / 1000.0
                edge_energy.append(node_p * node_t_s)
                edge_times.append(node_t_s)

        # Parallel execution only at layer 3 and layer 5
        if layer in [3, 5]:
            edge_total_time_s = max(edge_times) if edge_times else 0.0
            total_energy += max(edge_energy) if edge_energy else 0.0
        else:
            edge_total_time_s = sum(edge_times)
            total_energy += sum(edge_energy)

        # --- Cloud Idle Energy ---
        actual_idle_time_s = 0.0
        if np.any(current_action[:, 1] == 1):  # any node on cloud
            cloud_pending_s = cloud_pending_ms / 1000.0
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)
            total_energy += profiling.edge_idle_power * actual_idle_time_s

        # --- Completion Time (seconds) ---
        completion_time_s = edge_total_time_s + max_transmission_time + actual_idle_time_s
        # print(f"Layer {layer} | Edge Time: {edge_total_time_s*1000:.2f} ms | Transmission Time: {max_transmission_time*1000:.2f} ms | Idle Time: {actual_idle_time_s*1000:.2f} ms | Total Time: {completion_time_s*1000:.2f} ms | Energy: {total_energy:.4f} J, action: {current_action[:,1].tolist()}")

        return total_energy, completion_time_s



    def sigmoid(self,x, k=5):
        """
        Standard sigmoid, steeper at higher k (>0).
        Output approaches 1 for large positive x, 0 for large negative x.
        """
        return 1 / (1 + np.exp(-k * x))

    def calculate_reward(
        self,
        layer,
        total_energy,   # in Joules
        completion_time_s,  # in seconds
        previous_surplus,   # in ms
        negative_surplus_count, # unused, placeholder for future custom penalties
        isA2C=False,
    ):
        # Compute per-layer allowed time (ms), based on profiling fractions
        fractional_deadline_ms = (
            self.profiling.get_edge_time_for_layer(layer)
            / self.profiling.get_total_edge_time()
        ) * self.profiling.deadline

        completion_time_ms = completion_time_s * 1000.0

        surplus_ms = fractional_deadline_ms + previous_surplus - completion_time_ms

        completion_time_ms = completion_time_ms + surplus_ms

        missed_deadline = completion_time_ms > fractional_deadline_ms

        sigmoid_wight = self.sigmoid(surplus_ms/1000, k=1)

        reward = 0
        if (missed_deadline):
            negative_surplus_count +=1
        
        reward = ((sigmoid_wight * total_energy*1000) + ((1-sigmoid_wight)*abs(completion_time_ms)))
        if (negative_surplus_count > 1):
            reward += negative_surplus_count * abs(completion_time_ms)

        reward *= -1

        return reward, surplus_ms, negative_surplus_count, fractional_deadline_ms


    def compute_whole_action_reward(self, total_energy, total_completion_time):
        # print(total_completion_time)

        completion_time_ms = total_completion_time * 1000.0

        missed_deadline = completion_time_ms > self.profiling.deadline

        surplus_ms = self.profiling.deadline = completion_time_ms

        sigmoid_wight = self.sigmoid(surplus_ms/1000, k=1)

        reward = 0
        reward = ((sigmoid_wight * total_energy*1000) + ((1-sigmoid_wight)*abs(completion_time_ms)))
        if (missed_deadline):
            reward +=  abs(completion_time_ms)

        reward *= -1

        return reward
