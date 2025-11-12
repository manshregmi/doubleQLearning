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
        # Handle both scalar and vector action representations
        if isinstance(current_action, (int, np.integer)):
            cloud_nodes = [current_action] if current_action == 1 else []
        elif current_action.ndim == 1:
            cloud_nodes = np.where(current_action == 1)[0]
        else:
            cloud_nodes = np.where(current_action[:, 1] == 1)[0]
        congestion = abs(self.profiling.get_max_layer_cloud_time(layer) * (self.profiling.numberOfEdgeDevice - 1) * np.random.uniform(0, 0.5))
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
        congestion = random.uniform(15, 50)  # stochastic congestion ms
        new_cloud_pending += congestion

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
        bw_change = np.random.normal(-0.5, 0.5)
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
    

    def handle_ndim_action(self, current_action):
        """
        Normalize current_action to a standard format.
        Handles scalar, 1D, 2D, and None.
        Returns a NumPy array suitable for indexing.
        """
        if current_action is None:
            return np.array([])  # empty array for no previous action

        if isinstance(current_action, (int, np.integer)):
            return np.array([current_action])
        elif isinstance(current_action, np.ndarray):
            if current_action.ndim == 1:
                return current_action
            else:
                return current_action[:, 1]  # extract offloading decisions
        else:
            raise ValueError(f"Unexpected action type: {type(current_action)}")


    def compute_energy_and_time(self, current_state, current_action, cloud_pending_ms):
        bandwidth, _, layer, prev_action, _, negative_surplus_count = current_state
        layer = int(layer)
        total_energy = 0.0
        profiling = self.profiling
        deps = profiling.dependencies

        # --- Transmission Time (Dependency-based) ---
        transmission_times = []
        current_action = self.handle_ndim_action(current_action)  # 1D array
        prev_action = self.handle_ndim_action(prev_action)        # 1D array or empty

        if prev_action.size > 0 and layer > 0:
            # Layer > 0, has previous layer
            for curr_node in range(len(current_action)):
                parent_nodes = deps.get((layer, curr_node), [])
                for (p_layer, p_node) in parent_nodes:
                    parent_loc = prev_action[p_node] if p_layer == layer - 1 else 0
                    curr_loc = current_action[curr_node]

                    # Transmission only if parent and child are on different locations
                    if parent_loc != curr_loc:
                        output_size = profiling.get_output_size(p_layer, p_node)
                        transmission_time = max(
                            (output_size / 1024.0) / max(bandwidth, 1e-6),
                            profiling.rtt / 1000.0
                        )
                        transmission_times.append(transmission_time)

        else:
            # First layer → upload input to cloud if cloud execution
            for i in range(len(current_action)):
                if current_action[i] == 1:  # cloud
                    transmission_time = max(
                        (profiling.get_input_size() / 1024.0) / max(bandwidth, 1e-6),
                        profiling.rtt / 1000.0
                    )
                    transmission_times.append(transmission_time)

        max_transmission_time = max(transmission_times) if transmission_times else 0.0
        if max_transmission_time > 0:
            total_energy += profiling.edge_communication_power * max_transmission_time

        # --- Edge Processing Energy ---
        edge_times = []
        edge_energy = []
        for i in range(len(current_action)):
            if current_action[i] == 0:  # edge node
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
        if np.any(current_action == 1):  # any node on cloud
            cloud_pending_s = cloud_pending_ms / 1000.0
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)
            total_energy += profiling.edge_idle_power * actual_idle_time_s

        # --- Completion Time (seconds) ---
        completion_time_s = edge_total_time_s + max_transmission_time + actual_idle_time_s

        # Optional debug print:
        # print(f"Layer {layer} | Edge Time: {edge_total_time_s*1000:.2f} ms | "
        #       f"Transmission: {max_transmission_time*1000:.2f} ms | "
        #       f"Idle: {actual_idle_time_s*1000:.2f} ms | "
        #       f"Total Time: {completion_time_s*1000:.2f} ms | Energy: {total_energy:.4f} J, "
        #       f"action: {current_action.tolist()}")

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
        Reward = High when energy is low and completion is within the fractional deadline.
        If isA2C=True, the raw (negative) reward is normalized into [0, 10].
        """

        fractional_deadline_ms = (
            self.profiling.get_edge_time_for_layer(layer)
            / self.profiling.get_total_edge_time()
        ) * self.profiling.deadline

        completion_time_ms = completion_time_s * 1000.0
        layer_surplus_ms = fractional_deadline_ms + previous_surplus - completion_time_ms

        lambda_param_e = 5.0
        lambda_param_d = 5.0

        if layer_surplus_ms < 0:
            delay_penalty = abs(layer_surplus_ms) * lambda_param_d
        else:
            delay_penalty = 0.0

        norm_surplus = layer_surplus_ms / 1000.0
        sigmoid_weight = 1 / (1 + np.exp(-norm_surplus))

        energy_weight = lambda_param_e * sigmoid_weight
        delay_weight = lambda_param_d * (1 - sigmoid_weight)

        total_penalty = (
            energy_weight * ((total_energy * 100) if not isA2C else total_energy)
            + delay_weight * delay_penalty
        )

        reward = -total_penalty  # <-- still negative = cost to minimize

        if layer_surplus_ms < 0:
            reward -= 10000 * abs(layer_surplus_ms / fractional_deadline_ms)

        if isA2C:
            reward = 10 * np.tanh(reward / 1000.0)

        return reward, layer_surplus_ms, negative_surplus_count, fractional_deadline_ms
    
    def run_full_task(self, action_plan, initial_bandwidth):
        """
        Execute a full DNN task given a complete assignment plan (list of per-layer actions).

        Each action in 'action_plan' is an array of shape (num_nodes_in_layer, 2),
        where [:,0] = layer index, [:,1] = {0=edge, 1=cloud}.

        Returns:
            total_energy (float): total energy consumed for all layers.
            total_time (float): total completion time in milliseconds.
            total_reward (float): reward for the full task (computed from total performance).
        """

        # Initialize running states
        bandwidth = initial_bandwidth
        cloud_pending_ms = 0.0
        surplus = 0.0
        negative_surplus_count = 0
        total_energy = 0.0
        total_time_ms = 0.0
        total_reward = 0.0

        # Step through each layer sequentially
        # print(f"Executing full task with action plan: {[a for a in action_plan]}")
        for layer_idx, action in enumerate(action_plan):
            current_state = (
                bandwidth,          # current bandwidth
                cloud_pending_ms,   # cloud waiting time
                layer_idx,          # current layer
                None if layer_idx == 0 else action_plan[layer_idx - 1],
                surplus,
                negative_surplus_count,
            )

            # Compute next state cloud waiting time (depends on this layer's offloading)
            next_state_cloud_processing = self.get_next_state_cloud_waiting_time(
                next_layer=layer_idx if (layer_idx + 1) < len(self.profiling.layers) else layer_idx,
                current_action=action,
                isAllCloud=False,
            )

            # Compute energy and time for this layer
            energy, completion_time_s = self.compute_energy_and_time(
                current_state=current_state,
                current_action=action,
                cloud_pending_ms=next_state_cloud_processing,
            )

            total_energy += energy
            total_time_ms += completion_time_s * 1000  # convert s → ms

            # Compute layer-level reward components (surplus etc.)
            reward, surplus, __, _ = self.calculate_reward(
                layer_idx, energy, completion_time_s, surplus, negative_surplus_count, isA2C=False
            )
            total_reward += reward

            # Compute next state (for continuity)
            next_state, terminal, _ = self.get_next_state(
                current_state,
                action,
                0,
                0,
                new_cloud_pending=next_state_cloud_processing,
            )

            # Update dynamic variables
            bandwidth, cloud_pending_ms = next_state[0], next_state[1]

            if terminal:
                break

        # # Final reward for full task — use total energy/time
        # total_reward, _, _, _ = self.calculate_reward(
        #     len(action_plan) - 1, total_energy, total_time_ms , surplus, negative_surplus_count, isA2C=False
        # )

        # print(f"Full Task | Total Energy: {total_energy:.4f} J | Total Time: {total_time_ms:.2f} ms | Total Reward: {total_reward:.3f}")

        return total_energy, total_time_ms, total_reward, bandwidth


