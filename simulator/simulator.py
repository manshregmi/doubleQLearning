import numpy as np
import random
import pandas as pd
from bisect import bisect_left
from typing import List, Tuple, Optional
from profiling.profile import ProfilingData


class CloudEdgeSimulator:
    """Level-wise cloud-edge offloading simulator with ternary action space.

    Action modes per node:
        0 = local execution on edge device
        1 = optimistic cloud execution (result download deferred)
        2 = conservative cloud execution (result download immediate)
    """

    def __init__(self, profiling_data: ProfilingData,
                 trace_csv_path: Optional[str] = None,
                 timeout_threshold_ms: float = 150.0,
                 packet_loss_prob: float = 0.10):
        self.profiling = profiling_data
        self.timeout_threshold_ms = float(timeout_threshold_ms)
        self.packet_loss_prob = float(packet_loss_prob)

        self.trace_tracker = None
        self.cumulative_time_seconds = 0.0
        self.episode_offset = 0.0
        self.cumulative_energy_joules = 0.0

        self.optimistic_chain_start = -1
        self.data_location = 'EDGE'
        self.optimistic_chain_length = 0
        self.last_timeout_occurred = False
        self.last_action_mode = -1
        self.last_neg_count = 0

        self.edge_idle_power = getattr(profiling_data, 'edge_idle_power', 1.0)
        self.edge_comm_power = getattr(profiling_data, 'edge_communication_power', 2.0)

    # -- configuration setters (for sweeping) --
    def set_timeout_threshold(self, ms: float):
        self.timeout_threshold_ms = float(ms)

    def set_packet_loss(self, p: float):
        self.packet_loss_prob = float(p)

    def reset_episode_time(self):
        self.cumulative_time_seconds = 0.0
        self.cumulative_energy_joules = 0.0
        if self.trace_tracker:
            max_time = self.trace_tracker.normalized_timestamps[-1]
            self.episode_offset = random.uniform(0, max_time * 0.7)
        else:
            self.episode_offset = 0.0
        self.optimistic_chain_start = -1
        self.data_location = 'EDGE'
        self.optimistic_chain_length = 0
        self.last_timeout_occurred = False
        self.last_action_mode = -1
        self.last_neg_count = 0

    def get_current_bandwidth(self, bandwidth) -> float:
        # if self.trace_tracker:
        #     query = self.cumulative_time_seconds + self.episode_offset
        #     bw = self.trace_tracker.get_bandwidth_at_time(query, use_normalized=True)
        #     return float(max(0.5, bw / 8.0))  # Mbps -> MBps
        # return float(random.uniform(2.0, 8.0))

        bw_change_p = random.random()
        bw_change_n = - random.random()
        bw_change = bw_change_n + bw_change_p
        # bw_change = 0
        new_bandwidth = max(1.0, min(bandwidth + bw_change, 15.0))
        return float(new_bandwidth)

    def get_current_rtt(self) -> float:
        return 5

    def get_possible_actions(self, layer):
        if layer >= len(self.profiling.layers):
            return []
        nodes = self.profiling.get_num_nodes(layer)
        actions = []
        for pattern in range(3 ** nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            temp = pattern
            for i in range(nodes):
                a[i, 1] = temp % 3
                temp //= 3
            actions.append(a)
        return actions

    def get_next_state_cloud_waiting_time(self, current_layer, current_action, isAllCloud=False):
        layer = int(current_layer)
        cloud_nodes = np.where((current_action[:, 1] == 1) | (current_action[:, 1] == 2))[0]

        if not isAllCloud:
            n_competing = random.randint(0, max(0, self.profiling.numberOfEdgeDevice - 1))
            congestion = abs(self.profiling.get_max_layer_cloud_time(layer) * n_competing *
                             np.random.uniform(0.1, 0.5))
        else:
            congestion = 0.0

        new_cloud_pending = congestion
        if len(cloud_nodes) > 0:
            cloud_proc_ms = max(self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes)
            new_cloud_pending += max(0.0, cloud_proc_ms)

        if isAllCloud and len(cloud_nodes) > 0:
            cloud_proc_ms = max(self.profiling.get_node_cloud_time(layer, i) for i in cloud_nodes)
            new_cloud_pending = cloud_proc_ms * self.profiling.numberOfEdgeDevice

        return new_cloud_pending

    def get_next_state(self, current_state, action, new_cloud_pending, surplus, neg_count):
        bandwidth, _, _, layer, _, _, _, _, timeout_bin = current_state
        layer = int(layer)

        bw = self.get_current_bandwidth(bandwidth)
        rtt = self.get_current_rtt()

        if layer + 1 < len(self.profiling.layers):
            next_layer = layer + 1
            terminal = False
        else:
            next_layer = layer
            terminal = True

        # Stored as a flat tuple of per-node modes, e.g. (0, 2, 1).
        # compute_energy_and_time handles this format (and the full array too).
        prev_action_pattern = tuple(int(x) for x in action[:, 1])

        # FIX: on timeout the cloud result is abandoned and the layer is
        # recomputed on the edge, so its output physically lives on EDGE.
        # Record it as all-edge so the next layer's transfer is correct.
        # Requires compute_energy_and_time() to run BEFORE get_next_state().
        if self.last_timeout_occurred:
            prev_action_pattern = tuple(0 for _ in prev_action_pattern)

        next_state = (
            bw,
            rtt,
            new_cloud_pending,
            next_layer,
            prev_action_pattern,
            surplus,
            neg_count,
            self.optimistic_chain_length,
            timeout_bin,
        )
        return next_state, terminal

    @staticmethod
    def _extract_assignments(action) -> np.ndarray:
        """Return per-node modes from either a (nodes, 2) action array
        or a flat tuple/list of modes."""
        arr = np.asarray(action, dtype=int)
        if arr.ndim == 2:
            return arr[:, 1]
        return arr.reshape(-1)

    def compute_energy_and_time(self, current_state, current_action, cloud_pending_ms):
        """
        Compute energy and completion time for actions:
            0 = EDGE
            1 = OPTIMISTIC CLOUD
            2 = CONSERVATIVE CLOUD

        The original timing model is preserved for successful executions.
        Connection loss / timeout is added as a failure mechanism.

        On timeout/loss:
            - device waits until the timeout threshold
            - cloud result is abandoned
            - fallback computation is performed on the edge
            - the full failed transmission time is NOT counted
            - the full failed transmission energy is NOT charged

        Returns:
            total_energy (J)
            completion_time_s (seconds)
        """

        # ==============================================================
        # State
        # ==============================================================
        bandwidth, rtt_ms, _, layer, prev_action, _, _, _, _ = current_state
        layer = int(layer)

        profiling = self.profiling
        deps = profiling.dependencies

        # --------------------------------------------------------------
        # Action mode (multi-node: any 2 -> conservative,
        # else any 1 -> optimistic, else edge)
        # --------------------------------------------------------------
        action_values = np.asarray(current_action[:, 1], dtype=int)

        if np.any(action_values == 2):
            action_mode = 2
        elif np.any(action_values == 1):
            action_mode = 1
        else:
            action_mode = 0

        self.last_action_mode = action_mode
        any_cloud = bool(np.any(action_values != 0))

        # NOTE: units — data_mb / safe_bw assumes bandwidth is in MB/s.
        # bandwidth from get_current_bandwidth is labelled Mbps in logs.
        # Verify units of get_input_size()/get_output_size() before changing.
        safe_bw = max(float(bandwidth), 0.5)
        rtt_s = float(rtt_ms) / 1000.0

        # ==============================================================
        # Transmission time
        # ==============================================================
        transmission_times = []

        if prev_action is not None and layer > 0:
            # FIX: prev_action is stored by get_next_state as a flat tuple
            # of modes, not a (nodes, 2) array.
            prev_assignments = self._extract_assignments(prev_action)
            curr_assignments = np.asarray(current_action[:, 1], dtype=int)

            for curr_node in range(len(curr_assignments)):
                parent_nodes = deps.get((layer, curr_node), [])

                for (p_layer, p_node) in parent_nodes:
                    if p_layer == layer - 1 and p_node < len(prev_assignments):
                        parent_loc = int(prev_assignments[p_node])
                    else:
                        parent_loc = 0

                    curr_loc = int(curr_assignments[curr_node])

                    # FIX: compare physical location (edge vs cloud), not mode.
                    # Modes 1 and 2 are both on the cloud, so 1 -> 2 needs no transfer.
                    parent_on_cloud = parent_loc != 0
                    curr_on_cloud = curr_loc != 0

                    if parent_on_cloud != curr_on_cloud:
                        output_size = profiling.get_output_size(layer, curr_node)
                        data_mb = output_size / 1024.0
                        transmission_times.append(max(data_mb / safe_bw, rtt_s))
        else:
            # First layer: upload input if at least one node executes in cloud.
            if any_cloud:
                input_size = profiling.get_input_size()
                data_mb = input_size / 1024.0
                transmission_times.append(max(data_mb / safe_bw, rtt_s))

        max_transmission_time = max(transmission_times) if transmission_times else 0.0

        # ==============================================================
        # Edge computation
        # ==============================================================
        edge_times = []
        edge_energy = []

        for i in range(len(current_action)):
            if current_action[i, 1] == 0:
                node_t_s = profiling.get_node_edge_time(layer, i) / 1000.0
                node_p = profiling.get_node_edge_power(layer, i)
                edge_times.append(node_t_s)
                edge_energy.append(node_p * node_t_s)

        # Preserve old parallelism model
        if layer in [3, 5]:
            edge_total_time_s = max(edge_times) if edge_times else 0.0
            edge_energy_total = max(edge_energy) if edge_energy else 0.0
        else:
            edge_total_time_s = sum(edge_times) if edge_times else 0.0
            edge_energy_total = sum(edge_energy) if edge_energy else 0.0

        # ==============================================================
        # Cloud waiting / overlap
        # ==============================================================
        cloud_pending_s = float(cloud_pending_ms) / 1000.0

        if any_cloud:
            # Edge computation overlaps part of the cloud waiting interval.
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)
        else:
            actual_idle_time_s = 0.0

        # ==============================================================
        # Connection loss / timeout
        # ==============================================================
        upload_lost = False
        if any_cloud:
            upload_lost = random.random() < self.packet_loss_prob

        # Only conservative mode has an immediate download phase.
        download_lost = False
        if action_mode == 2:
            download_lost = random.random() < self.packet_loss_prob

        total_wait_s = max_transmission_time + cloud_pending_s
        timeout_s = self.timeout_threshold_ms / 1000.0

        timeout_by_time = any_cloud and (total_wait_s > timeout_s)

        timeout_occurred = any_cloud and (upload_lost or download_lost or timeout_by_time)
        self.last_timeout_occurred = bool(timeout_occurred)

        # ==============================================================
        # Recompute after failure
        # ==============================================================
        recompute_energy = 0.0
        recompute_time = 0.0

        # FIX: the chain start is only updated in "Chain management" below,
        # so on the FIRST optimistic layer it is still -1 here. Resolve it now
        # so a failure on that layer still recomputes it.
        if action_mode == 1 and self.optimistic_chain_start == -1:
            chain_start = layer
        else:
            chain_start = self.optimistic_chain_start

        if timeout_occurred:
            if action_mode == 2:
                # Conservative: recompute only the current layer's cloud nodes on EDGE.
                for i in range(len(current_action)):
                    if current_action[i, 1] != 0:
                        node_t_s = profiling.get_node_edge_time(layer, i) / 1000.0
                        node_p = profiling.get_node_edge_power(layer, i)
                        recompute_time += node_t_s
                        recompute_energy += node_p * node_t_s

            elif action_mode == 1:
                # Optimistic: the whole optimistic chain (including this layer)
                # is recomputed locally.
                for l in range(chain_start, layer + 1):
                    for i in range(profiling.get_num_nodes(l)):
                        node_t_s = profiling.get_node_edge_time(l, i) / 1000.0
                        node_p = profiling.get_node_edge_power(l, i)
                        recompute_time += node_t_s
                        recompute_energy += node_p * node_t_s

        # ==============================================================
        # Energy
        # ==============================================================
        # Failed execution only pays communication/idle energy until timeout.
        if timeout_occurred:
            communication_time_s = min(max_transmission_time, timeout_s)
            idle_time_for_energy = min(actual_idle_time_s, timeout_s)
        else:
            communication_time_s = max_transmission_time
            idle_time_for_energy = actual_idle_time_s

        comm_energy = self.edge_comm_power * communication_time_s
        idle_energy = self.edge_idle_power * idle_time_for_energy

        total_energy = edge_energy_total + idle_energy + comm_energy + recompute_energy

        # ==============================================================
        # Completion time
        # ==============================================================
        if timeout_occurred:
            # Device stops waiting at the timeout, then recomputes locally.
            waited_s = min(total_wait_s, timeout_s)
            completion_time_s = edge_total_time_s + waited_s + recompute_time
        else:
            waited_s = max_transmission_time + actual_idle_time_s
            completion_time_s = edge_total_time_s + max_transmission_time + actual_idle_time_s

        # ==============================================================
        # Chain management
        # ==============================================================
        if action_mode == 0:
            self.optimistic_chain_start = -1
            self.data_location = 'EDGE'
        elif action_mode == 1:
            if self.optimistic_chain_start == -1:
                self.optimistic_chain_start = layer
            self.data_location = 'CLOUDLET'
        else:
            # Conservative
            self.data_location = 'EDGE'
            self.optimistic_chain_start = -1

        # Failed optimistic execution falls back to EDGE.
        if timeout_occurred and action_mode == 1:
            self.optimistic_chain_start = -1
            self.data_location = 'EDGE'

        if self.optimistic_chain_start == -1:
            self.optimistic_chain_length = 0
        else:
            self.optimistic_chain_length = min(7, layer - self.optimistic_chain_start + 1)

        # ==============================================================
        # Cumulative statistics
        # ==============================================================
        self.cumulative_time_seconds += completion_time_s
        self.cumulative_energy_joules += total_energy

        # ==============================================================
        # Debug output
        # ==============================================================
        # print(
        #     f"Layer {layer} | "
        #     f"Cloud wait: {cloud_pending_ms:.2f} ms | "
        #     f"Edge time: {edge_total_time_s * 1000:.2f} ms | "
        #     f"Transmission: {max_transmission_time * 1000:.2f} ms | "
        #     f"Waited: {waited_s * 1000:.2f} ms | "
        #     f"Recompute: {recompute_time * 1000:.2f} ms | "
        #     f"Total time: {completion_time_s * 1000:.2f} ms | "
        #     f"Energy: {total_energy:.4f} J | "
        #     f"Action: {action_values.tolist()} | "
        #     f"Bandwidth: {bandwidth:.2f} Mbps | "
        #     f"Timeout thr: {self.timeout_threshold_ms:.1f} ms | "
        #     f"Upload loss: {upload_lost} | "
        #     f"Download loss: {download_lost} | "
        #     f"Timeout: {timeout_occurred}"
        # )

        return total_energy, completion_time_s

    def calculate_reward(self, layer, total_energy, completion_time_s,
                         previous_surplus=0.0, negative_surplus_count=0, isA2C=False):
        # Energy penalty
        energy_penalty = total_energy * 30.0

        # Timeout penalty
        timeout_penalty = 0.0
        if self.last_timeout_occurred:
            timeout_penalty = 200.0 * (1 + self.optimistic_chain_length)

        # Small optimistic penalty even without timeout
        optimistic_penalty = 0.0
        if self.last_action_mode == 1:
            optimistic_penalty = 20.0

        # Catastrophic energy penalty
        recompute_penalty = 0.0
        if total_energy > 1000.0:
            recompute_penalty = 500.0 * (total_energy / 1000.0)

        reward = -(energy_penalty + timeout_penalty + optimistic_penalty + recompute_penalty)
        reward = float(np.clip(reward, -5000.0, 50.0))

        if isA2C:
            reward *= 0.15

        # Fractional deadline tracking
        fractional_deadline_ms = (
            self.profiling.get_edge_time_for_layer(layer)
            / max(self.profiling.get_total_edge_time(), 1e-9)
        ) * getattr(self.profiling, 'deadline', 500.0)
        completion_time_ms = completion_time_s * 1000.0
        effective_deadline_ms = fractional_deadline_ms + previous_surplus
        surplus_ms = effective_deadline_ms - completion_time_ms
        if completion_time_ms > effective_deadline_ms:
            negative_surplus_count += 1

        self.last_neg_count = negative_surplus_count
        return reward, surplus_ms, negative_surplus_count, fractional_deadline_ms