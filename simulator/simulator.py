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
        1 = optimistic cloud execution   (result stays on cloudlet; download
                                          deferred until an edge layer needs it)
        2 = conservative cloud execution (result downloaded immediately; data
                                          then available on BOTH edge and cloud)

    Data-location semantics used for transfers (per parent node):
        parent 0 -> data on EDGE   : cloud child needs an UPLOAD
        parent 1 -> data on CLOUD  : edge child needs a (deferred) DOWNLOAD
        parent 2 -> data on BOTH   : no transfer needed for any child
        after a timeout the layer is recomputed locally -> recorded as 0 (EDGE)

    Reward (paper, constrained formulation):
        R_imm = -eta * E                              (Eq. 10)
        R_final = R_imm - phi * max(0, delta - tau)/tau   (Eq. 11, applied
                                                           only on violation
                                                           in the runner)
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

        # ---- Reward configuration (paper Eq. 10) ----
        # Only energy drives the immediate reward. Timeouts and recomputation
        # show up through the energy they add; there is no separate per-step
        # penalty that would bias the constrained objective.
        self.eta = 150.0
        self.optimistic_penalty = 0.0   # kept for API compatibility, unused

        # Per-layer debug print. Keep False for long training runs.
        self.verbose = False

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
        bw_change_p = random.random()
        bw_change_n = - random.random()
        bw_change = bw_change_n + bw_change_p
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

        # Flat tuple of per-node modes, e.g. (0, 2, 1).
        prev_action_pattern = tuple(int(x) for x in action[:, 1])

        # On timeout the layer was recomputed locally: its output lives on EDGE.
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
        """Per-node modes from either a (nodes, 2) array or a flat tuple."""
        arr = np.asarray(action, dtype=int)
        if arr.ndim == 2:
            return arr[:, 1]
        return arr.reshape(-1)

    def _layer_edge_cost(self, layer, node_indices):
        """Sequential edge time (s) and energy (J) for the given nodes of a layer."""
        t_total, e_total = 0.0, 0.0
        for i in node_indices:
            t = self.profiling.get_node_edge_time(layer, i) / 1000.0
            p = self.profiling.get_node_edge_power(layer, i)
            t_total += t
            e_total += p * t
        return t_total, e_total

    def compute_energy_and_time(self, current_state, current_action, cloud_pending_ms):
        """
        Compute energy (J) and completion time (s) for one layer.

        Network phases for this layer:
            upload            : edge -> cloud (EDGE parent, cloud child; or input)
            deferred download : cloud -> edge (OPTIMISTIC parent, edge child)
            immediate download: result of CONSERVATIVE nodes in this layer

        Any network phase can be lost (packet_loss_prob) or exceed the timeout.
        The device cannot observe a loss directly, so on ANY failure it waits the
        full timeout threshold, abandons the cloud result and recomputes locally:
            - the current layer's cloud nodes, and
            - the preceding optimistic chain, if one is active (its results only
              existed on the cloud and are now unreachable).
        """

        # ==============================================================
        # State / action
        # ==============================================================
        bandwidth, rtt_ms, _, layer, prev_action, _, _, _, _ = current_state
        layer = int(layer)

        profiling = self.profiling
        deps = profiling.dependencies

        action_values = np.asarray(current_action[:, 1], dtype=int)

        if np.any(action_values == 2):
            action_mode = 2
        elif np.any(action_values == 1):
            action_mode = 1
        else:
            action_mode = 0
        self.last_action_mode = action_mode

        any_cloud = bool(np.any(action_values != 0))
        cloud_node_idx = [i for i, v in enumerate(action_values) if v != 0]
        edge_node_idx = [i for i, v in enumerate(action_values) if v == 0]

        # NOTE: units — data_mb / safe_bw assumes bandwidth in MB/s.
        safe_bw = max(float(bandwidth), 0.5)
        rtt_s = float(rtt_ms) / 1000.0
        timeout_s = self.timeout_threshold_ms / 1000.0

        def xfer_time(size_kb):
            return max((size_kb / 1024.0) / safe_bw, rtt_s)

        # Is an optimistic chain (cloud-only results) feeding this layer?
        chain_active = self.optimistic_chain_start != -1

        # ==============================================================
        # Inbound transfers (upload / deferred download)
        # ==============================================================
        upload_times = []
        deferred_download_times = []

        if prev_action is not None and layer > 0:
            prev_assignments = self._extract_assignments(prev_action)

            for curr_node, curr_loc in enumerate(action_values):
                curr_on_cloud = curr_loc != 0

                for (p_layer, p_node) in deps.get((layer, curr_node), []):
                    if p_layer == layer - 1 and p_node < len(prev_assignments):
                        parent_loc = int(prev_assignments[p_node])
                    else:
                        parent_loc = 0  # older layers: assume on EDGE

                    if parent_loc == 2:
                        continue  # conservative result is on both sides

                    parent_on_cloud = parent_loc == 1
                    if parent_on_cloud == curr_on_cloud:
                        continue  # data already where it is needed

                    # Transfer the PARENT's output tensor.
                    t = xfer_time(profiling.get_output_size(p_layer, p_node))
                    if curr_on_cloud:
                        upload_times.append(t)               # edge -> cloud
                    else:
                        deferred_download_times.append(t)    # cloud -> edge
        else:
            # First layer: upload model input if any node runs on the cloud.
            if any_cloud:
                upload_times.append(xfer_time(profiling.get_input_size()))

        upload_time_s = max(upload_times) if upload_times else 0.0
        deferred_dl_time_s = max(deferred_download_times) if deferred_download_times else 0.0
        inbound_time_s = max(upload_time_s, deferred_dl_time_s)

        # ==============================================================
        # Immediate download of conservative results (this layer)
        # ==============================================================
        immediate_dl = [xfer_time(profiling.get_output_size(layer, i))
                        for i, v in enumerate(action_values) if v == 2]
        immediate_dl_time_s = max(immediate_dl) if immediate_dl else 0.0

        # ==============================================================
        # Edge computation
        # ==============================================================
        edge_times, edge_energy = [], []
        for i in edge_node_idx:
            t = profiling.get_node_edge_time(layer, i) / 1000.0
            edge_times.append(t)
            edge_energy.append(profiling.get_node_edge_power(layer, i) * t)

        # Preserve old parallelism model
        if layer in [3, 5]:
            edge_total_time_s = max(edge_times) if edge_times else 0.0
            edge_energy_total = max(edge_energy) if edge_energy else 0.0
        else:
            edge_total_time_s = sum(edge_times)
            edge_energy_total = sum(edge_energy)

        # ==============================================================
        # Cloud waiting / overlap
        # ==============================================================
        cloud_pending_s = float(cloud_pending_ms) / 1000.0
        if any_cloud:
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)
        else:
            actual_idle_time_s = 0.0

        # ==============================================================
        # Loss / timeout
        # ==============================================================
        uses_network = any_cloud or bool(deferred_download_times)

        upload_lost = any_cloud and (random.random() < self.packet_loss_prob)
        downlink_needed = bool(immediate_dl) or bool(deferred_download_times)
        download_lost = downlink_needed and (random.random() < self.packet_loss_prob)

        total_wait_s = (inbound_time_s
                        + (cloud_pending_s if any_cloud else 0.0)
                        + immediate_dl_time_s)

        timeout_by_time = uses_network and (total_wait_s > timeout_s)
        timeout_occurred = uses_network and (upload_lost or download_lost or timeout_by_time)
        self.last_timeout_occurred = bool(timeout_occurred)

        # ==============================================================
        # Recompute after failure
        # ==============================================================
        recompute_time = 0.0
        recompute_energy = 0.0

        if timeout_occurred:
            # Current layer's cloud nodes -> edge
            t, e = self._layer_edge_cost(layer, cloud_node_idx)
            recompute_time += t
            recompute_energy += e

            # Preceding optimistic chain (cloud-only results are lost)
            if chain_active:
                for l in range(self.optimistic_chain_start, layer):
                    t, e = self._layer_edge_cost(l, range(profiling.get_num_nodes(l)))
                    recompute_time += t
                    recompute_energy += e

        # ==============================================================
        # Energy
        # ==============================================================
        comm_success_s = inbound_time_s + immediate_dl_time_s
        if timeout_occurred:
            # Radio active for (at most) the transfer, then idle until the
            # timeout fires.
            communication_time_s = min(comm_success_s, timeout_s)
            idle_time_for_energy = max(0.0, timeout_s - communication_time_s)
        else:
            communication_time_s = comm_success_s
            idle_time_for_energy = actual_idle_time_s

        comm_energy = self.edge_comm_power * communication_time_s
        idle_energy = self.edge_idle_power * idle_time_for_energy
        total_energy = edge_energy_total + idle_energy + comm_energy + recompute_energy

        # ==============================================================
        # Completion time
        # ==============================================================
        if timeout_occurred:
            # A lost packet is only detected when the timer expires, and a
            # slow transfer is abandoned at the timer: either way the device
            # waits exactly the threshold.
            waited_s = timeout_s
            completion_time_s = edge_total_time_s + waited_s + recompute_time
        else:
            waited_s = inbound_time_s + actual_idle_time_s + immediate_dl_time_s
            completion_time_s = edge_total_time_s + waited_s

        # ==============================================================
        # Chain management
        # ==============================================================
        if timeout_occurred:
            # Everything was recomputed locally.
            self.optimistic_chain_start = -1
            self.data_location = 'EDGE'
        elif action_mode == 1:
            if self.optimistic_chain_start == -1:
                self.optimistic_chain_start = layer
            self.data_location = 'CLOUDLET'
        else:
            # Edge (chain collected) or conservative (result downloaded)
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
        if self.verbose:
            print(
                f"Layer {layer} | "
                f"Cloud wait: {cloud_pending_ms:.2f} ms | "
                f"Edge time: {edge_total_time_s * 1000:.2f} ms | "
                f"Upload: {upload_time_s * 1000:.2f} ms | "
                f"Deferred DL: {deferred_dl_time_s * 1000:.2f} ms | "
                f"Immediate DL: {immediate_dl_time_s * 1000:.2f} ms | "
                f"Waited: {waited_s * 1000:.2f} ms | "
                f"Recompute: {recompute_time * 1000:.2f} ms | "
                f"Total time: {completion_time_s * 1000:.2f} ms | "
                f"Energy: {total_energy:.4f} J | "
                f"Action: {action_values.tolist()} | "
                f"Bandwidth: {bandwidth:.2f} | "
                f"Timeout thr: {self.timeout_threshold_ms:.1f} ms | "
                f"Upload loss: {upload_lost} | "
                f"Download loss: {download_lost} | "
                f"Timeout: {timeout_occurred}"
            )

        return total_energy, completion_time_s

    def calculate_reward(self, layer, total_energy, completion_time_s,
                         previous_surplus=0.0, negative_surplus_count=0, isA2C=False):
        """Immediate reward = -eta * E  (paper Eq. 10).

        The terminal penalty (Eq. 11) is applied by the runner after all
        levels have been processed. Deadline information is returned as
        fractional surplus so it can go into the agent's state; it does NOT
        enter the reward.
        """
        reward = -self.eta * float(total_energy)

        # Fractional deadline tracking (state info only)
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