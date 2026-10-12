import numpy as np
import random
import pandas as pd
from typing import Optional
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
        after an ACK loss the layer is recomputed locally -> recorded as 0

    Network-loss model (RTT-driven ACK loss only):
        - RTT is a persistent random walk, initial 4.5 ms, step U(-0.5, 0.5),
          clipped to [1.0, 6.0]. Bandwidth is a persistent random walk too.
        - Every upload and every download is followed by an ACK. The ACK is
          lost with probability  p_loss(rtt) = clip((rtt - 4.5)/1.5, 0, 1).
        - Upload and download ACKs are drawn independently.
        - On ACK loss the current layer (and, for optimistic, the whole
          optimistic chain) is recomputed locally with no timeout wait.
          Detection costs one RTT of idle wait; the attempted transfer is
          charged at comm power.
        - `recent_ack_loss_count` tracks losses over recent layers (0-3),
          reset to 0 on a clean layer; it is part of the observation.

    Reward (paper, constrained formulation):
        R_imm   = -eta * E                                (Eq. 10)
        R_final = R_imm - phi * max(0, delta - tau)/tau   (Eq. 11, in runner)
    """

    def __init__(self, profiling_data: ProfilingData,
                 trace_csv_path: Optional[str] = None,
                 idle_wait_power_factor: float = 0.1,
                 max_cloud_pending_ms: float = 500.0,
                 max_competing_devices: int = 3):
        self.profiling = profiling_data

        self.trace_tracker = None
        self.cumulative_time_seconds = 0.0
        self.episode_offset = 0.0
        self.cumulative_energy_joules = 0.0

        self.optimistic_chain_start = -1
        self.data_location = 'EDGE'
        self.optimistic_chain_length = 0
        self.last_ack_loss_occurred = False
        self.last_action_mode = -1
        self.last_neg_count = 0
        self.recent_ack_loss_count = 0

        self.edge_idle_power = getattr(profiling_data, 'edge_idle_power', 1.0)
        self.edge_comm_power = getattr(profiling_data, 'edge_communication_power', 2.0)
        self.idle_wait_power_factor = float(idle_wait_power_factor)

        self.max_cloud_pending_ms = float(max_cloud_pending_ms)
        self.max_competing_devices = int(max_competing_devices)

        # ---- Persistent link state (RTT + bandwidth) ----
        self.rtt_baseline_ms = 4.5
        self.rtt_step_ms = 1          # was 0.25

        # New: scale/floor for ACK-loss timeout probability
        self.ack_loss_gain = 2.0         # >1 makes timeout more likely
        self.ack_loss_floor = 0.0    
        self.current_rtt_ms = self.rtt_baseline_ms

        self.bandwidth_step = 1.0
        self.bandwidth_min = 1.0
        self.bandwidth_max = 15.0
        self.current_bandwidth = float(getattr(profiling_data, 'bandwidth', 8.0))

        # ---- Reward configuration (paper Eq. 10) ----
        self.eta = 150.0
        self.optimistic_penalty = 0.0

        self.verbose = False

    # -- network-loss helpers --
    def ack_loss_prob(self, rtt_ms: float) -> float:
        base = (float(rtt_ms) - self.rtt_baseline_ms) / max(
            self.rtt_max_ms - self.rtt_baseline_ms, 1e-9
        )
        p = self.ack_loss_floor + (1.0 - self.ack_loss_floor) * base * self.ack_loss_gain
        return float(np.clip(p, 0.0, 1.0))

    def _sample_ack(self) -> bool:
        return random.random() < self.ack_loss_prob(self.current_rtt_ms)

    def reset_episode_time(self):
        self.cumulative_time_seconds = 0.0
        self.cumulative_energy_joules = 0.0
        self.episode_offset = 0.0

        self.optimistic_chain_start = -1
        self.data_location = 'EDGE'
        self.optimistic_chain_length = 0
        self.last_ack_loss_occurred = False
        self.last_action_mode = -1
        self.last_neg_count = 0
        self.recent_ack_loss_count = 0

    # -- link evolution --
    def get_current_bandwidth(self) -> float:
        delta = random.uniform(-self.bandwidth_step, self.bandwidth_step)
        self.current_bandwidth = float(np.clip(
            self.current_bandwidth + delta,
            self.bandwidth_min, self.bandwidth_max))
        return self.current_bandwidth

    def get_current_rtt(self) -> float:
        delta = random.uniform(-self.rtt_step_ms, self.rtt_step_ms)
        self.current_rtt_ms = float(np.clip(
            self.current_rtt_ms + delta,
            self.rtt_min_ms, self.rtt_max_ms))
        return self.current_rtt_ms

    # -- misc --
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
            n_competing = random.randint(0, min(
                max(0, self.profiling.numberOfEdgeDevice - 1),
                self.max_competing_devices))
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

        new_cloud_pending = float(np.clip(new_cloud_pending, 0.0, self.max_cloud_pending_ms))
        return new_cloud_pending

    def get_next_state(self, current_state, action, new_cloud_pending, surplus, neg_count):
        bandwidth, rtt, _, layer, _, _, _, _, context, _ = current_state
        layer = int(layer)

        bw = self.get_current_bandwidth()
        rtt_next = self.get_current_rtt()

        if layer + 1 < len(self.profiling.layers):
            next_layer = layer + 1
            terminal = False
        else:
            next_layer = layer
            terminal = True

        prev_action_pattern = tuple(int(x) for x in action[:, 1])

        if self.last_ack_loss_occurred:
            prev_action_pattern = tuple(0 for _ in prev_action_pattern)

        next_state = (
            bw,
            rtt_next,
            new_cloud_pending,
            next_layer,
            prev_action_pattern,
            surplus,
            neg_count,
            self.optimistic_chain_length,
            context,
            self.recent_ack_loss_count,   # new field
        )
        return next_state, terminal

    @staticmethod
    def _extract_assignments(action) -> np.ndarray:
        arr = np.asarray(action, dtype=int)
        if arr.ndim == 2:
            return arr[:, 1]
        return arr.reshape(-1)

    def _layer_edge_cost(self, layer, node_indices):
        t_total, e_total = 0.0, 0.0
        for i in node_indices:
            t = self.profiling.get_node_edge_time(layer, i) / 1000.0
            p = self.profiling.get_node_edge_power(layer, i)
            t_total += t
            e_total += p * t
        return t_total, e_total

    def compute_energy_and_time(self, current_state, current_action, cloud_pending_ms):
        """Compute energy (J) and completion time (s) for one layer.

        Energy accounting:
            - Edge computation energy.
            - Comm energy: edge_comm_power for actual transfer time only.
            - Idle energy: idle_wait_power_factor * edge_idle_power for
              (a) cloud wait not overlapped by local compute, and
              (b) one-RTT ACK-detection wait on a lost ACK.
            - Recompute energy: local execution of the failed layer and any
              active optimistic chain, on ACK loss.
        """
        bandwidth, rtt_ms, _, layer, prev_action, _, _, _, _, recent_ack_loss = current_state
        layer = int(layer)

        bandwidth = float(self.current_bandwidth)
        rtt_ms = float(self.current_rtt_ms)

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

        safe_bw = max(float(bandwidth), 0.5)
        rtt_s = rtt_ms / 1000.0

        def xfer_time(size_kb):
            return (size_kb / 1024.0) / safe_bw

        chain_active = self.optimistic_chain_start != -1

        # ---- Inbound transfers (upload / deferred download) ----
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
                        parent_loc = 0

                    if parent_loc == 2:
                        continue
                    parent_on_cloud = parent_loc == 1
                    if parent_on_cloud == curr_on_cloud:
                        continue

                    t = xfer_time(profiling.get_output_size(p_layer, p_node))
                    if curr_on_cloud:
                        upload_times.append(t)
                    else:
                        deferred_download_times.append(t)
        else:
            if any_cloud:
                upload_times.append(xfer_time(profiling.get_input_size(0, 0)))

        upload_time_s = max(upload_times) if upload_times else 0.0
        deferred_dl_time_s = max(deferred_download_times) if deferred_download_times else 0.0

        # ---- Immediate download of conservative results (this layer) ----
        immediate_dl = [xfer_time(profiling.get_output_size(layer, i))
                        for i, v in enumerate(action_values) if v == 2]
        immediate_dl_time_s = max(immediate_dl) if immediate_dl else 0.0

        # ---- Edge computation ----
        edge_times, edge_energy = [], []
        for i in edge_node_idx:
            t = profiling.get_node_edge_time(layer, i) / 1000.0
            edge_times.append(t)
            edge_energy.append(profiling.get_node_edge_power(layer, i) * t)

        if layer in [3, 5]:
            edge_total_time_s = max(edge_times) if edge_times else 0.0
            edge_energy_total = max(edge_energy) if edge_energy else 0.0
        else:
            edge_total_time_s = sum(edge_times)
            edge_energy_total = sum(edge_energy)

        # ---- Cloud wait / overlap ----
        cloud_pending_s = float(cloud_pending_ms) / 1000.0
        if any_cloud:
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)
        else:
            actual_idle_time_s = 0.0

        # ---- ACK-loss evaluation ----
        up_attempted = any_cloud
        down_attempted = bool(immediate_dl) or bool(deferred_download_times)

        upload_ack_lost = up_attempted and self._sample_ack()
        download_ack_lost = down_attempted and self._sample_ack()
        ack_loss = bool(upload_ack_lost or download_ack_lost)
        self.last_ack_loss_occurred = ack_loss

        # ---- Update recent-ACK-loss counter (part of next state) ----
        if ack_loss:
            self.recent_ack_loss_count = min(3, self.recent_ack_loss_count + 1)
        else:
            self.recent_ack_loss_count = 0

        failure = ack_loss

        # ---- Recompute after failure ----
        recompute_time = 0.0
        recompute_energy = 0.0

        if failure:
            t, e = self._layer_edge_cost(layer, cloud_node_idx)
            recompute_time += t
            recompute_energy += e

            if chain_active:
                for l in range(self.optimistic_chain_start, layer):
                    t, e = self._layer_edge_cost(l, range(profiling.get_num_nodes(l)))
                    recompute_time += t
                    recompute_energy += e

        # ---- Transfer vs. RTT-wait separation ----
        if not ack_loss:
            upload_paid_s = upload_time_s if up_attempted else 0.0
            download_paid_s = ((deferred_dl_time_s + immediate_dl_time_s)
                               if down_attempted else 0.0)
            transfer_time_s = upload_paid_s + download_paid_s
            rtt_wait_s = 0.0
            waited_s = (upload_time_s + deferred_dl_time_s + immediate_dl_time_s
                        + actual_idle_time_s)
        else:
            upload_paid_s = upload_time_s if up_attempted else 0.0
            download_paid_s = ((deferred_dl_time_s + immediate_dl_time_s)
                               if down_attempted else 0.0)
            transfer_time_s = upload_paid_s + download_paid_s
            rtt_wait_s = rtt_s
            waited_s = transfer_time_s + rtt_wait_s

        comm_energy = self.edge_comm_power * transfer_time_s
        idle_energy = (self.edge_idle_power * self.idle_wait_power_factor
                       * (actual_idle_time_s + rtt_wait_s))

        total_energy = edge_energy_total + idle_energy + comm_energy + recompute_energy

        # ---- Completion time ----
        if failure:
            completion_time_s = edge_total_time_s + waited_s + recompute_time
        else:
            completion_time_s = edge_total_time_s + waited_s

        # ---- Chain bookkeeping ----
        if failure:
            self.optimistic_chain_start = -1
            self.data_location = 'EDGE'
        elif action_mode == 1:
            if self.optimistic_chain_start == -1:
                self.optimistic_chain_start = layer
            self.data_location = 'CLOUDLET'
        else:
            self.optimistic_chain_start = -1
            self.data_location = 'EDGE'

        if self.optimistic_chain_start == -1:
            self.optimistic_chain_length = 0
        else:
            self.optimistic_chain_length = min(7, layer - self.optimistic_chain_start + 1)

        # ---- Cumulative stats ----
        self.cumulative_time_seconds += completion_time_s
        self.cumulative_energy_joules += total_energy

        if self.verbose:
            print(
                f"Layer {layer} | "
                f"RTT: {rtt_ms:.2f} ms | "
                f"p_loss: {self.ack_loss_prob(rtt_ms):.2f} | "
                f"Cloud wait: {cloud_pending_ms:.2f} ms | "
                f"Edge time: {edge_total_time_s * 1000:.2f} ms | "
                f"Upload: {upload_time_s * 1000:.2f} ms | "
                f"Deferred DL: {deferred_dl_time_s * 1000:.2f} ms | "
                f"Immediate DL: {immediate_dl_time_s * 1000:.2f} ms | "
                f"Transfer time: {transfer_time_s * 1000:.2f} ms | "
                f"RTT wait: {rtt_wait_s * 1000:.2f} ms | "
                f"Recompute: {recompute_time * 1000:.2f} ms | "
                f"Total time: {completion_time_s * 1000:.2f} ms | "
                f"Edge E: {edge_energy_total:.4f} J | "
                f"Idle E: {idle_energy:.4f} J | "
                f"Comm E: {comm_energy:.4f} J | "
                f"Recompute E: {recompute_energy:.4f} J | "
                f"Total E: {total_energy:.4f} J | "
                f"Action: {action_values.tolist()} | "
                f"BW: {bandwidth:.2f} MB/s | "
                f"UpACK lost: {upload_ack_lost} | "
                f"DownACK lost: {download_ack_lost} | "
                f"recent_ack_loss: {self.recent_ack_loss_count}"
            )

        return total_energy, completion_time_s

    def calculate_reward(self, layer, total_energy, completion_time_s,
                         previous_surplus=0.0, negative_surplus_count=0, isA2C=False):
        """Immediate reward = -eta * E  (paper Eq. 10)."""
        reward = -self.eta * float(total_energy)

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