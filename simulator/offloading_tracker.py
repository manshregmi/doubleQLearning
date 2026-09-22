import numpy as np
import random
import pandas as pd
from bisect import bisect_left
from typing import List, Tuple, Optional
from profiling.profile import ProfilingData


class NetworkTraceTracker:
    def __init__(self, trace_data: List[Tuple[float, float, float]]):
        if not trace_data:
            raise ValueError("No trace data provided")
        trace_data.sort(key=lambda x: x[0])
        self.timestamps = np.array([t for t, _, _ in trace_data], dtype=float)
        self.bandwidths = np.array([bw for _, bw, _ in trace_data], dtype=float)
        self.rtts = np.array([rtt for _, _, rtt in trace_data], dtype=float)
        self.min_timestamp = float(self.timestamps[0])
        self.normalized_timestamps = self.timestamps - self.min_timestamp
        
        print(f"✅ Trace loaded: {len(trace_data)} samples")
        print(f"   BW range: {self.bandwidths.min():.2f} - {self.bandwidths.max():.2f} MBps")
        print(f"   RTT range: {self.rtts.min():.2f} - {self.rtts.max():.2f} ms")

    def get_bandwidth_at_time(self, time_seconds: float, use_normalized: bool = False) -> float:
        query = time_seconds if use_normalized else time_seconds - self.min_timestamp
        if query <= self.normalized_timestamps[0]:
            return float(self.bandwidths[0])
        if query >= self.normalized_timestamps[-1]:
            return float(self.bandwidths[-1])
        idx = bisect_left(self.normalized_timestamps, query)
        t0, t1 = self.normalized_timestamps[idx-1], self.normalized_timestamps[idx]
        b0, b1 = self.bandwidths[idx-1], self.bandwidths[idx]
        ratio = (query - t0) / (t1 - t0)
        return float(max(0.5, b0 + ratio * (b1 - b0)))

    def get_rtt_at_time(self, time_seconds: float, use_normalized: bool = False) -> float:
        query = time_seconds if use_normalized else time_seconds - self.min_timestamp
        if query <= self.normalized_timestamps[0]:
            return float(self.rtts[0])
        if query >= self.normalized_timestamps[-1]:
            return float(self.rtts[-1])
        idx = bisect_left(self.normalized_timestamps, query)
        t0, t1 = self.normalized_timestamps[idx-1], self.normalized_timestamps[idx]
        r0, r1 = self.rtts[idx-1], self.rtts[idx]
        ratio = (query - t0) / (t1 - t0)
        return float(max(0.0, r0 + ratio * (r1 - r0)))


class CloudEdgeSimulator:
    def __init__(self, profiling_data: ProfilingData,
                 trace_csv_path: Optional[str] = None,
                 timeout_threshold_ms: float = 150.0):
        self.profiling = profiling_data

        self.trace_tracker = None
        if trace_csv_path:
            try:
                df = pd.read_csv(trace_csv_path)
                df = df.dropna(subset=['timestamp', 'bandwidth_mbps', 'rtt_ms'])
                trace_data = list(zip(
                    df['timestamp'].astype(float),
                    df['bandwidth_mbps'].astype(float),
                    df['rtt_ms'].astype(float)
                ))
                if trace_data:
                    self.trace_tracker = NetworkTraceTracker(trace_data)
            except Exception as e:
                print(f"⚠️ Could not load trace: {e}")

        self.cumulative_time_seconds = 0.0
        self.episode_offset = 0.0
        self.cumulative_energy_joules = 0.0

        self.timeout_threshold_ms = timeout_threshold_ms
        self.optimistic_chain_start = -1
        self.data_location = 'EDGE'
        self.optimistic_chain_length = 0
        self.last_timeout_occurred = False
        self.last_action_mode = -1  # Track action for debugging

        self.edge_idle_power = getattr(profiling_data, 'edge_idle_power', 1.0)
        self.edge_comm_power = getattr(profiling_data, 'edge_communication_power', 2.0)

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

    def get_current_bandwidth(self) -> float:
        if self.trace_tracker:
            query = self.cumulative_time_seconds + self.episode_offset
            bw = self.trace_tracker.get_bandwidth_at_time(query, use_normalized=True)
            return float(max(0.5, bw))
        else:
            return random.uniform(2.0, 15.0)

    def get_current_rtt(self) -> float:
        if self.trace_tracker:
            query = self.cumulative_time_seconds + self.episode_offset
            return self.trace_tracker.get_rtt_at_time(query, use_normalized=True)
        else:
            return float(self.profiling.rtt)

    def get_possible_actions(self, layer):
        if layer >= len(self.profiling.layers):
            return []
        nodes = self.profiling.get_num_nodes(layer)
        if layer == len(self.profiling.layers) - 1:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer
            return [a]

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

    def get_next_state_cloud_waiting_time(self, next_layer, current_action, isAllCloud=False):
        layer = int(next_layer)
        cloud_nodes = np.where((current_action[:, 1] == 1) | (current_action[:, 1] == 2))[0]

        if not isAllCloud:
            n_competing = random.randint(0, self.profiling.numberOfEdgeDevice - 1)
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
        _, _, _, layer, _ = current_state[:5]
        layer = int(layer)

        bw = self.get_current_bandwidth()
        rtt = self.get_current_rtt()

        if layer + 1 < len(self.profiling.layers):
            next_layer = layer + 1
            terminal = False
        else:
            next_layer = layer
            terminal = True

        prev_action_pattern = tuple(int(x) for x in action[:, 1])

        next_state = (
            bw,
            rtt,
            new_cloud_pending,
            next_layer,
            prev_action_pattern,
            surplus,
            neg_count,
            self.optimistic_chain_length
        )
        return next_state, terminal

    def compute_energy_and_time(self, current_state, current_action, cloud_pending_ms):
        bandwidth, rtt_ms, _, layer, prev_action, _, _, _ = current_state
        layer = int(layer)

        action_values = current_action[:, 1]
        action_mode = int(np.max(action_values))
        self.last_action_mode = action_mode  # Track for debugging

        profiling = self.profiling
        deps = profiling.dependencies

        # ---------- Transmission times ----------
        tx_to_cloud = 0.0
        rx_from_cloud = 0.0
        transmission_times = []
        offloaded = (action_mode == 1 or action_mode == 2)

        # SAFETY: Ensure bandwidth is never too small
        safe_bw = max(bandwidth, 0.5)

        if prev_action is not None and layer > 0:
            prev_assignments = np.asarray(prev_action, dtype=int)
            curr_assignments = np.asarray(current_action[:, 1], dtype=int)

            for curr_node in range(len(curr_assignments)):
                parent_nodes = deps.get((layer, curr_node), [])
                for (p_layer, p_node) in parent_nodes:
                    if p_layer == layer - 1:
                        parent_loc = prev_assignments[p_node] if p_node < len(prev_assignments) else 0
                    else:
                        parent_loc = 0
                    curr_loc = curr_assignments[curr_node]

                    if parent_loc != curr_loc:
                        output_size = profiling.get_output_size(layer, curr_node)  # KB
                        # Convert KB to MB, divide by bandwidth (MBps)
                        data_mb = output_size / 1024.0
                        tx_time = max(
                            data_mb / safe_bw,
                            rtt_ms / 1000.0
                        )
                        
                        # DEBUG: Catch suspicious values
                        if tx_time > 10.0:
                            print(f"⚠️ WARNING: tx_time={tx_time:.2f}s | Layer {layer}, Node {curr_node}")
                            print(f"   output_size={output_size}KB, bandwidth={bandwidth:.2f}MBps")
                            print(f"   data_mb={data_mb:.2f}MB, data_mb/safe_bw={data_mb/safe_bw:.2f}s")
                        
                        transmission_times.append(tx_time)
                        if curr_loc in [1, 2]:
                            tx_to_cloud = max(tx_to_cloud, tx_time)
                        else:
                            rx_from_cloud = max(rx_from_cloud, tx_time)
        else:
            if offloaded:
                input_size = profiling.get_input_size()  # KB
                data_mb = input_size / 1024.0
                tx_time = max(
                    data_mb / safe_bw,
                    rtt_ms / 1000.0
                )
                if tx_time > 10.0:
                    print(f"⚠️ WARNING: First layer tx_time={tx_time:.2f}s | input_size={input_size}KB")
                transmission_times.append(tx_time)
                tx_to_cloud = tx_time

        # Conservative (2) => immediate send-back
        if action_mode == 2:
            output_sizes = [profiling.get_output_size(layer, i) for i in range(len(current_action))]
            max_out = max(output_sizes) if output_sizes else 0
            data_mb = max_out / 1024.0
            rx_time = max(
                data_mb / safe_bw,
                rtt_ms / 1000.0
            )
            if rx_time > 10.0:
                print(f"⚠️ WARNING: Conservative rx_time={rx_time:.2f}s | max_out={max_out}KB")
            transmission_times.append(rx_time)
            rx_from_cloud = max(rx_from_cloud, rx_time)

        max_transmission_time = max(transmission_times) if transmission_times else 0.0

        # ---------- Edge computation ----------
        edge_times = []
        edge_energy = []
        for i in range(len(current_action)):
            if current_action[i, 1] == 0:
                node_t_s = profiling.get_node_edge_time(layer, i) / 1000.0
                node_p = profiling.get_node_edge_power(layer, i)
                edge_times.append(node_t_s)
                edge_energy.append(node_p * node_t_s)

        if layer in [3, 5]:
            edge_total_time_s = max(edge_times) if edge_times else 0.0
            edge_energy_total = max(edge_energy) if edge_energy else 0.0
        else:
            edge_total_time_s = sum(edge_times)
            edge_energy_total = sum(edge_energy)

        # ---------- Cloud idle/waiting ----------
        actual_idle_time_s = 0.0
        if offloaded:
            cloud_pending_s = cloud_pending_ms / 1000.0
            actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s)

        # ---------- Timeout check ----------
        total_wait_s = tx_to_cloud + cloud_pending_ms/1000.0 + rx_from_cloud
        recompute_energy = 0.0
        recompute_time = 0.0
        self.last_timeout_occurred = False

        # Only check timeout if offloaded
        if offloaded and total_wait_s * 1000 > self.timeout_threshold_ms:
            self.last_timeout_occurred = True

            if action_mode == 2:   # Conservative: recompute ONLY current layer
                for i in range(len(current_action)):
                    if current_action[i, 1] != 0:
                        node_t_s = profiling.get_node_edge_time(layer, i) / 1000.0
                        node_p = profiling.get_node_edge_power(layer, i)
                        recompute_time += node_t_s
                        recompute_energy += node_p * node_t_s

            elif action_mode == 1:   # Optimistic: recompute from chain_start to current
                if self.optimistic_chain_start != -1:
                    # Only recompute from chain_start to current (not all layers)
                    for l in range(self.optimistic_chain_start, layer + 1):
                        nodes = profiling.get_num_nodes(l)
                        for i in range(nodes):
                            # Only recompute nodes that were offloaded (we don't know which, so all)
                            node_t_s = profiling.get_node_edge_time(l, i) / 1000.0
                            node_p = profiling.get_node_edge_power(l, i)
                            recompute_time += node_t_s
                            recompute_energy += node_p * node_t_s

        # ---------- Energy ----------
        idle_energy = self.edge_idle_power * actual_idle_time_s
        comm_energy = self.edge_comm_power * (tx_to_cloud + rx_from_cloud)
        total_energy = edge_energy_total + idle_energy + comm_energy + recompute_energy

        # ---------- Completion time ----------
        if self.last_timeout_occurred:
            completion_time_s = (self.timeout_threshold_ms / 1000.0) + recompute_time
        else:
            completion_time_s = edge_total_time_s + actual_idle_time_s + max_transmission_time

        # ================================================================
        # Chain reset ONLY on Action 0 (Edge)
        # ================================================================
        if action_mode == 0:   # EDGE -> RESET chain
            self.optimistic_chain_start = -1
            self.data_location = 'EDGE'
        elif action_mode == 1: # OPTIMISTIC -> EXTEND chain
            if self.optimistic_chain_start == -1:
                self.optimistic_chain_start = layer
            self.data_location = 'CLOUDLET'
        else:  # action_mode == 2 (CONSERVATIVE) -> DO NOT RESET
            self.data_location = 'EDGE'

        if self.optimistic_chain_start == -1:
            self.optimistic_chain_length = 0
        else:
            self.optimistic_chain_length = layer - self.optimistic_chain_start + 1

        # Cap chain length to prevent explosion
        if self.optimistic_chain_length > 7:
            self.optimistic_chain_length = 7

        self.cumulative_time_seconds += completion_time_s
        self.cumulative_energy_joules += total_energy

        return total_energy, completion_time_s

    def calculate_reward(self, layer, total_energy, completion_time_s,
                         previous_surplus=0.0, negative_surplus_count=0, isA2C=False):
        # Energy penalty (scaled to be meaningful)
        energy_penalty = total_energy * 50.0
        
        # Strong timeout penalty to discourage risky offloading
        timeout_penalty = 0.0
        if self.last_timeout_occurred:
            timeout_penalty = 30.0 * (1 + self.optimistic_chain_length)
        
        # Strong penalty for choosing Optimistic (Action 1) if it causes problems
        action_penalty = 0.0
        if self.last_action_mode == 1 and self.last_timeout_occurred:
            action_penalty = 50.0  # Extra penalty for risky action that failed

        reward = - (energy_penalty + timeout_penalty + action_penalty)
        
        if isA2C:
            reward *= 0.15

        # Surplus / Deadline tracking
        fractional_deadline_ms = (
            self.profiling.get_edge_time_for_layer(layer)
            / self.profiling.get_total_edge_time()
        ) * getattr(self.profiling, 'deadline', 500.0)
        completion_time_ms = completion_time_s * 1000.0
        effective_deadline_ms = fractional_deadline_ms + previous_surplus
        surplus_ms = effective_deadline_ms - completion_time_ms
        if completion_time_ms > effective_deadline_ms:
            negative_surplus_count += 1

        return reward, surplus_ms, negative_surplus_count, fractional_deadline_ms