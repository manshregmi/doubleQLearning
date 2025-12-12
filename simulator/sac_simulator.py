# simulator/sac_simulator.py
import time
import numpy as np
import torch
import random
from a2c.sac_agent import SACAgent
from simulator.simulator import CloudEdgeSimulator
from profiling.profile import ProfilingData

# ----------------------------
# State Flattening
# ----------------------------
def flatten_state(state, max_layers, max_nodes):
    """
    Flatten tuple state into 6-element vector for SAC network.
    Applies A2C-style discretization bins to bandwidth, cloud_time, and surplus
    before normalization.

    State: (bandwidth, cloud_time, layer, prev_action, surplus, negative_surplus_count)
    """

    # --- Discretization bins (same as A2C) ---
    bandwidth_bins = np.linspace(1, 15, 6)      # Mbps
    cloudtime_bins = np.linspace(0, 100, 20)    # ms
    surplus_bins = np.linspace(-5, 5, 21)       # s

    def discretize(value, bins):
        idx = np.digitize(value, bins) - 1
        idx = np.clip(idx, 0, len(bins) - 1)
        return float(bins[idx])

    # --- Apply discretization ---
    bw_disc = discretize(state[0], bandwidth_bins)
    ct_disc = discretize(state[1], cloudtime_bins)
    surplus_disc = discretize(state[4], surplus_bins)

    # --- Normalize for network input ---
    bw_norm = bw_disc / 30.0
    cloud_time_norm = ct_disc / 100.0   # cap 0–100 ms
    layer_norm = state[2] / max_layers
    surplus_norm = (surplus_disc + 5) / 10.0  # map [-5,5] → [0,1]
    neg_count_norm = state[5] / 20.0

    # --- Previous action summary ---
    prev_action = state[3]
    prev_action_scalar = 0.0
    if prev_action is not None and prev_action.size > 0:
        nodes_on_cloud = np.sum(prev_action[:, 1] == 1)
        num_nodes = prev_action.shape[0]
        prev_action_scalar = nodes_on_cloud / num_nodes if num_nodes > 0 else 0.0

    flat = np.array([
        bw_norm,
        cloud_time_norm,
        layer_norm,
        surplus_norm,
        neg_count_norm,
        prev_action_scalar
    ], dtype=np.float32)

    return flat

# ----------------------------
# Flatten actions to fixed size
# ----------------------------
def flatten_action(action_matrix, max_nodes):
    """
    Converts an action matrix (num_nodes, 2) into a fixed-size vector.
    Pads with zeros if num_nodes < max_nodes.
    Only uses column 1 (assignment: 0=edge, 1=cloud).
    """
    num_nodes = action_matrix.shape[0]
    flat = action_matrix[:, 1]  # only assignment column
    if num_nodes < max_nodes:
        flat = np.pad(flat, (0, max_nodes - num_nodes), 'constant', constant_values=0)
    return flat.astype(np.float32)

# ----------------------------
# Map SAC continuous output to discrete action
# ----------------------------
def map_continuous_action_to_discrete(cont_action, layer_idx, simulator):
    all_actions = simulator.get_possible_actions(layer_idx)
    if not all_actions:
        return np.array([])

    # Handle scalar or array
    if isinstance(cont_action, (np.ndarray, list)):
        cont_val = float(np.clip(cont_action[0], -1, 1))
    else:
        cont_val = float(np.clip(cont_action, -1, 1))

    # Map [-1,1] -> [0,1] -> index
    normalized = (cont_val + 1) / 2
    idx = int(np.clip(normalized * len(all_actions), 0, len(all_actions) - 1))
    return all_actions[idx]

# ----------------------------
# SAC Simulation Runner
# ----------------------------
def run_sac_simulation(profiling_data: ProfilingData, episodes=1000, max_steps=20):
    simulator = CloudEdgeSimulator(profiling_data)
    max_layers = len(profiling_data.layers)
    max_nodes = profiling_data.get_max_nodes()
    bandwidth = profiling_data.bandwidth
    cloud_time = 0.0

    state_dim = 6
    action_dim = max_nodes  # fixed-size flattened action vector
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, is_test=False)

    checkpoint_file = 'sac_checkpoint.pth'
    try:
        agent.load_checkpoint(checkpoint_file)
    except Exception as e:
        print(f"Checkpoint load failed: {e}, starting fresh.")

    edge_energy_log = []
    completion_time_log = []
    computation_time_log = []

    for ep in range(episodes):
        total_reward = 0.0
        total_energy = 0.0
        total_time = 0.0

        current_state = (bandwidth, cloud_time, 0, None, 0.0, 0)
        current_surplus = 0.0
        current_neg_count = 0
        actions_taken = []
        comp_time = 0.0

        for step in range(max_steps):
            flat_state = flatten_state(current_state, max_layers, max_nodes)
            # --- SAC selects action (returns discrete matrix) ---

            start_time = time.time()
            discrete_action = agent.select_action(flat_state, layer=current_state[2], simulator=simulator, epsilon=0.2)
            end_time = time.time()

            comp_time += (end_time - start_time)
            actions_taken.append(discrete_action.copy())

            if discrete_action.size == 0:
                terminal = True
                next_state = current_state
                reward, energy, completion_time_s = 0.0, 0.0, 0.0
            else:
                # next_state, terminal, cloud_pending = simulator.get_next_state(
                #     current_state, discrete_action, current_surplus, current_neg_count
                # )
                # energy, completion_s = simulator.compute_energy_and_time(current_state, discrete_action, cloud_pending)
                # reward, current_surplus, current_neg_count, _ = simulator.calculate_reward(
                #     current_state[2], energy, completion_s, current_surplus, current_neg_count, isA2C=True
                # )
                
                next_state_cloud_processing = simulator.get_next_state_cloud_waiting_time(
                    next_layer = (int(current_state[2])) if ((int(current_state[2]) + 1)  < len(simulator.profiling.layers)) else int(current_state[2]),
                    current_action=discrete_action
                )

                # Simulator step(s)
                energy, completion_time_s = simulator.compute_energy_and_time(
                    current_state=current_state, current_action=discrete_action, cloud_pending_ms=next_state_cloud_processing
                )

                # Reward computation (simulator returns scaled reward)
                reward, surplus, negative_surplus_count, fractional_deadline = simulator.calculate_reward(
                    int(current_state[2]), energy, completion_time_s, current_state[4], current_state[5], isA2C=True
                )
                # surplus /= 1000.0  # convert to seconds

                # Next state from simulator
                next_state, terminal, _ = simulator.get_next_state(
                    current_state, discrete_action, surplus, negative_surplus_count, new_cloud_pending=next_state_cloud_processing
                )

        
                # Flatten action before storing in buffer
                flat_action = flatten_action(discrete_action, max_nodes)
                flat_next = flatten_state(next_state, max_layers, max_nodes)
                agent.replay_buffer.push(flat_state, flat_action, reward, flat_next, terminal)

            # Train SAC
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update_parameters()

            total_reward += reward
            total_energy += energy
            total_time += completion_time_s * 1000  # s -> ms
            current_state = next_state

            if terminal:
                # bandwidth = current_state[0]
                # cloud_time = current_state[1]
                break

        edge_energy_log.append(total_energy)
        completion_time_log.append(total_time)
        computation_time_log.append(comp_time)



        E = np.array(edge_energy_log)
        T = np.array(completion_time_log)
        comp_times = np.array(computation_time_log)

        # --- Print nicely per episode ---
        if (ep + 1) % 1000 == 0:
            print(f"Episode {ep}, Energy: {total_energy:.3f}, Time: {total_time:.3f}, Reward: {total_reward:.3f}")
        # Save checkpoint
    try:
        agent.save_checkpoint(checkpoint_file)
        # print(f"[SAC] ✅ Checkpoint saved -> {checkpoint_file}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")


    E = np.array(edge_energy_log)
    T = np.array(completion_time_log)
    print("----- SAC Simulation Results -----")
    print(f"SAC Avg Energy: {E.mean():.3f} J, Std: {E.std():.3f}, Lower Bound: {E.min():.3f} J, Upper Bound: {E.max():.3f} J , Lowest index: {np.argmin(E)} time at that index: {T[np.argmin(E)]:.3f} ms")
    print(f"SAC Avg Time: {T.mean():.3f} ms, Std: {T.std():.3f}, Lower Bound: {T.min():.3f} ms, Upper Bound: {T.max():.3f} ms, Lowest index: {np.argmin(T)} energy at that index: {E[np.argmin(T)]:.3f} J")      
    print(f"SAC Avg Computation Time per Episode: {comp_times.mean():.10f} s, Std: {comp_times.std():.10f} s Total Computation Time: {comp_times.sum():.4f} s Min: {comp_times.min():.10f} s, Max: {comp_times.max():.10f} s Max Episode Index: {comp_times.max():.10f} s")
    return np.mean(E), np.mean(T)
