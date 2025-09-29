import random
import numpy as np
import matplotlib.pyplot as plt
from simulator.simulator import CloudEdgeSimulator
from profiling.profile import ProfilingData


def get_random_action(profiling_data: ProfilingData, layer_idx: int):
    """
    Generate a random action for a single layer in the format:
    [[layer_idx, decision], ...]
    """
    num_nodes = profiling_data.get_num_nodes(layer_idx)

    a = np.zeros((num_nodes, 2), dtype=int)
    a[:, 0] = layer_idx

    if layer_idx == 0 or layer_idx == (len(profiling_data.layers) - 1):
        a[:, 1] = 0  # first & last layer forced to edge
    else:
        for node in range(num_nodes):
            a[node, 1] = random.choice([0, 1])

    return a
def get_all_edge_action(profilingData: ProfilingData, layer_idx: int):
    """All nodes on edge for a single layer."""
    num_nodes = profilingData.get_num_nodes(layer_idx)
    a = np.zeros((num_nodes, 2), dtype=int)
    a[:, 0] = layer_idx
    a[:, 1] = 0  # all edge
    return a


def get_all_cloud_action(profilingData: ProfilingData, layer_idx: int):
    """All nodes on cloud for a single layer (except first/last, forced to edge)."""
    num_nodes = profilingData.get_num_nodes(layer_idx)
    a = np.zeros((num_nodes, 2), dtype=int)
    a[:, 0] = layer_idx
    if layer_idx == 0 or layer_idx == (len(profilingData.layers) - 1):
        a[:, 1] = 0  # input/output must be edge
    else:
        a[:, 1] = 1  # all cloud
    return a

def run_random_scheduler(profiling_data: ProfilingData, episodes=10, max_steps=20, is_random=True, is_all_cloud=False):
    """
    Run random offloading scheduler benchmark over multiple episodes.
    Collect per-episode reward, energy, and completion time.
    """
    episode_energies = []
    episode_completion_times = []

    for ep in range(episodes):
        energies = []
        times = []

        simulator = CloudEdgeSimulator(profiling_data)
        initial_bandwidth = 15.0
        initial_cloud_time = 0.0
        initial_layer = 0
        prev_action = None
        state = (initial_bandwidth, initial_cloud_time, initial_layer, prev_action, 0, 0)  # (bandwidth, cloud_time, layer, prev_action, surplus, negative_surplus_count)

        for step in range(max_steps):
            action = get_random_action(profiling_data, state[2]) if is_random else get_all_cloud_action(profiling_data, state[2]) if is_all_cloud else get_all_edge_action(profiling_data, state[2])   
            next_state, terminal, cloud_time = simulator.get_next_state(state, action, 0, state[5])
            total_energy, completion_time = simulator.compute_energy_and_time(state, action, cloud_time)

        #             energy, completion_time = self.simulator.compute_energy_and_time(current_state=current_state, current_action=action, cloud_pending_ms= current_state[1])
        # reward, surplus, negative_surplus_count = self.simulator.calculate_reward(int(current_state[2]), energy, completion_time, current_state[4], current_state[5])
        # next_state, terminal, _ = self.simulator.get_next_state(current_state, action, surplus, negative_surplus_count)

            energies.append(total_energy)
            times.append((completion_time * 1000))  # convert to ms

            state = next_state
            if terminal:
                initial_bandwidth = next_state[0]
                break

        # record episode-level results
        episode_energies.append(np.sum(energies))
        episode_completion_times.append(np.sum(times))

    # --- Plotting ---
    # plt.figure(figsize=(6,4))
    # plt.plot(np.arange(1, episodes+1, 1), episode_energies, marker='o')
    # plt.xlabel("Episodes")
    # plt.ylabel("Total Edge Energy (Joules)")
    # plt.title(f"{'Random' if is_random else 'All Cloud' if is_all_cloud else 'All Edge'} Scheduler: Energy vs Episodes, {'average energy=' + str(np.mean(episode_energies)) if episodes > 0 else ''}")
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(6,4))
    # plt.plot(np.arange(1, episodes+1, 1), episode_completion_times, marker='s')
    # plt.xlabel("Episodes")
    # plt.ylabel("Avg Completion Time (ms)")
    # plt.title(f"{'Random' if is_random else 'All Cloud' if is_all_cloud else 'All Edge'}: Completion Time vs Episodes, {'average time=' + str(np.mean(episode_completion_times)) if episodes > 0 else ''}")
    # plt.grid(True)
    # plt.show()

    return np.mean(episode_energies), np.mean(episode_completion_times)
