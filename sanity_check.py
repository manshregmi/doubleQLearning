import numpy as np
from profiling.initialize_profiling import get_profiling_data
from simulator.simulator import CloudEdgeSimulator


profiling_data = get_profiling_data(500)
simulator = CloudEdgeSimulator(profiling_data=profiling_data)

bandwidth = profiling_data.bandwidth  # Mbps

    # Example state and action
current_state = (bandwidth, 0, 0, None, 0.0, 0)  # bandwidth, cloud_time, layer, prev_action, surplus, neg_count
actions = [
        [[0,0]],
        [[1,1]],
        [[2,1]],
        [[3,0],[3,1],[3,1]],
        [[4,1],[4,1],[4,1]],
        [[5,1],[5,1],[5,1]],
        [[6,0]]
    ]
energy_t = 0.0
time_ms_t = 0.0
for i in range(len(actions)):
    action = np.array(actions[i])  # convert to numpy array
    prev_action = None if i == 0 else np.array(actions[i-1])
    cloud_processing_time = simulator.get_next_state_cloud_waiting_time(i, action)
    current_state = (bandwidth , cloud_processing_time, i, prev_action, 0.0, 0)
    energy, time_ms = simulator.compute_energy_and_time(current_state, action, cloud_pending_ms=cloud_processing_time)
    energy_t += energy
    time_ms_t += time_ms



print("Energy (Joules):", energy_t)
print("Time (ms):", time_ms_t)    