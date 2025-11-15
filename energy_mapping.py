import numpy as np
from itertools import product

from profiling.initialize_profiling import get_profiling_data
from simulator.simulator import CloudEdgeSimulator

def generate_assignments_with_metrics(simulator):
    """
    Generates all possible device assignments for every node in layers 0-5,
    layer 6 is always device 0.
    Returns list of [assignment_list, total_energy, total_time]
    """

    # Hardcoded layer sizes
    LAYER_SIZES = [1, 1, 1, 3, 3, 3, 1]  # layers 0-6
    NUM_LAYERS = len(LAYER_SIZES)
    results = []

    # Create per-layer device options
    layer_device_options = []
    for layer_idx in range(NUM_LAYERS - 1):  # layer 0-5
        num_nodes = LAYER_SIZES[layer_idx]
        # Each node can be 0 or 1 → product of devices per node in this layer
        node_options = list(product([0,1], repeat=num_nodes))
        layer_device_options.append(node_options)

    # Generate all combinations across layers 0-5
    for combo_layers in product(*layer_device_options):
        assignment_list = []

        # Build assignment arrays per layer 0-5
        for layer_idx, layer_nodes in enumerate(combo_layers):
            arr = np.array([[layer_idx, device] for device in layer_nodes])
            assignment_list.append(arr)

        # Layer 6 fixed
        assignment_list.append(np.array([[6, 0]] * LAYER_SIZES[6]))

        # Compute total energy and time
        total_energy = 0.0
        total_time = 0.0
        cloud_waiting_time = 0.0
        prev_assignment = None

        for index, assignment in enumerate(assignment_list):
            cloud_waiting_time = simulator.get_next_state_cloud_waiting_time(index, assignment)
            previous_assignment = None if index == 0 else assignment_list[index - 1]
            current_state = (
                simulator.profiling.bandwidth,
                cloud_waiting_time,
                index,
                previous_assignment,
                0.0,
                0
            )
            energy, time_ms = simulator.compute_energy_and_time(
                current_state,
                assignment,
                cloud_pending_ms=cloud_waiting_time
            )
            total_energy += energy
            total_time += time_ms

        # Append result
        results.append([
            assignment_list,
            total_energy,
            total_time
        ])

    return results




all_combinations = generate_assignments_with_metrics(simulator=CloudEdgeSimulator(get_profiling_data(500)))
print("Total combinations generated:", len(all_combinations))
# # Simulation parameters
# Sort all combinations by energy (x[1])
sorted_by_energy = sorted(all_combinations, key=lambda x: x[1])
lowest_energy = sorted_by_energy[0][1]
heighest_energy = sorted_by_energy[-1][1]

print(f"Lowest Energy: {lowest_energy}, Highest Energy: {heighest_energy}")

# # Get top 10 lowest-energy combinations
# top10 = sorted_by_energy[:10]

# print("Top 10 lowest energy combinations:")
# for i, item in enumerate(top10, 1):
#     assignment, energy, time_ms = item
#     print(f"{i}. Energy={energy}, Time={time_ms}")
#     print("   Assignment:", assignment)


# sorted_by_time = sorted(all_combinations, key=lambda x: x[2])
# top10_time = sorted_by_time[:10]

# print("Top 10 lowest time combinations:")
# for i, item in enumerate(top10_time, 1):
#     assignment, energy, time_ms = item
#     print(f"{i}. Energy={energy}, Time={time_ms}")
#     print("   Assignment:", assignment)

