from profiling.profile import ProfilingData


def get_profiling_data(deadline):
    layers = [
        [0],                                             # layer 0
        [0, 1, 2, 3, 4, 5],                             # layer 1
        [0, 1, 2, 3, 4, 5, 6],                          # layer 2
        [0, 1, 2, 3, 4, 5],                              # layer 3
        [0],                                             # layer 4
    ]

    numberOfEdgeDevice = 3  
    # -------------------------------
    # ⏱️ Edge execution times (ms)
    # -------------------------------
    node_edge_times = {
        # Node 0
        (0, 0): 1,

        # Node 1 (linearly scaled 0–5)
        (1, 0): 35, (1, 1): 28, (1, 2): 22, (1, 3): 16, (1, 4): 10, (1, 5): 4,

        # Node 2 (linearly scaled 0–6)
        (2, 0): 45, (2, 1): 40, (2, 2): 35, (2, 3): 30, (2, 4): 25, (2, 5): 20, (2, 6): 15,

        # Node 3 (linearly scaled 0–13)
        (3, 0): 30, (3, 1): 25, (3, 2): 38, (3, 3): 42, (3, 4): 46, (3, 5): 50,

        # Node 4 (similar to node 4)
        (4, 0): 1,
    }

    # -------------------------------
    # ☁️ Cloud execution times (ms)
    # -------------------------------
    node_cloud_times = {
        # Node 0
        (0, 0): 0,

        # Node 1 (linearly scaled)
        (1, 0): 16, (1, 1): 12, (1, 2): 10, (1, 3): 8, (1, 4): 6, (1, 5): 4,

        # Node 2
        (2, 0): 20, (2, 1): 24, (2, 2): 28, (2, 3): 32, (2, 4): 36, (2, 5): 40, (2, 6): 44,

        # Node 3
        (3, 0): 12, (3, 1): 10, (3, 2): 14, (3, 3): 18, (3, 4): 22, (3, 5): 26,

        # Node 5 (same)
        (4, 0): 0,
    }

    # -------------------------------
    # ⚡ Edge power consumption (W)
    # -------------------------------
    node_edge_powers = {
        # Node 0
        (0, 0): 0.5,

        # Node 1
        (1, 0): 12.132, (1, 1): 11.305, (1, 2): 10.596,
        (1, 3): 9.887, (1, 4): 9.178, (1, 5): 8.469,

        # Node 2
        (2, 0): 13.304, (2, 1): 12.717, (2, 2): 12.130,
        (2, 3): 11.543, (2, 4): 10.956, (2, 5): 10.369, (2, 6): 9.782,

        # Node 3
        (3, 0): 11.542, (3, 1): 10.923, (3, 2): 12.553, (3, 3): 13.076,
        (3, 4): 13.599, (3, 5): 14.122, 

        # Node 3
        (5, 0): 0.5,
    }

    profiling_data = ProfilingData(
        numberOfEdgeDevice=numberOfEdgeDevice,
        layers=layers,
        node_edge_times=node_edge_times,
        node_cloud_times=node_cloud_times,
        bandwidth=12,
        rtt=10.0,
        output_size=5,
        node_edge_powers=node_edge_powers,
        edge_idle_power=2.0,
        deadline=deadline,
        edge_communication_power=5.0,
    )
    return profiling_data

