from profiling.profile import ProfilingData


def get_profiling_data(deadline):
    layers = [
        [0], [0, 1, 2], [0, 1], [0, 1, 2, 3], [0],
    ]
    numberOfEdgeDevice = 2

    node_edge_times = {
        (0, 0): 1,
        (1, 0): 35, (1, 1): 28, (1, 2): 22,
        (2, 0): 45, (2, 1): 40,
        (3, 0): 30, (3, 1): 25, (3, 2): 38, (3, 3): 42,
        (4, 0): 1
    }
    node_cloud_times = {
        (0, 0): 0,
        (1, 0): 16, (1, 1): 12, (1, 2): 10,
        (2, 0): 20, (2, 1): 24,
        (3, 0): 12, (3, 1): 10, (3, 2): 14, (3, 3): 18,
        (4, 0): 0
    }
    node_edge_powers = {
        (0, 0): 0.5,
        (1, 0): 12.132, (1, 1): 11.305, (1, 2): 10.596,
        (2, 0): 13.304, (2, 1): 12.717,
        (3, 0): 11.542, (3, 1): 10.923, (3, 2): 12.553, (3, 3): 13.076,
        (4, 0): 0.5
    }

    profiling_data = ProfilingData(
        numberOfEdgeDevice=numberOfEdgeDevice,
        layers=layers,
        node_edge_times=node_edge_times,
        node_cloud_times=node_cloud_times,
        bandwidth=5.0,
        rtt=10.0,
        output_size=5,
        node_edge_powers=node_edge_powers,
        edge_idle_power=4.0,
        deadline=deadline,
        edge_communication_power=5.0,
    )
    return profiling_data

