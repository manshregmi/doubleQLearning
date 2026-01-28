from profiling.profile import ProfilingData


def get_profiling_data(deadline, edge_devices):
    layers = [
        [0],         # v1                level 0                    
        [0],        # v2                 level 1
        [0],        # v3                 level 2
        [0,1,2],        # v4+ v7 +v10        level 3
        [0,1,2],    # v5 v8 v11          level 4
        [0,1,2],         # v6+ v9 + v12      level 5    
        [0],         # v13               level 6
    ]
    numberOfEdgeDevice = edge_devices 
    # -------------------------------
    # ⏱️ Edge execution times (ms)
    # -------------------------------
    node_edge_times = {
        # level 0
        (0, 0): 25.8,  #v1

        # Level 1
        (1, 0): 300.6, #v2

        # Level 2 
        (2, 0): 5, #v3

        # Level 3 
        (3, 0): 15.8, (3,1):23 , (3,2) : 22.6,   # v4,v7,v10

        # level 4
        (4, 0): 18.5, (4, 1): 105.1, (4, 2): 88.2,  # v5,v8,v11

        # level 5
        (5,0): 0.4, (5,1): 0.4, (5,2): 0.4, #max of v6 , v9 , v12

        # level 6
        (6,0): 0.1, #v13

    }

    # -------------------------------
    # ☁️ Cloud execution times (ms)
    # -------------------------------
    node_cloud_times = {
       # level 0
        (0, 0): 8.3,  #v1

        # Level 1
        (1, 0): 13.7, #v2

        # Level 2 
        (2, 0): 2.1, #v3

        # Level 3 
        (3, 0): 4.6, (3,1): 3.5, (3,2): 2.6,  #  v4,v7,v10

        # level 4 
        (4, 0): 5.5, (4, 1): 7.0, (4, 2): 7.0,  # v5,v8,v11

        (5,0): 0.2,  (5,1): 0.1, (5,2): 0.1,  # v6 , v9 , v12

        (6,0): 0.1,     #v13
    }

    # -------------------------------
    # ⚡ Edge power consumption (W)
    # -------------------------------
    node_edge_powers = {
         # level 0
        (0, 0): 5.32,  #v1

        # Level 1
        (1, 0): 8.58, #v2

        # Level 2 
        (2, 0): 4.96, #v3

        # Level 3 
        (3, 0): 5.1, (3,1): 5.45, (3,2): 5.13,  # v4,v7,v10

        # level 4 (similar to node 4)
        (4, 0): 8.13, (4, 1): 8.26, (4, 2): 8.26,  # v5,v8,v11

        (5,0): 5.11, (5,1) : 5.07, (5,2): 5.05,   #v6 , v9 , v12

        (6,0): 0.1,     #v13
    }

    input_size = {
        (0,0): 8100,
        (1,0): 3072,
        (2,0): 1275,
        (3,0): 26.44, (3,1): 192, (3,2): 108, #  v4,v7,v10
        (4,0): 26.44, (4,1): 192, (4,2): 108,
        (5,0): 3.72, (5,1): 1, (5,2): 1, # v6 , v9 , v12
        (6,0): 0
    }

    dependencies = {
        (0,0): [],
        (1,0): [ (0,0) ],
        (2,0): [ (1,0) ],
        (3,0): [ (2,0) ],
        (3,1): [ (2,0) ],
        (3,2): [ (2,0) ],
        (4,0): [ (3,0) ],
        (4,1): [ (3,1) ],
        (4,2): [ (3,2) ],
        (5,0): [ (4,0) ],
        (5,1): [ (4,1) ],
        (5,2): [ (4,2) ],
        (6,0): [ (5,0), (5,1), (5,2)],
    }


    profiling_data = ProfilingData(
        numberOfEdgeDevice=numberOfEdgeDevice,
        layers=layers,
        node_edge_times=node_edge_times,
        node_cloud_times=node_cloud_times,
        bandwidth=8,
        rtt=4.5,
        input_size=input_size,
        node_edge_powers=node_edge_powers,
        edge_idle_power=4.24,
        deadline=deadline,
        edge_communication_power=5.94,
        dependencies=dependencies,
    )
    return profiling_data

