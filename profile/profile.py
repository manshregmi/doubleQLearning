class ProfilingData:
    def __init__(
        self,
        numberOfEdgeDevice,
        layers,  # List of lists with node sizes per layer
        node_edge_times,  # Dict {(layer_idx, node_idx): comp_time on edge}
        node_cloud_times, # Dict {(layer_idx, node_idx): comp_time on cloud}
        bandwidth,
        rtt,
        output_size,
        node_edge_powers,  # Dict {(layer_idx, node_idx): power on edge}
        edge_idle_power,
        deadline
    ):
        self.numberOfEdgeDevice = numberOfEdgeDevice
        self.layers = layers
        self.node_edge_times = node_edge_times
        self.node_cloud_times = node_cloud_times
        self.bandwidth = bandwidth
        self.rtt = rtt
        self.output_size = output_size
        self.node_edge_powers = node_edge_powers
        self.edge_idle_power = edge_idle_power
        self.deadline = deadline

    def get_num_nodes(self, layer_idx):
        return len(self.layers[layer_idx])

    def get_node_edge_time(self, layer_idx, node_idx):
        return self.node_edge_times.get((layer_idx, node_idx), 0.0)

    def get_node_cloud_time(self, layer_idx, node_idx):
        return self.node_cloud_times.get((layer_idx, node_idx), 0.0)

    def get_node_edge_power(self, layer_idx, node_idx):
        return self.node_edge_powers.get((layer_idx, node_idx), 0.0)

    def get_total_nodes(self):
        total_nodes = sum(len(layer) for layer in self.layers)
        return total_nodes
