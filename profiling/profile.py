# profiling/profile.py
import numpy as np


class ProfilingData:
    """
    Interface used by the simulator and agent.

    Methods the simulator depends on:
        get_num_nodes(layer)               -> int
        get_node_edge_time(layer, node)    -> ms
        get_node_cloud_time(layer, node)   -> ms
        get_node_edge_power(layer, node)   -> W
        get_input_size(layer, node)        -> KB   (parent's output)
        get_output_size(layer, node)       -> KB   (this node's output)
        get_max_layer_cloud_time(layer)    -> ms
        get_edge_time_for_layer(layer)     -> ms   (sum over nodes)
        get_total_edge_time()              -> ms   (sum over layers)
        get_layer_input_size(layer)        -> KB   (model input for layer 0)
    """

    def __init__(self, numberOfEdgeDevice, layers,
                 node_edge_times, node_cloud_times, bandwidth, rtt,
                 input_size, node_edge_powers, edge_idle_power,
                 deadline, edge_communication_power, dependencies):
        self.numberOfEdgeDevice = numberOfEdgeDevice
        self.layers = layers
        self.node_edge_times = node_edge_times
        self.node_cloud_times = node_cloud_times
        self.bandwidth = bandwidth
        self.rtt = rtt
        self.input_size = input_size
        self.node_edge_powers = node_edge_powers
        self.edge_idle_power = edge_idle_power
        self.deadline = deadline
        self.edge_communication_power = edge_communication_power
        self.dependencies = dependencies

    # ---- shape ----
    def get_num_nodes(self, layer):
        return len(self.layers[layer])

    def get_num_layers(self):
        return len(self.layers)

    # ---- compute ----
    def get_node_edge_time(self, layer, node):
        return self.node_edge_times[(layer, node)]

    def get_node_cloud_time(self, layer, node):
        return self.node_cloud_times[(layer, node)]

    def get_node_edge_power(self, layer, node):
        return self.node_edge_powers[(layer, node)]

    def get_max_layer_cloud_time(self, layer):
        return max(self.node_cloud_times[(layer, n)]
                   for n in range(self.get_num_nodes(layer)))

    # ---- data sizes (KB) ----
    def get_input_size(self, layer, node):
        return self.input_size[(layer, node)]

    def get_output_size(self, layer, node):
        """Size (KB) of (layer, node)'s output tensor.

        Semantics: output of (l, n) == input of its child.  Since the DFG may
        converge (multiple nodes sharing one child, or one node feeding many),
        we look at the children in `dependencies` and take the *input_size* of
        the first child.  If a node has no children (true sink), output = 0.
        """
        children = [
            (cl, cn)
            for (cl, cn), parents in self.dependencies.items()
            if (layer, node) in parents
        ]
        if not children:
            return 0.0
        # Input size of the child == output size of this node
        cl, cn = children[0]
        return self.input_size[(cl, cn)]

    def get_layer_input_size(self, layer):
        return self.input_size[(0, 0)]

    # ---- deadline helpers ----
    def get_edge_time_for_layer(self, layer):
        """Sum of node edge times for the layer (paper Eq. 6)."""
        return sum(self.node_edge_times[(layer, n)]
                   for n in range(self.get_num_nodes(layer)))

    def get_total_edge_time(self):
        return sum(self.get_edge_time_for_layer(l)
                   for l in range(self.get_num_layers()))