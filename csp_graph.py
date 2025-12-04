# csp_final.py
import itertools
import numpy as np
import networkx as nx
from profiling.initialize_profiling import get_profiling_data
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator

def pattern_to_action_matrix(layer, pattern_str):
    """
    Convert pattern like 'CCC' or 'E' or 'CC' to simulator action.
    'C' -> 1 (cloud)
    'E' -> 0 (edge)
    
    Returns matrix of shape (nodes, 2):
    [ [layer, 0/1], ... ]
    """
    nodes = len(pattern_str)
    action = np.zeros((nodes, 2), dtype=int)
    action[:, 0] = layer

    for i, ch in enumerate(pattern_str):
        if ch.upper() == 'C':
            action[i, 1] = 1
        else:
            action[i, 1] = 0

    return action


# -----------------------------
# compute energy/time per-layer (uses child input size)
# -----------------------------
def compute_energy_and_time_layer(prof: ProfilingData,
                                  layer: int,
                                  current_assignments: np.ndarray,
                                  prev_assignments: np.ndarray,
                                  bandwidth,
                                  congestion=1
                                  ):
    """
    current_assignments: np.array of 0 (Edge) / 1 (Cloud)
    prev_assignments: same shape for previous layer or None for first layer
    bandwidth: same unit you use (seconds formula: (in_kB/1024)/bandwidth )
    Returns: (total_energy_J, completion_time_s, details_dict)
    """
    print(f"Computing energy/time for bandwidth {bandwidth} and congestion {congestion}")
    deps = prof.dependencies
    total_energy = 0.0

    # --- Transmission Times (seconds) using child input size ---
    transmission_times = []

    curr_arr = np.asarray(current_assignments, dtype=int)
    prev_arr = None
    if prev_assignments is not None:
        prev_arr = np.asarray(prev_assignments, dtype=int)

    if prev_arr is not None and layer > 0:
        # for each child, examine its parents and check location difference
        for curr_node in range(len(curr_arr)):
            parent_nodes = deps.get((layer, curr_node), [])
            for (p_layer, p_node) in parent_nodes:
                parent_loc = prev_arr[p_node] if p_layer == layer - 1 else 0
                curr_loc = curr_arr[curr_node]
                if parent_loc != curr_loc:
                    # use child's input size (KB)
                    in_kB = prof.input_size.get((layer, curr_node),
                                                None)
                    if in_kB is None:
                        in_kB = prof.input_size.get((0,0), 0)
                    tx_time_s = max((in_kB / 1024.0) / max(bandwidth, 1e-9),
                                    prof.rtt / 1000.0)
                    transmission_times.append(tx_time_s)
    else:
        # first layer: upload input if any node runs on cloud
        if np.any(curr_arr == 1):
            in_kB = prof.input_size.get((0, 0), None)
            if in_kB is None:
                # fallback to method if present or zero
                in_kB = prof.get_input_size() if hasattr(prof, "get_input_size") else 0
            tx_time_s = max((in_kB / 1024.0) / max(bandwidth, 1e-9),
                            prof.rtt / 1000.0)
            transmission_times.append(tx_time_s)

    max_transmission_time_s = max(transmission_times) if transmission_times else 0.0
    if max_transmission_time_s > 0:
        total_energy += prof.edge_communication_power * max_transmission_time_s

    # --- Edge Processing Energy/Time ---
    edge_times_s = []
    edge_energy_j = []
    for i in range(len(curr_arr)):
        if curr_arr[i] == 0:  # Edge
            p_w = prof.node_edge_powers[(layer, i)]
            t_s = prof.node_edge_times[(layer, i)] / 1000.0
            edge_times_s.append(t_s)
            edge_energy_j.append(p_w * t_s)

    # Layers 3 and 5 are parallel (per your original code)
    if layer in [3, 5]:
        edge_total_time_s = max(edge_times_s) if edge_times_s else 0.0
        total_energy += max(edge_energy_j) if edge_energy_j else 0.0
    else:
        edge_total_time_s = sum(edge_times_s)
        total_energy += sum(edge_energy_j)

    # --- Cloud pending & Idle Energy ---
    cloud_times_s = []
    for i in range(len(curr_arr)):
        if curr_arr[i] == 1:
            compute_s = prof.node_cloud_times[(layer, i)] / 1000.0
            cloud_times_s.append(compute_s)


    cloud_pending_s = max(cloud_times_s) if cloud_times_s else 0.0

    actual_idle_time_s = 0.0
    if cloud_pending_s > 0:
        cloud_pending_s += (congestion * (prof.get_max_layer_cloud_time(layer) / 1000.0))
        max_cloud_time_ms = prof.get_max_layer_cloud_time(layer)
        cloud_pending_s += ((congestion * max_cloud_time_ms) / 1000.0)
        actual_idle_time_s = max(0.0, cloud_pending_s - edge_total_time_s) 
        total_energy += (prof.edge_idle_power * actual_idle_time_s)

    # Completion time
    completion_time_s = edge_total_time_s + max_transmission_time_s + actual_idle_time_s

    details = {
        "max_tx_ms": max_transmission_time_s * 1000,
        "edge_time_ms": edge_total_time_s * 1000,
        "edge_energy_j": (sum(edge_energy_j) if (layer not in [3,5]) else (max(edge_energy_j) if edge_energy_j else 0.0)),
        "cloud_pending_ms": cloud_pending_s * 1000,
        "idle_ms": actual_idle_time_s * 1000,
        # "transmissions_count": len(transmission_times)
    }

    return total_energy, completion_time_s, details


# -----------------------------
# Helpers: convert state-name <-> assignment array
# -----------------------------
def state_name_to_assignment(name):
    """
    name examples: 'L0_C', 'L3_CCC', 'L6_E'
    returns numpy array of 0/1 for that layer
    """
    if "_" not in name:
        raise ValueError("Bad state name: " + str(name))
    parts = name.split("_", 1)
    pattern = parts[1]
    if pattern == "E":
        # all-edge; but we don't know k here — caller should use mapping list that stores array alongside name.
        # This function handles single-node patterns and multi-letter patterns only.
        return np.array([0])
    if pattern == "C":
        return np.array([1])
    # multi-letter like "ECC" or "CEC"
    return np.array([1 if c == "C" else 0 for c in pattern], dtype=int)


# -----------------------------
# Build CSP graph using compute_energy_and_time_layer
# -----------------------------
def build_csp_graph_from_profiling(prof: ProfilingData):
    """
    Builds graph G and returns (G, last_layer_state_list)
    last_layer_state_list is list of (name, assignment_array)
    """
    G = nx.DiGraph()
    layers = prof.layers
    num_layers = len(layers)
    last_layer_idx = num_layers - 1

    # Prepare layer states: list of (name, arr)
    layer_states = {}
    for layer_idx, nodes in enumerate(layers):
        k = len(nodes)
        states = []
        if layer_idx == last_layer_idx:
            # forced Edge: all zeros
            arr = np.zeros(k, dtype=int)
            name = f"L{layer_idx}_E"
            states.append((name, arr))
            G.add_node(name)
        else:
            for combo in itertools.product([0, 1], repeat=k):
                arr = np.array(combo, dtype=int)
                name = f"L{layer_idx}_" + "".join("C" if v == 1 else "E" for v in combo)
                states.append((name, arr))
                G.add_node(name)
        layer_states[layer_idx] = states

    # add source and compute edges from source to first-layer states (compute cost/bound here)
    G.add_node("s")


    congesstion = (prof.numberOfEdgeDevice - 1) * np.random.uniform(0.15,0.5)
    bandwidth = prof.bandwidth
    bandwidth_changes = []
    for _ in range(len(layers)):
        bandwidth = prof.bandwidth + (np.random.uniform(-0.5,0.5))
        bandwidth_changes.append(bandwidth)

    bandwidth = np.mean(bandwidth_changes)
    
    # print(f"Using bandwidth: {bandwidth} and congestion factor: {congesstion}")
    for name, arr in layer_states[0]:
        energy_j, time_s, details = compute_energy_and_time_layer(prof, 0, arr, None, bandwidth, congesstion)
        G.add_edge("s", name, cost=energy_j, bound=time_s * 1000.0, details=details)

    # connect subsequent layers
    for layer in range(1, num_layers):
        prev_states = layer_states[layer - 1]
        curr_states = layer_states[layer]
        for prev_name, prev_arr in prev_states:
            for curr_name, curr_arr in curr_states:
                energy_j, time_s, details = compute_energy_and_time_layer(prof, layer, curr_arr, prev_arr, bandwidth, congesstion)
                G.add_edge(prev_name, curr_name, cost=energy_j, bound=time_s * 1000.0, details=details)

    return G, layer_states[last_layer_idx]


# -----------------------------
# Lagrangian-relaxation CSP solver
# -----------------------------
def constrained_shortest_path(G: nx.DiGraph, last_layer_states, B_ms: float, verbose=True):
    """
    G edges must have 'cost' (J) and 'bound' (ms).
    last_layer_states: list of (name, arr) for terminal layer (we only need names)
    """
    terminal_names = [s[0] if isinstance(s, tuple) else s for s in last_layer_states]

    def path_sum(path, key):
        return sum(G[path[i]][path[i+1]][key] for i in range(len(path)-1))

    # initial p_c (min-cost) and p_b (min-bound)
    p_c = None
    best_cost = float("inf")
    for t in terminal_names:
        try:
            p = nx.dijkstra_path(G, "s", t, weight="cost")
            c = path_sum(p, "cost")
            if c < best_cost:
                best_cost = c; p_c = p
        except nx.NetworkXNoPath:
            continue
    if p_c is None:
        if verbose: print("No cost path to any terminal"); return None
    if path_sum(p_c, "bound") <= B_ms:
        if verbose: print("Cost-optimal path already feasible"); return p_c

    p_b = None
    best_bound = float("inf")
    for t in terminal_names:
        try:
            p = nx.dijkstra_path(G, "s", t, weight="bound")
            b = path_sum(p, "bound")
            if b < best_bound:
                best_bound = b; p_b = p
        except nx.NetworkXNoPath:
            continue
    if p_b is None:
        if verbose: print("No bound path to any terminal"); return None
    if path_sum(p_b, "bound") > B_ms:
        if verbose: print("Bound-optimal exceeds deadline -> no feasible"); return None

    # Lagrangian iteration
    lambda_sub = 0.0
    iteration = 0
    while True:
        iteration += 1
        cost_c = path_sum(p_c, "cost"); bound_c = path_sum(p_c, "bound")
        cost_b = path_sum(p_b, "cost"); bound_b = path_sum(p_b, "bound")

        denom = (bound_b - bound_c)
        if abs(denom) > 1e-12:
            la = (cost_c - cost_b) / denom
        else:
            # tiny subgradient step
            lambda_sub += 1e-6
            la = lambda_sub

        # set weights
        for u, v, d in G.edges(data=True):
            d["w"] = d["cost"] + la * d["bound"]

        # find r = argmin_w among terminals
        r = None; best_w = float("inf")
        for t in terminal_names:
            try:
                p = nx.dijkstra_path(G, "s", t, weight="w")
                w = sum(G[p[i]][p[i+1]]["w"] for i in range(len(p)-1))
                if w < best_w:
                    best_w = w; r = p
            except nx.NetworkXNoPath:
                continue

        if r is None:
            if verbose: print("No w-path found"); return None

        w_r = sum(G[r[i]][r[i+1]]["w"] for i in range(len(r)-1))
        w_c = sum(G[p_c[i]][p_c[i+1]]["w"] for i in range(len(p_c)-1))

        if verbose and (iteration % 50 == 0):
            print(f"iter {iteration}: la={la:.6g} w_r={w_r:.6g} w_c={w_c:.6g} cost_r={path_sum(r,'cost'):.6g} bound_r={path_sum(r,'bound'):.6g}")

        # convergence
        if abs(w_r - w_c) <= 1e-9:
            if verbose: print("Converged; returning feasible p_b"); return p_b

        # update
        if path_sum(r, "bound") <= B_ms:
            p_b = r
        else:
            p_c = r


# -----------------------------
# Helpers to pretty-print the chosen path assignments
# -----------------------------
def decode_assignments_from_path(path):
    assigns = []
    for node in path:
        if node == "s": continue
        if node.startswith("L"):
            parts = node.split("_", 1)
            if len(parts) > 1:
                assigns.append(parts[1])
    return assigns


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    prof = get_profiling_data(700)   # choose deadline in ms
    print("Profiling data initialized.")

    sim = CloudEdgeSimulator(prof)

    state = (prof.bandwidth,       # initial bandwidth
         0.0,                  # cloud pending
         0,                    # starting layer
         None,                 # prev_action
         0.0,                  # surplus
         0)                    # negative_surplus_count


    print("Building CSP graph...")
    G, last_layer_states = build_csp_graph_from_profiling(prof)
    print("Graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
    print("Terminals (last layer):", [s[0] for s in last_layer_states])

    print("Solving CSP...")
    solution = constrained_shortest_path(G, last_layer_states, prof.deadline, verbose=True)

    if solution is None:
        print("\nNO FEASIBLE SOLUTION FOUND (bound-optimal > deadline)")
    else:
        print("\n=== SOLUTION PATH ===")
        for n in solution:
            print(n)
        total_energy = sum(G[solution[i]][solution[i+1]]["cost"] for i in range(len(solution)-1))
        total_bound = sum(G[solution[i]][solution[i+1]]["bound"] for i in range(len(solution)-1))
        print(f"\nTotal Energy (J): {total_energy:.6f}")
        print(f"Total Time (ms): {total_bound:.3f}")

        assigns = decode_assignments_from_path(solution)
        print("Per-layer assignments (layer0 -> ... -> last):", assigns)
        print("\nPer-layer details:")
        for i in range(len(solution)-1):
            u, v = solution[i], solution[i+1]
            d = G[u][v].get("details", {})
            print(f"{u} -> {v}: cost={G[u][v]['cost']:.3f} J, bound={G[u][v]['bound']:.3f} ms, details={d}")

        total_energy_sim = 0.0
        total_time_sim = 0.0

        prev_action = None
        surplus = 0.0
        neg_count = 0
        cloud_pending = 0.0

        for layer_idx, pat in enumerate(assigns):

            # convert CSP pattern → action matrix
            action = pattern_to_action_matrix(layer_idx, pat)
            # print(f"\n[SIMULATION] Layer {layer_idx}, Pattern: {pat}, Action:\n{action}")

            # compute waiting time
            cloud_pending = sim.get_next_state_cloud_waiting_time(
                layer_idx,
                action,
            )

            # compute energy and time
            E, T = sim.compute_energy_and_time(
                (state[0], state[1], layer_idx, prev_action, surplus, neg_count),
                action,
                cloud_pending
            )

            # reward (if needed)
            reward, surplus, neg_count, frac = sim.calculate_reward(
                layer_idx, E, T, surplus, neg_count
            )
            # print(f"[SIMULATION] Layer {layer_idx}, Energy: {E:.6f} J, Time: {T*1000.0:.3f} ms")

            total_energy_sim += E
            total_time_sim += T

            # prepare next state
            state, terminal, cloud_pending = sim.get_next_state(
                (state[0], state[1], layer_idx, prev_action, surplus, neg_count),
                action,
                surplus,
                neg_count,
                cloud_pending
            )

            prev_action = action.copy()

            if terminal:
                break
        print(f"\n[SIMULATION CHECK] Total Energy (J): {total_energy_sim:.6f}")
        print(f"[SIMULATION CHECK] Total Time (ms): {total_time_sim:.3f}")


            