#!/usr/bin/env python3
"""
MEC_state_sim.py

Purpose:
- Run MEC-paper-style experiments (state = (tc, ac)) on your profiling data.
- Simulate cloud waiting time using your approach: queue_ms = max_cloud_time * K * U(0,1)
- Produce average total time, average total energy, and deadline-satisfaction probability
  for each method: DP-oracle, Greedy, Threshold, Tabular Q-learning (MEC-state).

Usage:
    python MEC_state_sim.py --deadline 500 --N 8 --trials 4000 --episodes 4000

Edit the import to point to your get_profiling_data function.
"""

import math, random, argparse, os
from collections import defaultdict
import numpy as np

# ----------------- EDIT THIS import to your module -----------------
# Example:
# from profiling.myprofilemodule import get_profiling_data, ProfilingData
try:
    from profiling.profile import ProfilingData
    from profiling.initialize_profiling import get_profiling_data   # <<--- CHANGE THIS to your module
except Exception as e:
    raise ImportError("Edit the import at top of script to point to your get_profiling_data function. Error: " + str(e))
# ------------------------------------------------------------------

# Simulation parameters (tuneable)
W_mhz = 10.0
P_tx_w = 0.5
N0_w_per_hz = 1e-20
RTT_ms = 4.5
P_comm = None  # will use profiling.edge_communication_power
P_idle = None  # will use profiling.edge_idle_power

# cost weights (matching MEC-like weighting)
W_TIME = 0.5
W_ENERGY = 0.5

# helper functions
def upload_rate_mbps(W_mhz, K_offloading, P_tx_w, h, N0_w_per_hz):
    if K_offloading <= 0: return 0.0
    W = W_mhz * 1e6
    W_per_user = W / K_offloading
    noise_power = N0_w_per_hz * W_per_user
    if noise_power <= 0: return 0.0
    snr = (P_tx_w * h) / noise_power
    r_bps = W_per_user * math.log2(1.0 + max(0.0, snr))
    return r_bps / 1e6

def offload_time_ms(upload_kb, rate_mbps, cloud_time_ms, queue_ms, rtt_ms=RTT_ms):
    if rate_mbps <= 0:
        return 1e9
    tx_ms = (upload_kb * 8.0) / rate_mbps
    return tx_ms + cloud_time_ms + queue_ms + rtt_ms

def energy_offload_j(upload_kb, rate_mbps, cloud_time_ms, idle_power_w, comm_power_w, queue_ms):
    if rate_mbps <= 0:
        return 1e9
    tx_s = (upload_kb * 8.0) / (rate_mbps * 1000.0)
    idle_s = (cloud_time_ms + queue_ms) / 1000.0
    e_comm = comm_power_w * tx_s
    e_idle = idle_power_w * idle_s
    return e_comm + e_idle

def total_cost(energy_j, delay_ms, w_time=W_TIME, w_energy=W_ENERGY):
    return w_time * delay_ms + w_energy * energy_j

# collapse profiling to single task (whole DNN per UAV)
def collapse_full_task(prof):
    total_edge_time = 0.0
    total_cloud_time = 0.0
    total_upload_kb = 0.0
    total_edge_energy_j = 0.0
    total_edge_power = 0.0
    node_count = 0
    for (l, n), t in prof.node_edge_times.items():
        total_edge_time += float(t)
        node_count += 1
    for (l, n), t in prof.node_cloud_times.items():
        total_cloud_time += float(t)
    for (l, n), s in prof.output_size.items():
        total_upload_kb += float(s)
    for (l, n), p in prof.node_edge_powers.items():
        total_edge_power += float(p)
        total_edge_energy_j += float(p) * (prof.node_edge_times[(l, n)]/1000.0)
    avg_edge_power = total_edge_power / max(1, node_count)
    task = {
        "edge_time_ms": total_edge_time,
        "cloud_time_ms": total_cloud_time,
        "upload_kb": total_upload_kb,
        "edge_energy_j": total_edge_energy_j,
        "edge_power_w_avg": avg_edge_power,
        "idle_power_w": prof.edge_idle_power,
        "comm_power_w": prof.edge_communication_power
    }
    return task

# DP oracle (per sample deterministic)
def dp_oracle_decision(task, rate_mbps, queue_ms, deadline_ms):
    # compute costs for local and offload
    delay_local = task["edge_time_ms"]
    energy_local = task["edge_energy_j"]
    cost_local = total_cost(energy_local, delay_local)

    delay_off = offload_time_ms(task["upload_kb"], rate_mbps, task["cloud_time_ms"], queue_ms)
    energy_off = energy_offload_j(task["upload_kb"], rate_mbps, task["cloud_time_ms"], task["idle_power_w"], task["comm_power_w"], queue_ms)
    cost_off = total_cost(energy_off, delay_off)

    feasible_local = (delay_local <= deadline_ms)
    feasible_off = (delay_off <= deadline_ms)
    # choose best feasible; if both infeasible pick one with smaller cost (MEC did similar)
    if feasible_local and feasible_off:
        pick = 0 if cost_local <= cost_off else 1
    elif feasible_local:
        pick = 0
    elif feasible_off:
        pick = 1
    else:
        pick = 0 if cost_local <= cost_off else 1
    return pick, {"local": (delay_local, energy_local, cost_local), "off": (delay_off, energy_off, cost_off)}

# Greedy: offload if (comm+cloud) < edge_time (no deadline awareness)
def greedy_decision(task, rate_mbps, queue_ms):
    delay_off = offload_time_ms(task["upload_kb"], rate_mbps, task["cloud_time_ms"], queue_ms)
    return 1 if delay_off < task["edge_time_ms"] else 0

# Threshold: offload if rate>thr and some slack positive (we use simple slack here)
def threshold_decision(task, rate_mbps, queue_ms, deadline_ms, thr_mbps=1.0):
    slack = deadline_ms - task["edge_time_ms"]
    return 1 if (rate_mbps > thr_mbps and slack > 0) else 0

# Tabular Q-learner that uses MEC-paper state (tc, ac) discretized
class MecStateQAgent:
    def __init__(self, tc_bins=8, ac_bins=6, alpha=0.1, gamma=0.95, eps=1.0, eps_decay=0.995, eps_min=0.05):
        self.tc_bins = tc_bins
        self.ac_bins = ac_bins
        self.alpha = alpha; self.gamma = gamma
        self.eps = eps; self.eps_decay = eps_decay; self.eps_min = eps_min
        self.Q = defaultdict(lambda: np.zeros(2))  # actions: 0 local, 1 offload

    def discretize(self, tc, ac, tc_cap=5000.0, ac_cap=1.0):
        # clamp and bin tc and ac
        tc_clamped = max(0.0, min(tc, tc_cap))
        tc_idx = int((tc_clamped / tc_cap) * (self.tc_bins - 1))
        ac_clamped = max(0.0, min(ac, ac_cap))
        ac_idx = int((ac_clamped / ac_cap) * (self.ac_bins - 1))
        return (tc_idx, ac_idx)

    def select(self, state):
        if random.random() < self.eps:
            return random.choice([0,1])
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next):
        q = self.Q[s][a]
        qn = np.max(self.Q[s_next])
        self.Q[s][a] = q + self.alpha * (r + self.gamma * qn - q)

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

# Main experiment
def run_mec_style_experiment(deadline_ms=500.0, N=8, trials=2000, episodes=3000, F_mec=10.0, p_offload_est=0.5, seed=None):
    """
    F_mec: total MEC capacity (arbitrary units). We'll map each offloaded task's CPU demand to task["cloud_time_ms"] scaled.
    p_offload_est: used to estimate K in training; actual K in each sample is determined by policy in evaluation.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    prof = get_profiling_data(deadline_ms)
    task_template = collapse_full_task(prof)
    # set comm/idle globals
    global P_comm, P_idle
    P_comm = task_template["comm_power_w"]
    P_idle = task_template["idle_power_w"]

    # We'll train a MecStateQAgent on simulated episodes using same stochastic models
    agent = MecStateQAgent()
    # simple CPU mapping: define each task CPU_request = task.cloud_time_ms / 1000.  (units of seconds)
    # MEC capacity F_mec is in same units (seconds of CPU available concurrently) - this is coarse but consistent with MEC state "ac".
    cpu_req = task_template["cloud_time_ms"] / 1000.0

    # Training (episodes): at each episode we simulate a batch of N users making decisions according to current policy (since MEC paper does joint decisions)
    for ep in range(episodes):
        # start episode with empty allocations -> ac = F_mec
        tc = 0.0
        allocated = 0.0
        # We will process users sequentially in a random order; MEC paper did centralized decisions, but tabular agent will see tc and ac incremental
        for u in range(N):
            # sample channel & queue for this user
            h = random.uniform(0.2, 1.0)
            # we approximate current K as number of already allocated (allocated >0) + 1 if this user offloads
            # For training, estimate K as max(1, int(N * p_offload_est))
            K_est = max(1, int(N * p_offload_est))
            rate = upload_rate_mbps(W_mhz, K_est, P_tx_w, h, N0_w_per_hz)
            # simulate queue using your approach: T_queue = max_cloud_time * K * U(0,1)
            max_cloud_time = task_template["cloud_time_ms"]
            # here we estimate K current offloaders as allocated/count; for training we use K_est
            queue_ms = max_cloud_time * K_est * random.random()

            # agent state = (tc, ac)
            ac = max(0.0, F_mec - allocated)
            s = agent.discretize(tc, ac, tc_cap=20000.0, ac_cap=F_mec)
            a = agent.select(s)

            if a == 0:
                delay = task_template["edge_time_ms"]; energy = task_template["edge_energy_j"]
            else:
                delay = offload_time_ms(task_template["upload_kb"], rate, task_template["cloud_time_ms"], queue_ms)
                energy = energy_offload_j(task_template["upload_kb"], rate, task_template["cloud_time_ms"], task_template["idle_power_w"], task_template["comm_power_w"], queue_ms)
                # if offloaded, allocate MEC cpu (coarse)
                allocated += cpu_req

            cost = total_cost(energy, delay)
            tc_next = tc + cost
            ac_next = max(0.0, F_mec - allocated)
            s_next = agent.discretize(tc_next, ac_next, tc_cap=20000.0, ac_cap=F_mec)
            reward = -cost
            agent.update(s, a, reward, s_next)
            agent.decay()
            tc = tc_next

    # Evaluation: compare DP, Greedy, Threshold, and agent (centralized decisions)
    dp_costs = []; greedy_costs = []; thr_costs = []; agent_costs = []
    dp_deadline_ok = []; greedy_deadline_ok = []; thr_deadline_ok = []; agent_deadline_ok = []

    for t in range(trials):
        # sample per-user channels and queues, and then compute joint decisions for each method
        # compute K per policy as the number of users offloading (policy dependent)
        # We'll do decisions sequentially but compute joint totals.
        # For fair comparison we process users in random order each trial.
        user_order = list(range(N))
        random.shuffle(user_order)

        # ---- DP Oracle centralized: compute for each user the cost if local vs offload under sampled env,
        # and pick the feasible option with lowest cost, but DP oracle ideally considers global allocation - we use greedy optimal per-sample here.
        # For better oracle you'd solve combinatorially; this is a practical approximation.
        decisions_dp = []
        decisions_greedy = []
        decisions_thr = []
        decisions_agent = []

        # sample all channels and queue values for each user now
        hs = [random.uniform(0.2,1.0) for _ in range(N)]
        queues = [task_template["cloud_time_ms"] * (random.random() * N) for _ in range(N)]  # your queue model using K~N*U(0,1)
        rates = [upload_rate_mbps(W_mhz, max(1, int(N*0.5)), P_tx_w, hs[i], N0_w_per_hz) for i in range(N)]

        # DP decisions per user (approx): choose the option with smaller cost among feasible options
        dp_cost = 0.0; greedy_cost = 0.0; thr_cost = 0.0; agent_cost = 0.0
        dp_dead_ok = True; greedy_dead_ok = True; thr_dead_ok = True; agent_dead_ok = True

        # For agent, we will use the learned Q-table sequentially with state updates
        tc_agent = 0.0; allocated_agent = 0.0
        tc_dp = 0.0; allocated_dp = 0.0

        for i in range(N):
            # DP
            rate = rates[i]; queue_ms = queues[i]
            pick, infos = dp_oracle_decision(task_template, rate, queue_ms, deadline_ms)
            if pick == 0:
                delay, energy, cost = infos["local"]
            else:
                delay, energy, cost = infos["off"]
            dp_cost += cost
            if delay > deadline_ms:
                dp_dead_ok = False

            # Greedy
            a_g = greedy_decision(task_template, rates[i], queues[i])
            if a_g == 0:
                d_g = task_template["edge_time_ms"]; e_g = task_template["edge_energy_j"]
            else:
                d_g = offload_time_ms(task_template["upload_kb"], rates[i], task_template["cloud_time_ms"], queues[i])
                e_g = energy_offload_j(task_template["upload_kb"], rates[i], task_template["cloud_time_ms"], task_template["idle_power_w"], task_template["comm_power_w"], queues[i])
            greedy_cost += total_cost(e_g, d_g)
            if d_g > deadline_ms:
                greedy_dead_ok = False

            # Threshold
            a_t = threshold_decision(task_template, rates[i], queues[i], deadline_ms, thr_mbps=1.0)
            if a_t == 0:
                d_t = task_template["edge_time_ms"]; e_t = task_template["edge_energy_j"]
            else:
                d_t = offload_time_ms(task_template["upload_kb"], rates[i], task_template["cloud_time_ms"], queues[i])
                e_t = energy_offload_j(task_template["upload_kb"], rates[i], task_template["cloud_time_ms"], task_template["idle_power_w"], task_template["comm_power_w"], queues[i])
            thr_cost += total_cost(e_t, d_t)
            if d_t > deadline_ms:
                thr_dead_ok = False

            # Agent (MEC-state)
            ac = max(0.0, F_mec - allocated_agent)
            s = agent.discretize(tc_agent, ac, tc_cap=20000.0, ac_cap=F_mec)
            if s in agent.Q:
                a_a = int(np.argmax(agent.Q[s]))
            else:
                a_a = random.choice([0,1])
            if a_a == 0:
                d_a = task_template["edge_time_ms"]; e_a = task_template["edge_energy_j"]
            else:
                d_a = offload_time_ms(task_template["upload_kb"], rates[i], task_template["cloud_time_ms"], queues[i])
                e_a = energy_offload_j(task_template["upload_kb"], rates[i], task_template["cloud_time_ms"], task_template["idle_power_w"], task_template["comm_power_w"], queues[i])
                allocated_agent += cpu_req
            agent_cost += total_cost(e_a, d_a)
            if d_a > deadline_ms:
                agent_dead_ok = False
            tc_agent += total_cost(e_a, d_a)

        dp_costs.append(dp_cost); greedy_costs.append(greedy_cost); thr_costs.append(thr_cost); agent_costs.append(agent_cost)
        dp_deadline_ok.append(dp_dead_ok); greedy_deadline_ok.append(greedy_dead_ok); thr_deadline_ok.append(thr_dead_ok); agent_deadline_ok.append(agent_dead_ok)

    # aggregate
    out = {
        "dp_mean_cost": np.mean(dp_costs),
        "greedy_mean_cost": np.mean(greedy_costs),
        "thr_mean_cost": np.mean(thr_costs),
        "agent_mean_cost": np.mean(agent_costs),
        "dp_deadline_prob": np.mean(dp_deadline_ok),
        "greedy_deadline_prob": np.mean(greedy_deadline_ok),
        "thr_deadline_prob": np.mean(thr_deadline_ok),
        "agent_deadline_prob": np.mean(agent_deadline_ok),
        # Also return per-user deterministic local baseline (sum)
        "local_total_time_ms": task_template["edge_time_ms"] * N,
        "local_total_energy_j": task_template["edge_energy_j"] * N
    }
    return out

# If run as script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=float, default=500.0)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--F_mec", type=float, default=10.0)
    args = parser.parse_args()

    res = run_mec_style_experiment(deadline_ms=args.deadline, N=args.N, trials=args.trials, episodes=args.episodes, F_mec=args.F_mec)
    print("\n=== MEC-style experiment results on your profiling ===")
    print(res)
