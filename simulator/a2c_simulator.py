from a2c.actor_critic_agent import TabularActorCriticAgent
from profiling.profile import ProfilingData
import numpy as np
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd


def run_a2c_simulation(
    profiling_data,
    episodes=1000,
    max_steps=10,
    is_test=False,
    visualize_stats=False,
    plot_rewards=False,
    smoothing_window=50,
    trace_csv_path=None,
    packet_loss_prob=0.10,
    model_path="a2c_tables.pkl",
    seed=None,
    action_modes=(0, 1, 2),   # (0,1,2) ternary, (0,1) optimistic, (0,2) conservative
    threshold_aware=True,     # include timeout ratio in the agent's state
    deadline_levels=(500.0,), # all deadlines the agent is configured for
    threshold_ratios=(0.10,), # all timeout ratios (timeout = ratio * deadline)
    deadline_ms=None,         # fixed deadline (evaluation); None -> sample from levels
    threshold_ratio=None,     # fixed ratio (evaluation);   None -> sample from ratios
    label="EdgeWise A2C",
    verbose=False,
):
    """Run training or evaluation.

    Each episode uses a deadline D and a timeout ratio rho, with
    timeout_threshold_ms = rho * D. During training (deadline_ms /
    threshold_ratio = None) both are sampled uniformly per episode.

    Reward (paper, constrained MDP):
        R_imm   = -eta * E                             (Eq. 10, eta = 150)
        R_final = R_imm - phi * max(0, delta - tau)/tau  on violation
                  (Eq. 11), applied only to levels that contributed a
                  slack violation (neg_delta > 0).
    No positive bonus for finishing early.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    deadline_levels = tuple(float(d) for d in deadline_levels)
    threshold_ratios = tuple(float(r) for r in threshold_ratios)

    agent = TabularActorCriticAgent(
        profiling_data, is_test=is_test,
        trace_csv_path=trace_csv_path,
        packet_loss_prob=packet_loss_prob,
        action_modes=action_modes,
        threshold_aware=threshold_aware,
        deadline_levels=deadline_levels,
        threshold_ratios=threshold_ratios,
        verbose=verbose,
    )
    agent.load(model_path)

    edge_energy = []
    completion_time = []
    rewards = []
    episode_modified_rewards = []
    per_episode_timeouts = []
    action_counts = {0: 0, 1: 0, 2: 0}

    deadline_missed_count = 0
    deadline_met_count = 0

    if visualize_stats:
        edge_execution_stats = defaultdict(int)
        optimistic_execution_stats = defaultdict(int)
        conservative_execution_stats = defaultdict(int)

    start_time = time.time()
    d_str = f"{deadline_ms:.0f}ms" if deadline_ms is not None else f"sampled {deadline_levels}"
    r_str = f"{threshold_ratio:.2f}" if threshold_ratio is not None else f"sampled {threshold_ratios}"
    print(f"Running {label} (is_test={is_test}, modes={tuple(action_modes)}, "
          f"ratio_in_state={threshold_aware})")
    print(f"  Episodes: {episodes} | deadline: {d_str} | timeout ratio: {r_str}")

    # Terminal penalty constant from the paper (Eq. 11)
    PHI = 1000.0

    for ep in range(episodes):
        # ---- episode context: deadline and timeout = ratio * deadline ----
        D = float(deadline_ms) if deadline_ms is not None else random.choice(deadline_levels)
        rho = float(threshold_ratio) if threshold_ratio is not None else random.choice(threshold_ratios)

        profiling_data.deadline = D          # used by reward + deadline check
        agent.simulator.set_timeout_threshold(rho * D)
        context = agent.context_bin(D, rho)

        agent.simulator.reset_episode_time()
        bw = agent.simulator.profiling.bandwidth
        rtt = agent.simulator.profiling.rtt
        current_state = (bw, rtt, 0.0, 0, None, 0.0, 0, 0, context)

        total_energy = 0.0
        total_time = 0.0
        total_reward = 0.0
        trajectory = []
        episode_timeouts = 0
        episode_action_counts = {0: 0, 1: 0, 2: 0}
        prev_neg = 0

        for step in range(max_steps):
            state_key = agent._state_to_key(current_state)

            action, reward, next_state, terminal, energy, completion_time_s, \
                new_bandwidth, surplus, fractional_deadline, neg_count = agent.train(current_state)

            action_mode = int(np.max(action[:, 1]))
            episode_action_counts[action_mode] += 1
            action_counts[action_mode] += 1

            if agent.simulator.last_timeout_occurred:
                episode_timeouts += 1

            if visualize_stats:
                current_layer = int(current_state[3])
                for node_idx, (_, location) in enumerate(action):
                    key = (current_layer, node_idx)
                    if location == 0:
                        edge_execution_stats[key] += 1
                    elif location == 1:
                        optimistic_execution_stats[key] += 1
                    else:
                        conservative_execution_stats[key] += 1

            neg_delta = int(neg_count) - int(prev_neg)
            prev_neg = int(neg_count)

            trajectory.append({
                "state_key": state_key,
                "action_key": agent._action_to_key(action),
                "reward": reward,
                "original_reward": reward,
                "surplus": surplus,
                "energy": energy,
                "time": completion_time_s,
                "timeout": agent.simulator.last_timeout_occurred,
                "chain_length": current_state[7],
                "action_mode": action_mode,
                "neg_delta": neg_delta,
            })

            total_energy += energy
            total_time += completion_time_s * 1000.0
            total_reward += reward
            current_state = next_state

            if terminal:
                break

        per_episode_timeouts.append(episode_timeouts)

        # ---------- Terminal deadline penalty (paper Eq. 11) ----------
        # Only applied if the global deadline is missed, and only to levels
        # whose execution caused a fractional-deadline slack violation.
        deadline_violated = total_time > D
        if deadline_violated:
            deadline_missed_count += 1
            penalty = PHI * max(0.0, total_time - D) / D
            violating = [i for i, s in enumerate(trajectory) if s["neg_delta"] > 0]
            if violating:
                for i in violating:
                    trajectory[i]["reward"] -= penalty
            else:
                # No level flagged a violation — still reflect the miss in the
                # episode return by penalizing the last step.
                trajectory[-1]["reward"] -= penalty
        else:
            deadline_met_count += 1
            # No positive bonus: this is a constrained problem, not a dual
            # energy-time objective.

        if not is_test:
            agent.update_trajectory(trajectory)
            agent.notify_episode_end(sum(s["reward"] for s in trajectory))

        modified_reward = sum(s["reward"] for s in trajectory)
        edge_energy.append(total_energy)
        completion_time.append(total_time)
        rewards.append(total_reward)
        episode_modified_rewards.append(modified_reward)

        if (ep + 1) % max(1, episodes // 10) == 0 or ep == 0:
            action_str = f"E:{episode_action_counts[0]} O:{episode_action_counts[1]} C:{episode_action_counts[2]}"
            print(f"[{label}] Ep {ep+1}/{episodes}: D={D:.0f}ms tau={rho*D:.0f}ms "
                  f"E={total_energy:.2f}J, T={total_time:.1f}ms, R={modified_reward:.1f}, "
                  f"Timeouts={episode_timeouts}, [{action_str}]")

    elapsed = time.time() - start_time
    unseen_rate = (agent.eval_unseen_states / agent.eval_decisions
                   if agent.eval_decisions else 0.0)
    print(f"\n{'='*60}")
    print(f"{label} COMPLETE: {episodes} episodes in {elapsed:.1f}s")
    print(f"  Avg Energy: {np.mean(edge_energy):.3f} J")
    print(f"  Avg Time:   {np.mean(completion_time):.1f} ms")
    print(f"  DL-Met:     {deadline_met_count}/{episodes} "
          f"({100.0*deadline_met_count/episodes:.1f}%)")
    print(f"  Avg Timeouts/ep: {np.mean(per_episode_timeouts):.2f}")
    print(f"  Actions: E={action_counts[0]}, O={action_counts[1]}, C={action_counts[2]}")
    if is_test:
        print(f"  Unseen states at eval: {agent.eval_unseen_states}/{agent.eval_decisions} "
              f"({100.0 * unseen_rate:.1f}%)")
    print(f"{'='*60}")

    if plot_rewards and episodes > 1:
        plot_smoothed_reward(episode_modified_rewards, smoothing_window, label)

    if visualize_stats:
        print_execution_stats(edge_execution_stats, optimistic_execution_stats,
                              conservative_execution_stats, profiling_data, episodes, label)

    # Only persist tables after training; evaluation must not overwrite them.
    if not is_test:
        agent.save(model_path)

    return {
        "avg_energy": float(np.mean(edge_energy)),
        "avg_time_ms": float(np.mean(completion_time)),
        "deadline_miss_rate": deadline_missed_count / episodes,
        "avg_timeouts_per_ep": float(np.mean(per_episode_timeouts)),
        "action_counts": dict(action_counts),
        "unseen_state_rate": unseen_rate,
        "n_episodes": episodes,
        "energies": edge_energy,
        "times": completion_time,
        "rewards": episode_modified_rewards,
    }


def plot_smoothed_reward(rewards, smoothing_window=50, label="EdgeWise"):
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = pd.Series(rewards).rolling(window=smoothing_window,
                                          center=True, min_periods=1).mean()
    plt.figure(figsize=(14, 8))
    plt.plot(episodes, smoothed, color='#1f77b4', linewidth=3)
    plt.xlabel('Episode', fontsize=28, fontfamily='Times New Roman')
    plt.ylabel('Smoothed Reward', fontsize=28, fontfamily='Times New Roman')
    plt.title(f'A2C Convergence ({label})', fontsize=28,
              fontfamily='Times New Roman', fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def print_execution_stats(edge_stats, opt_stats, cons_stats, profiling_data,
                          total_episodes, label="EdgeWise"):
    print("\n" + "=" * 90)
    print(f"NODE EXECUTION STATISTICS ({label})")
    print("=" * 90)
    print(f"{'Node':<10} {'Edge':<10} {'Opt':<10} {'Cons':<10} "
          f"{'Total':<10} {'E%':<8} {'O%':<8} {'C%':<8} {'Dominant':<12}")
    print("-" * 90)

    layer_names = ['v1', 'v2', 'v3', ['v4', 'v7', 'v10'],
                   ['v5', 'v8', 'v11'], ['v6', 'v9', 'v12'], 'v13']

    for layer_idx, layer_nodes in enumerate(profiling_data.layers):
        for node_idx in range(len(layer_nodes)):
            key = (layer_idx, node_idx)
            e = edge_stats.get(key, 0)
            o = opt_stats.get(key, 0)
            c = cons_stats.get(key, 0)
            total = e + o + c
            ep = (e/total)*100 if total else 0
            op = (o/total)*100 if total else 0
            cp = (c/total)*100 if total else 0

            if len(layer_nodes) == 1:
                node_label = layer_names[layer_idx]
            else:
                node_label = layer_names[layer_idx][node_idx]

            if ep > 60:
                dominant = "EDGE"
            elif op >= cp:
                dominant = "OPTIMISTIC"
            else:
                dominant = "CONSERVATIVE"

            print(f"{node_label:<10} {e:<10} {o:<10} {c:<10} {total:<10} "
                  f"{ep:<8.1f} {op:<8.1f} {cp:<8.1f} {dominant:<12}")
    print("=" * 90)