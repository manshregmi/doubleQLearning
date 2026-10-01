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
    timeout_threshold_ms=150.0,
    packet_loss_prob=0.10,
    model_path="a2c_tables.pkl",
    train_thresholds=None,  # list of thresholds to sample during training
    seed=None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    agent = TabularActorCriticAgent(
        profiling_data, is_test=is_test,
        trace_csv_path=trace_csv_path,
        timeout_threshold_ms=timeout_threshold_ms,
        packet_loss_prob=packet_loss_prob,
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
    print(f"Running EdgeWise A2C (is_test={is_test})")
    print(f"  Episodes: {episodes}")
    print(f"  timeout_threshold_ms: {timeout_threshold_ms}")

    for ep in range(episodes):
        # Sample random threshold during training to make agent threshold-aware
        if not is_test and train_thresholds is not None:
            th = float(random.choice(train_thresholds))
            agent.simulator.set_timeout_threshold(th)

        agent.simulator.reset_episode_time()
        bw = agent.simulator.profiling.bandwidth
        rtt = agent.simulator.profiling.rtt
        timeout_bin = agent._timeout_bin()
        current_state = (bw, rtt, 0.0, 0, None, 0.0, 0, 0, timeout_bin)

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

        # ---------- Terminal deadline penalty (retroactive) ----------
        avg_step_reward = float(np.clip(
            np.mean([abs(s["original_reward"]) for s in trajectory]),
            100.0, 3000.0
        ))

        deadline_violated = total_time > profiling_data.deadline
        if deadline_violated:
            deadline_missed_count += 1
            excess = total_time - profiling_data.deadline
            base_penalty = -0.6 * avg_step_reward
            scale = float(np.clip(excess / profiling_data.deadline, 0.0, 1.5))
            penalty = base_penalty * (1.0 + scale)

            # Apply only to levels that caused slack violations
            violating = [i for i, s in enumerate(trajectory) if s["neg_delta"] > 0]
            if violating:
                per = penalty / len(violating)
                for i in violating:
                    trajectory[i]["reward"] += per
            else:
                # No individual violators recorded — apply uniformly
                for s in trajectory:
                    s["reward"] += penalty / len(trajectory)
        else:
            deadline_met_count += 1
            saved = profiling_data.deadline - total_time
            bonus = 0.25 * avg_step_reward * (1.0 + saved / profiling_data.deadline)
            for s in trajectory:
                s["reward"] += bonus / len(trajectory)

        for s in trajectory:
            s["reward"] = float(np.clip(s["reward"], -500.0, 50.0))

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
            print(f"Ep {ep+1}/{episodes}: E={total_energy:.2f}J, "
                  f"T={total_time:.1f}ms, R={modified_reward:.1f}, "
                  f"Timeouts={episode_timeouts}, [{action_str}]")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE: {episodes} episodes in {elapsed:.1f}s")
    print(f"  Avg Energy: {np.mean(edge_energy):.3f} J")
    print(f"  Avg Time:   {np.mean(completion_time):.1f} ms")
    print(f"  DL-Met:     {deadline_met_count}/{episodes} "
          f"({100.0*deadline_met_count/episodes:.1f}%)")
    print(f"  Avg Timeouts/ep: {np.mean(per_episode_timeouts):.2f}")
    print(f"  Actions: E={action_counts[0]}, O={action_counts[1]}, C={action_counts[2]}")
    print(f"{'='*60}")

    if plot_rewards and episodes > 1:
        plot_smoothed_reward(episode_modified_rewards, smoothing_window)

    if visualize_stats:
        print_execution_stats(edge_execution_stats, optimistic_execution_stats,
                              conservative_execution_stats, profiling_data, episodes)

    agent.save(model_path)

    return {
        "avg_energy": float(np.mean(edge_energy)),
        "avg_time_ms": float(np.mean(completion_time)),
        "deadline_miss_rate": deadline_missed_count / episodes,
        "avg_timeouts_per_ep": float(np.mean(per_episode_timeouts)),
        "energies": edge_energy,
        "times": completion_time,
        "rewards": episode_modified_rewards,
    }


def plot_smoothed_reward(rewards, smoothing_window=50):
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = pd.Series(rewards).rolling(window=smoothing_window,
                                          center=True, min_periods=1).mean()
    plt.figure(figsize=(14, 8))
    plt.plot(episodes, smoothed, color='#1f77b4', linewidth=3)
    plt.xlabel('Episode', fontsize=28, fontfamily='Times New Roman')
    plt.ylabel('Smoothed Reward', fontsize=28, fontfamily='Times New Roman')
    plt.title('A2C Convergence (EdgeWise)', fontsize=28,
              fontfamily='Times New Roman', fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def print_execution_stats(edge_stats, opt_stats, cons_stats, profiling_data, total_episodes):
    print("\n" + "=" * 90)
    print("NODE EXECUTION STATISTICS (EdgeWise)")
    print("=" * 90)
    print(f"{'Node':<10} {'Edge':<10} {'Opt':<10} {'Cons':<10} "
          f"{'Total':<10} {'E%':<8} {'O%':<8} {'C%':<8} {'Dominant':<12}")
    print("-" * 90)

    layer_names = ['v1', 'v2', 'v3', ['v4','v7','v10'],
                   ['v5','v8','v11'], ['v6','v9','v12'], 'v13']

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