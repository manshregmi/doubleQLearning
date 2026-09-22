from a2c.actor_critic_agent import TabularActorCriticAgent
from profiling.profile import ProfilingData
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 28
plt.rcParams['lines.linewidth'] = 3


def run_a2c_simulation(
    profiling_data: ProfilingData,
    episodes=1000,
    max_steps=10,
    is_test=False,
    visualize_stats=True,
    plot_rewards=False,
    smoothing_window=50,
    trace_csv_path=None,
    timeout_threshold_ms=150,
):
    agent = TabularActorCriticAgent(profiling_data, is_test=is_test)
    if trace_csv_path:
        from simulator.simulator import CloudEdgeSimulator
        agent.simulator = CloudEdgeSimulator(
            profiling_data,
            trace_csv_path=trace_csv_path,
            timeout_threshold_ms=timeout_threshold_ms
        )
    agent.load()

    edge_energy = []
    completion_time = []
    rewards = []
    episode_modified_rewards = []
    timeout_count = 0
    chain_lengths = []
    action_counts = {0: 0, 1: 0, 2: 0}  # Track action choices

    deadline_missed_count = 0
    deadline_met_count = 0

    if visualize_stats:
        edge_execution_stats = defaultdict(int)
        optimistic_execution_stats = defaultdict(int)
        conservative_execution_stats = defaultdict(int)

    start_time = time.time()
    print(f"Starting EdgeWise A2C simulation at: {start_time}")
    print(f"  Episodes: {episodes}")
    print(f"  Timeout threshold: {timeout_threshold_ms} ms")

    for ep in range(episodes):
        agent.simulator.reset_episode_time()
        bw = agent.simulator.get_current_bandwidth()
        rtt = agent.simulator.get_current_rtt()
        current_state = (bw, rtt, 0.0, 0, None, 0.0, 0, 0)

        total_energy = 0.0
        total_time = 0.0
        total_reward = 0.0
        trajectory = []
        episode_timeouts = 0
        episode_action_counts = {0: 0, 1: 0, 2: 0}

        for step in range(max_steps):
            state_key = agent._state_to_key(current_state)

            action, reward, next_state, terminal, energy, completion_time_s, \
                new_bandwidth, surplus, fractional_deadline, neg_count = agent.train(current_state)

            # Track action
            action_mode = int(np.max(action[:, 1]))
            episode_action_counts[action_mode] += 1
            action_counts[action_mode] += 1

            if hasattr(agent.simulator, 'last_timeout_occurred') and agent.simulator.last_timeout_occurred:
                episode_timeouts += 1
                timeout_count += 1

            chain_lengths.append(current_state[7])

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

            trajectory.append({
                "state_key": state_key,
                "action_key": agent._action_to_key(action),
                "original_reward": reward,
                "reward": reward,
                "surplus": surplus,
                "energy": energy,
                "time": completion_time_s,
                "timeout": hasattr(agent.simulator, 'last_timeout_occurred') and agent.simulator.last_timeout_occurred,
                "chain_length": current_state[7],
                "action_mode": action_mode,
            })

            total_energy += energy
            total_time += completion_time_s * 1000.0
            total_reward += reward
            current_state = next_state

            if terminal:
                break

        # Reward shaping based on deadline
        avg_step_reward = np.clip(
            np.mean([abs(s["original_reward"]) for s in trajectory]),
            100.0, 3000.0
        )

        deadline_violated = total_time > profiling_data.deadline

        if deadline_violated:
            deadline_missed_count += 1
            excess = total_time - profiling_data.deadline
            base_penalty = -0.6 * avg_step_reward
            scale = np.clip(excess / profiling_data.deadline, 0.0, 1.5)
            penalty = base_penalty * (1.0 + scale)
            for step in trajectory:
                step["reward"] += penalty / len(trajectory)
        else:
            deadline_met_count += 1
            saved = profiling_data.deadline - total_time
            bonus = 0.25 * avg_step_reward * (1.0 + saved / profiling_data.deadline)
            for step in trajectory:
                step["reward"] += bonus / len(trajectory)

        for step in trajectory:
            step["reward"] = np.clip(step["reward"], -500.0, 50.0)

        if not is_test:
            agent.update_trajectory(trajectory)
            agent.notify_episode_end(sum(step["reward"] for step in trajectory))

        modified_reward = sum(step["reward"] for step in trajectory)

        edge_energy.append(total_energy)
        completion_time.append(total_time)
        rewards.append(total_reward)
        episode_modified_rewards.append(modified_reward)

        # Print with action distribution
        if (ep + 1) % 100 == 0 or ep == 0:
            action_str = f"E:{episode_action_counts[0]} O:{episode_action_counts[1]} C:{episode_action_counts[2]}"
            print(f"Episode {ep+1}/{episodes}: Energy={total_energy:.2f}J, Time={total_time:.1f}ms, "
                  f"Reward={modified_reward:.1f}, Timeouts={episode_timeouts}, "
                  f"ChainLen={current_state[7]}, Actions=[{action_str}]")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE")
    print(f"  Episodes: {episodes}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Avg Energy: {np.mean(edge_energy):.3f} J")
    print(f"  Avg Time: {np.mean(completion_time):.1f} ms")
    print(f"  Deadline Met: {deadline_met_count}/{episodes} ({deadline_met_count/episodes*100:.1f}%)")
    print(f"  Total Timeouts: {timeout_count}")
    print(f"  Avg Chain Length: {np.mean(chain_lengths):.2f}")
    print(f"  Action Distribution: Edge={action_counts[0]}, Opt={action_counts[1]}, Cons={action_counts[2]}")
    print(f"{'='*60}")

    if plot_rewards and episodes > 1:
        plot_smoothed_reward(episode_modified_rewards, smoothing_window)

    if visualize_stats:
        print_execution_stats(edge_execution_stats, optimistic_execution_stats, 
                             conservative_execution_stats, profiling_data, episodes)

    agent.save()
    return np.mean(edge_energy), np.mean(completion_time), deadline_missed_count


def plot_smoothed_reward(rewards, smoothing_window=50):
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = pd.Series(rewards).rolling(window=smoothing_window, center=True, min_periods=1).mean()
    
    plt.figure(figsize=(14, 8))
    plt.plot(episodes, smoothed, color='#1f77b4', linewidth=3)
    plt.xlabel('Episode', fontsize=28, fontfamily='Times New Roman')
    plt.ylabel('Smoothed Reward', fontsize=28, fontfamily='Times New Roman')
    plt.title('A2C Convergence (EdgeWise)', fontsize=28, fontfamily='Times New Roman', fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.xticks(fontsize=24, fontfamily='Times New Roman')
    plt.yticks(fontsize=24, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.show()


def print_execution_stats(edge_stats, opt_stats, cons_stats, profiling_data, total_episodes):
    print("\n" + "=" * 90)
    print("NODE EXECUTION STATISTICS (EdgeWise)")
    print("=" * 90)
    print(f"{'Node':<10} {'Edge':<10} {'Opt':<10} {'Cons':<10} {'Total':<10} {'E%':<8} {'O%':<8} {'C%':<8} {'Dominant':<12}")
    print("-" * 90)

    layer_names = ['v1', 'v2', 'v3', ['v4','v7','v10'], ['v5','v8','v11'], ['v6','v9','v12'], 'v13']

    for layer_idx, layer_nodes in enumerate(profiling_data.layers):
        for node_idx in range(len(layer_nodes)):
            key = (layer_idx, node_idx)
            e = edge_stats.get(key, 0)
            o = opt_stats.get(key, 0)
            c = cons_stats.get(key, 0)
            total = e + o + c

            if total > 0:
                ep = (e / total) * 100
                op = (o / total) * 100
                cp = (c / total) * 100
            else:
                ep = op = cp = 0

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