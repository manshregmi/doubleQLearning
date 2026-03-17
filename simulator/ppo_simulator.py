from a2c.ppo_agent import PPOAgent
from profiling.profile import ProfilingData
import numpy as np
import time
import random
from collections import defaultdict


def run_ppo_simulation(
    profiling_data: ProfilingData,
    episodes=1,
    max_steps=20,
    is_test=False,
    visualize_stats=True,
):
    """
    PPO simulator that exactly matches A2C's logical flow
    """
    agent = PPOAgent(profiling_data, is_test=is_test)
    agent.load()
    
    # Metrics tracking
    edge_energy = []
    completion_time = []
    rewards = []
    cumulative_rewards = []
    
    episode_rewards = []
    episode_modified_rewards = []

    deadline_missed_count = 0
    deadline_met_count = 0
    layer_violation_stats = defaultdict(int)
    
    if visualize_stats:
        edge_execution_stats = defaultdict(int)
        cloud_execution_stats = defaultdict(int)

    start_time = time.time()
    print(f"Starting PPO simulation at: {start_time}")
    bandwidth = profiling_data.bandwidth

    for ep in range(episodes):
        total_energy = 0.0
        total_time = 0.0
        total_reward = 0.0

        cloud_time = 0.0
        current_state = (bandwidth, cloud_time, 0, None, 0, 0)

        trajectory = []
        step_surpluses = []

        for step in range(max_steps):
            try:
                (
                    action,
                    reward,
                    next_state,
                    terminal,
                    energy,
                    completion_time_s,
                    new_bandwidth,
                    surplus,
                    fractional_deadline,
                    neg_count,
                ) = agent.train(current_state)
            except Exception as e:
                print(f"Error in training step: {e}")
                # Fallback: take random action
                layer = int(current_state[2])
                possible_actions = agent._get_possible_actions(layer)
                action = random.choice(possible_actions)
                
                next_cloud = agent.simulator.get_next_state_cloud_waiting_time(
                    next_layer=min(layer + 1, len(profiling_data.layers) - 1),
                    current_action=action,
                    isAllCloud=False,
                )
                
                energy, completion_time_s = agent.simulator.compute_energy_and_time(
                    current_state=current_state,
                    current_action=action,
                    cloud_pending_ms=next_cloud,
                )
                
                reward, surplus, neg_count, fractional_deadline = \
                    agent.simulator.calculate_reward(
                        layer,
                        energy,
                        completion_time_s,
                        current_state[4],
                        current_state[5],
                        isA2C=True,
                    )
                
                next_state, terminal, _ = agent.simulator.get_next_state(
                    current_state,
                    action,
                    surplus,
                    neg_count,
                    new_cloud_pending=next_cloud,
                )
            
            # Track execution stats
            if visualize_stats:
                current_layer = int(current_state[2])
                for node_idx, (_, location) in enumerate(action):
                    key = (current_layer, node_idx)
                    if location == 0:
                        edge_execution_stats[key] += 1
                    else:
                        cloud_execution_stats[key] += 1

            bandwidth = next_state[0]
            cloud_time = next_state[1]

            prev_neg = current_state[5]
            neg_increased = neg_count > prev_neg

            trajectory.append({
                "original_reward": reward,
                "reward": reward,
                "surplus": surplus,
                "negative_surplus_increased": neg_increased,
            })

            if neg_increased:
                layer_violation_stats[int(current_state[2])] += 1

            step_surpluses.append(surplus)

            total_energy += energy
            total_time += completion_time_s * 1000.0
            total_reward += reward

            current_state = next_state

            if terminal:
                break

        # Reward reshaping
        if trajectory:
            avg_step_reward = np.clip(
                np.mean([abs(s["original_reward"]) for s in trajectory if s["original_reward"] != 0]),
                300.0, 3000.0
            )
        else:
            avg_step_reward = 300.0

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

        # Clip rewards
        for step in trajectory:
            step["reward"] = np.clip(step["reward"], -500.0, 50.0)

        # Update agent at episode end
        if not is_test and trajectory:
            agent.notify_episode_end(sum(step["reward"] for step in trajectory))

        modified_reward = sum(step["reward"] for step in trajectory) if trajectory else 0

        # Store metrics
        edge_energy.append(total_energy)
        completion_time.append(total_time)
        rewards.append(total_reward)
        cumulative_rewards.append(modified_reward)
        
        episode_rewards.append(total_reward)
        episode_modified_rewards.append(modified_reward)

        # Progress printing
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{episodes} - Reward: {modified_reward:.2f}, "
                  f"Time: {total_time:.2f}ms, Energy: {total_energy:.2f}J, "
                  f"Deadline Met: {not deadline_violated}")

    # Calculate statistics
    elapsed_time = time.time() - start_time
    print(f"\nPPO Simulation completed in {elapsed_time:.2f} seconds")
    print(f"Average Energy: {np.mean(edge_energy):.2f} J")
    print(f"Average Completion Time: {np.mean(completion_time):.2f} ms")
    print(f"Deadline Missed: {deadline_missed_count}/{episodes} ({deadline_missed_count/episodes*100:.1f}%)")
    print(f"Deadline Met: {deadline_met_count}/{episodes} ({deadline_met_count/episodes*100:.1f}%)")

    # Save agent
    agent.save()
    
    return (
        np.mean(edge_energy),
        np.mean(completion_time),
        deadline_missed_count,
    )