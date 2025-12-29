from a2c.actor_critic_agent import TabularActorCriticAgent
from profiling.profile import ProfilingData
import numpy as np
import time
from collections import defaultdict


def run_a2c_simulation(
    profiling_data: ProfilingData,
    episodes=1,
    max_steps=20,
    is_test=False,
):
    agent = TabularActorCriticAgent(profiling_data, is_test=is_test)
    agent.load()

    edge_energy = []
    completion_time = []
    rewards = []
    cumulative_rewards = []

    deadline_missed_count = 0
    deadline_met_count = 0
    layer_violation_stats = defaultdict(int)

    start_time = time.time()
    print(f"Starting A2C simulation at: {start_time}")
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

            state_key = agent._state_to_key(current_state)

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


            bandwidth = next_state[0]

            prev_neg = current_state[5]
            neg_increased = neg_count > prev_neg

            trajectory.append({
                "state_key": state_key,
                "action_key": agent._action_to_key(action),
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

        # ==================================================
        # Deadline reward reshaping (IDENTICAL)
        # ==================================================
        avg_step_reward = np.clip(
            np.mean([abs(s["original_reward"]) for s in trajectory]),
            300.0, 3000.0
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
        cumulative_rewards.append(modified_reward)

        if ep < 5 or (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Time={total_time:.1f}ms | "
                f"Energy={total_energy:.2f}J | "
                f"{'MISS' if deadline_violated else 'MET '} | "
                f"Reward={modified_reward:.2f}"
            )

    agent.save()

    print("\nSuccess rate:",
          100 * deadline_met_count / episodes, "%")

    return (
        np.mean(edge_energy),
        np.mean(completion_time),
        deadline_missed_count,
    )
