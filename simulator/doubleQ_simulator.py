from model.doubleQ import DoubleQLearningAgent
from profiling.profile import ProfilingData
import numpy as np
import time
from collections import defaultdict


def run_simulation(
    profiling_data: ProfilingData,
    episodes=1,
    max_steps=20,
    is_test=False,
):
    agent = DoubleQLearningAgent(profiling_data, is_test=is_test)
    agent.load_qtables()

    edge_energy = []
    completion_time = []
    rewards = []
    cumulative_rewards = []
    episode_computation_time = []

    deadline_missed_count = 0
    deadline_met_count = 0
    layer_violation_stats = defaultdict(int)

    start_time = time.time()
    print(f"Starting simulation at: {start_time}")

    for ep in range(episodes):

        total_energy = 0.0
        total_time = 0.0
        total_reward = 0.0
        total_computation_time = 0.0

        cloud_time = 0.0
        current_state = (profiling_data.bandwidth, cloud_time, 0, None, 0, 0)

        trajectory = []
        step_surpluses = []
        overall_action = []

        # ==========================================================
        # Rollout episode
        # ==========================================================
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
                negative_surplus_count,
                computation_time,
            ) = agent.train(current_state)
            total_computation_time += computation_time 

            # Compute boolean flag required by agent
            prev_negative_surplus = current_state[5]
            negative_surplus_increased = negative_surplus_count > prev_negative_surplus

            action_key = agent._action_to_key(action)

            trajectory.append({
                "state_key": state_key,
                "action_key": action_key,
                "original_reward": reward,
                "energy": energy,
                "completion_time": completion_time_s * 1000.0,
                "surplus": surplus,
                "layer": int(current_state[2]),
                "negative_surplus_count": negative_surplus_count,
                "negative_surplus_increased": negative_surplus_increased,
                "terminal": terminal,
            })

            step_surpluses.append(surplus)

            if negative_surplus_increased:
                layer_violation_stats[int(current_state[2])] += 1

            total_energy += energy
            total_time += completion_time_s * 1000.0
            total_reward += reward

            overall_action.append(tuple(map(tuple, action)))
            current_state = next_state

            if terminal:
                break

        episode_computation_time.append(total_computation_time)


        # ==========================================================
        # Compute reward scale from YOUR reward function
        # ==========================================================
        step_reward_magnitudes = [
            abs(step["original_reward"]) for step in trajectory
        ]
        avg_step_reward = np.mean(step_reward_magnitudes)
        avg_step_reward = np.clip(avg_step_reward, 30.0, 300.0)

        # ==========================================================
        # Deadline check
        # ==========================================================
        deadline_violated = total_time > profiling_data.deadline

        if deadline_violated:
            deadline_missed_count += 1
            excess_time = total_time - profiling_data.deadline
            base_penalty = -0.6 * avg_step_reward

            time_scale = np.clip(
                excess_time / profiling_data.deadline, 0.0, 1.5
            )

            negative_surpluses = [abs(s) for s in step_surpluses if s < 0]
            surplus_scale = np.clip(
                (np.mean(negative_surpluses) / profiling_data.deadline)
                if negative_surpluses else 0.5,
                0.5, 1.5,
            )

            deadline_penalty = base_penalty * (1.0 + time_scale + surplus_scale)

            # Distribute penalty proportionally to negative surpluses
            total_negative_mag = sum(negative_surpluses)
            for i, step in enumerate(trajectory):
                surplus = step_surpluses[i]
                if surplus < 0 and total_negative_mag > 0:
                    proportion = abs(surplus) / total_negative_mag
                else:
                    proportion = 1.0 / len(trajectory)
                step["reward"] = step["original_reward"] + deadline_penalty * proportion

        else:
            deadline_met_count += 1
            time_saved = profiling_data.deadline - total_time
            base_bonus = 0.25 * avg_step_reward

            time_saved_scale = np.clip(
                time_saved / profiling_data.deadline, 0.0, 1.0
            )

            positive_surpluses = [s for s in step_surpluses if s > 0]
            surplus_bonus_scale = np.clip(
                (np.mean(positive_surpluses) / profiling_data.deadline)
                if positive_surpluses else 0.0,
                0.0, 1.0,
            )

            energy_scale = np.clip(
                avg_step_reward / (total_energy * 1000.0 + 100.0),
                0.0, 0.5,
            )

            total_bonus = base_bonus * (1.0 + time_saved_scale + surplus_bonus_scale + energy_scale)

            total_positive_mag = sum(positive_surpluses)
            for i, step in enumerate(trajectory):
                surplus = step_surpluses[i]
                if surplus > 0 and total_positive_mag > 0:
                    proportion = surplus / total_positive_mag
                else:
                    proportion = 0.1 / len(trajectory)
                step["reward"] = step["original_reward"] + total_bonus * proportion

        # ==========================================================
        # Clip rewards
        # ==========================================================
        for step in trajectory:
            step["reward"] = np.clip(step["reward"], -500.0, 50.0)

        # ==========================================================
        # Update Q-values
        # ==========================================================
        if not is_test:
            agent._update_trajectory_q_values(trajectory)
            agent.notify_episode_end(sum(step["reward"] for step in trajectory))

        # ==========================================================
        # Logging
        # ==========================================================
        modified_total_reward = sum(step["reward"] for step in trajectory)

        edge_energy.append(total_energy)
        completion_time.append(total_time)
        rewards.append(total_reward)
        cumulative_rewards.append(modified_total_reward)

    agent.save_qtables()

    return (
        np.mean(edge_energy),
        np.mean(completion_time),
        deadline_missed_count,
        episode_computation_time
    )