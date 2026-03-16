import numpy as np
import time
from collections import defaultdict
import torch
from profiling.initialize_profiling import get_profiling_data
from simulator.simulator import CloudEdgeSimulator
from a2c.ppo_agent import PPOAgent
import os
 
def run_ppo_simulation(
    profiling_data: get_profiling_data,
    episodes=1000,
    max_steps=20,
    is_test=False,
    rollout_length=2048,      # steps before update
    visualize_stats=True,
    model_path="ppo_policy.pth",   # path to save/load model
):
    """
    Run PPO training/evaluation on the cloud‑edge simulator.
    """
    agent = PPOAgent(profiling_data)
 
    # Load existing model if requested (useful for testing or continuing training)
    if is_test or os.path.exists(model_path):
        # During testing we don't need the optimizer; during training we load it to resume.
        agent.load(model_path, load_optimizer=(not is_test))
 
    # For statistics
    edge_energy = []
    completion_time = []
    deadline_missed_count = 0
    deadline_met_count = 0
    layer_violation_stats = defaultdict(int)
    episode_rewards = []
    episode_modified_rewards = []
    overhead_time = []
 
    if visualize_stats:
        edge_execution_stats = defaultdict(int)
        cloud_execution_stats = defaultdict(int)
 
    start_time = time.time()
    print(f"Starting PPO simulation at: {start_time}")
    bandwidth = profiling_data.bandwidth
 
    for ep in range(episodes):
        # Reset environment
        cloud_time = 0.0
        current_state = (bandwidth, cloud_time, 0, None, 0, 0)
        terminal = False
 
        # Temporary storage for one episode (to apply reward shaping)
        episode_steps = []
 
        total_energy = 0.0
        total_time = 0.0
        original_total_reward = 0.0
 
        step = 0
        while not terminal and step < max_steps:
            # Choose action (returns list of 0/1 for existing nodes)
            overhead_time_start = time.time()
            action_list, log_prob, value = agent.choose_action(current_state)
            overhead_time_end = time.time()

            overhead_time.append(overhead_time_end - overhead_time_start)
            

            # Build the full action matrix required by the simulator
            layer = int(current_state[2])
            num_nodes = len(profiling_data.layers[layer])
            action_matrix = np.zeros((num_nodes, 2), dtype=int)
            action_matrix[:, 0] = layer
            for i, loc in enumerate(action_list):
                action_matrix[i, 1] = loc
 
            # Simulate one step using the action_matrix
            next_cloud = CloudEdgeSimulator(profiling_data).get_next_state_cloud_waiting_time(
                next_layer=min(layer + 1, len(profiling_data.layers) - 1),
                current_action=action_matrix,
                isAllCloud=False,
            )
 
            energy, comp_time_s = CloudEdgeSimulator(profiling_data).compute_energy_and_time(
                current_state=current_state,
                current_action=action_matrix,
                cloud_pending_ms=next_cloud,
            )
 
            reward, surplus, neg_count, fractional_deadline = CloudEdgeSimulator(profiling_data).calculate_reward(
                layer,
                energy,
                comp_time_s,
                current_state[4],  # previous surplus
                current_state[5],  # negative count
                isA2C=True,
            )
 
            next_state, terminal, _ = CloudEdgeSimulator(profiling_data).get_next_state(
                current_state,
                action_matrix,
                surplus,
                neg_count,
                new_cloud_pending=next_cloud,
            )
 
            # Accumulate totals for episode statistics
            total_energy += energy
            total_time += comp_time_s * 1000.0   # convert to ms
            original_total_reward += reward
 
            # Store step data for later reshaping
            episode_steps.append({
                "state": current_state,
                "action": action_list,           # store the list, not the matrix
                "log_prob": log_prob,
                "value": value,
                "original_reward": reward,
                "surplus": surplus,
                "neg_increased": neg_count > current_state[5],
                "layer": layer,
            })
 
            # Update current state
            current_state = next_state
            step += 1
 
        # End of episode: compute deadline performance and reshape rewards
        deadline_violated = total_time > profiling_data.deadline
        if deadline_violated:
            deadline_missed_count += 1
        else:
            deadline_met_count += 1
 
        # Average step reward magnitude (as in A2C runner)
        avg_step_reward = np.clip(
            np.mean([abs(s["original_reward"]) for s in episode_steps]),
            300.0, 3000.0
        )
 
        # Reshape rewards based on final deadline
        if deadline_violated:
            excess = total_time - profiling_data.deadline
            base_penalty = -0.6 * avg_step_reward
            scale = np.clip(excess / profiling_data.deadline, 0.0, 1.5)
            penalty = base_penalty * (1.0 + scale)
            for step_dict in episode_steps:
                step_dict["reshaped_reward"] = step_dict["original_reward"] + penalty / len(episode_steps)
        else:
            saved = profiling_data.deadline - total_time
            bonus = 0.25 * avg_step_reward * (1.0 + saved / profiling_data.deadline)
            for step_dict in episode_steps:
                step_dict["reshaped_reward"] = step_dict["original_reward"] + bonus / len(episode_steps)
 
        # Clip reshaped rewards as in A2C
        for step_dict in episode_steps:
            step_dict["reshaped_reward"] = np.clip(step_dict["reshaped_reward"], -500.0, 50.0)
 
        # Store transitions into agent's buffer (with reshaped rewards)
        for i, step_dict in enumerate(episode_steps):
            done = (i == len(episode_steps) - 1)  # last step of episode
            agent.store_transition(
                state=step_dict["state"],
                action=step_dict["action"],
                log_prob=step_dict["log_prob"],
                value=step_dict["value"],
                reward=step_dict["reshaped_reward"],
                done=done
            )
 
        # Episode statistics
        modified_episode_reward = sum(s["reshaped_reward"] for s in episode_steps)
        episode_rewards.append(original_total_reward)
        episode_modified_rewards.append(modified_episode_reward)
        edge_energy.append(total_energy)
        completion_time.append(total_time)
 
        # Tracking for visualisation
        if visualize_stats:
            for step_dict in episode_steps:
                layer = step_dict["layer"]
                action = step_dict["action"]
                num_nodes = len(profiling_data.layers[layer])
                for node_idx, loc in enumerate(action):
                    key = (layer, node_idx)
                    if loc == 0:
                        edge_execution_stats[key] += 1
                    else:
                        cloud_execution_stats[key] += 1
 
        # PPO update when buffer is full (only during training)
        if not is_test and len(agent.states) >= rollout_length:
            agent.update()
            print(f"Update performed after episode {ep+1}")
 
        # Optional: print progress
        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(episode_modified_rewards[-100:])
            print(f"Episode {ep+1} | Avg modified reward (last 100): {avg_reward:.2f} | "
                  f"Deadline met: {deadline_met_count}/{ep+1} ({deadline_met_count/(ep+1)*100:.1f}%)")
    

    print(f"\nTotal overhead time for action selection: {sum(overhead_time):.2f}s over {episodes} episodes, average {np.mean(overhead_time)*1000:.4f} ms per step, standard deviation {np.std(overhead_time)*1000:.4f} ms, removing warmup (first 100 steps) average {np.mean(overhead_time[100:])*1000:.4f} ms per step")

    # End of simulation – save the model if training
    if not is_test:
        agent.save(model_path)
 
    print(f"\nSimulation finished in {time.time() - start_time:.2f}s")
    print(f"Average energy: {np.mean(edge_energy):.4f} J")
    print(f"Average completion time: {np.mean(completion_time):.2f} ms")
    print(f"Deadline met: {deadline_met_count}/{episodes} ({deadline_met_count/episodes*100:.1f}%)")
 
 
    return edge_energy, completion_time,deadline_missed_count
 
    # return agent, {
    #     "energy": edge_energy,
    #     "time": completion_time,
    #     "deadline_met": deadline_met_count,
    #     "episode_rewards": episode_rewards,
    #     "episode_modified_rewards": episode_modified_rewards,
    #     "edge_stats": edge_execution_stats if visualize_stats else None,
    #     "cloud_stats": cloud_execution_stats if visualize_stats else None,
    # }
 
 
# Example usage:
if __name__ == "__main__":
    from profiling.initialize_profiling import get_profiling_data
 
    profiling = get_profiling_data(deadline=500, edge_devices=8)
    agent, stats = run_ppo_simulation(
        profiling,
        episodes=10000000,
        is_test=False,           
        rollout_length=2048,
        model_path="ppo_policy.pth"
    )