from model.doubleQ import DoubleQLearningAgent
from profiling.profile import ProfilingData
import numpy as np

def run_simulation(profiling_data: ProfilingData, episodes=10000, max_steps=20):
    agent = DoubleQLearningAgent(profiling_data)
    edge_energy = []
    completion_time = []
    rewards = []
    bandwidth = profiling_data.bandwidth
    cloud_time = 0.0
    agent.load_qtables()
    for ep in range(episodes):
        total_edge_energy = 0.0
        total_completion_time = 0.0
        total_reward = 0.0
        current_state = (bandwidth, cloud_time, 0, None, 0, 0) # (bandwidth, cloud_time, layer, prev_action, surplus, negativesurpluscount)

        for __ in range(max_steps):
            _, reward, next_state, terminal, energy, completionTime, new_bandwidth = agent.train(current_state)
            total_edge_energy += energy
            total_completion_time += (completionTime * 1000)  # ms
            total_reward += reward  
            current_state = next_state
            if terminal:
                bandwidth = new_bandwidth
                cloud_time = next_state[1]
                break
        print(f"[Ep {ep}] Reward={total_reward:.3f}, Energy={total_edge_energy:.3f}, Time={total_completion_time:.3f}")
        rewards.append(total_reward)

        edge_energy.append(total_edge_energy)
        completion_time.append(total_completion_time)

    agent.save_qtables()
    return np.mean(edge_energy), np.mean(completion_time)

