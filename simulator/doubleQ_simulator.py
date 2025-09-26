from model.doubleQ import DoubleQLearningAgent
from profiling.profile import ProfilingData
import numpy as np

def run_simulation(profiling_data: ProfilingData, episodes=10000, max_steps=20):
    agent = DoubleQLearningAgent(profiling_data)
    edge_energy = []
    completion_time = []
    bandwidth = profiling_data.bandwidth
    agent.load_qtables()
    for ep in range(episodes):
        total_edge_energy = 0.0
        total_completion_time = 0.0
        current_state = (bandwidth, 0, 0, None, 0)

        for __ in range(max_steps):
            _, ___, next_state, terminal, energy, completionTime, new_bandwidth = agent.train(current_state)
            total_edge_energy += energy
            total_completion_time += (completionTime * 1000)  # ms
            current_state = next_state
            if terminal:
                bandwidth = new_bandwidth
                break

        edge_energy.append(total_edge_energy)
        completion_time.append(total_completion_time)

    agent.save_qtables()
    return np.mean(edge_energy), np.mean(completion_time)

