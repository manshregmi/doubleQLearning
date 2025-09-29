from a2c.actor_critic_agent import A2CAgent   # <-- your new A2C agent file
from profiling.profile import ProfilingData
import numpy as np


def run__a2c_simulation(profiling_data: ProfilingData, episodes=10000, max_steps=20):
    agent = A2CAgent(profiling_data)
    edge_energy = []
    completion_time = []
    bandwidth = profiling_data.bandwidth

    # Try loading previous tables if available
    agent.load_tables()

    for ep in range(episodes):
        total_edge_energy = 0.0
        total_completion_time = 0.0
        current_state = (bandwidth, 0, 0, None, 0.0, 0)  # (bandwidth, cloud_time, layer, prev_action, surplus, negative_surplus_count)

        for _ in range(max_steps):
            # A2C train returns: (action, reward, next_state, terminal, energy, completionTime)
            _, reward, next_state, terminal, energy, completion_time_s = agent.train(current_state)

            total_edge_energy += energy
            total_completion_time += (completion_time_s * 1000)  # s → ms
            current_state = next_state

            if terminal:
                bandwidth = next_state[0]
                break

        edge_energy.append(total_edge_energy)
        completion_time.append(total_completion_time)

    # Save the learned tables
    agent.save_tables()

    return np.mean(edge_energy), np.mean(completion_time)
