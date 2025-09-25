from model.doubleQ import DoubleQLearningAgent
from profiling.profile import ProfilingData
import numpy as np
import matplotlib.pyplot as plt
from random_scheduler import run_random_scheduler

def run_simulation(profiling_data : ProfilingData, episodes=100000, max_steps=20):
    agent = DoubleQLearningAgent(profiling_data)
    edge_energy = []
    completion_time = []
    bandwidth = profiling_data.bandwidth

    # Run episodes for current deadline
    for ep in range(episodes):
        total_edge_energy = 0.0
        total_completion_time = 0.0
        current_state = (bandwidth, 0, 0, None)
        for __ in range(max_steps):
            _, __, next_state, terminal, energy, completionTime, new_bandwidth = agent.train(current_state)
            total_edge_energy += energy
            total_completion_time += (completionTime*1000)  # convert to ms
            current_state = next_state
            if terminal:
                bandwidth = new_bandwidth
                break
        edge_energy.append(total_edge_energy)
        completion_time.append(total_completion_time)
        
        # avg_energy = total_edge_energy / episodes
        # avg_reward = total_reward / episodes
        # avg_completion = total_completion_time / total_steps if total_steps > 0 else 0

        # avg_energy_per_deadline.append(avg_energy)
        # avg_reward_per_deadline.append(avg_reward)
        # avg_completion_per_deadline.append(avg_completion)

    # Plot Energy vs Deadline
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(edge_energy)+1, 1), edge_energy, marker='o')
    plt.xlabel("Episodes")
    plt.ylabel("Edge Energy (Joules)")
    plt.title(f"Energy vs Episodes, {'average energy=' + str(np.mean(edge_energy)) if episodes > 0 else ''}")
    plt.grid(True)
    plt.show()

    # Plot Reward vs Energy
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(completion_time)+1, 1), completion_time, marker='s')
    plt.xlabel("Episodes")
    plt.ylabel("Completion Time (ms)")
    plt.title(f"Completion Time vs Episodes, {'average time=' + str(np.mean(completion_time)) if episodes > 0 else ''}")
    plt.grid(True)
    plt.show()

def get_profiling_data():
    # Dummy profiling data setup
    layers = [
        [0],           # dummy layer 0 (input)
        [0, 1, 2],     # Layer 0 with 3 nodes
        [0, 1],        # Layer 1 with 2 nodes
        [0, 1, 2, 3],  # Layer 2 with 4 nodes
        [0],           # dummy layer 3 (output)
    ]

    numberOfEdgeDevice = 2

    # Edge computation times (ms)
    node_edge_times = {
        (0,0): 1,  # dummy node (input parsing)
        (1, 0): 35, (1, 1): 28, (1, 2): 22,   # Conv / FC layers
        (2, 0): 45, (2, 1): 40,               # pooling/transformer layers
        (3, 0): 30, (3, 1): 25, (3, 2): 38, (3, 3): 42,  # deeper layers
        (4,0): 1    # dummy node (output)
    }

    # Cloud computation times (ms)
    node_cloud_times = {
        (0,0): 0,  # dummy
        (1, 0): 16, (1, 1): 12, (1, 2): 10,
        (2, 0): 20, (2, 1): 24,
        (3, 0): 12, (3, 1): 10, (3, 2): 14, (3, 3): 18,
        (4, 0): 0   # dummy
    }

    # Edge power consumption (Watts)
    node_edge_powers = {
      (0,0): 0.5,
      (1,0): 12.132, (1,1): 11.305, (1,2): 10.596,
      (2,0): 13.304, (2,1): 12.717,
      (3,0): 11.542, (3,1): 10.923, (3,2): 12.553, (3,3): 13.076,
      (4,0): 0.5
    }

    bandwidth = 15.0               # Mbps
    rtt = 10.0                     # ms (cellular-like network)
    output_size = 5               # KB
    edge_idle_power = 8.0          # Watts
    edge_communication_power = 10.0  # Watts

    deadlines = 290                 # ms (total for all layers)

    profiling_data = ProfilingData(
        numberOfEdgeDevice=numberOfEdgeDevice,
        layers=layers,
        node_edge_times=node_edge_times,
        node_cloud_times=node_cloud_times,
        bandwidth=bandwidth,
        rtt=rtt,
        output_size=output_size,
        node_edge_powers=node_edge_powers,
        edge_idle_power=edge_idle_power,
        deadline=deadlines,
        edge_communication_power=edge_communication_power
        )
            
    return profiling_data


if __name__ == "__main__":
    profiling_data = get_profiling_data()
    episodes = 10000
    max_steps = 20
    run_simulation(profiling_data=profiling_data, episodes=episodes, max_steps=max_steps)
    run_random_scheduler(profiling_data=profiling_data, episodes=episodes, max_steps=max_steps, is_random=True, is_all_cloud=False)
    run_random_scheduler(profiling_data=profiling_data, episodes=episodes, max_steps=max_steps, is_random=False, is_all_cloud=False)
    run_random_scheduler(profiling_data=profiling_data, episodes=episodes, max_steps=max_steps, is_random=False, is_all_cloud=True)