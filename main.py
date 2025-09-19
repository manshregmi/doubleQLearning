from model.doubleQ import DoubleQLearningAgent
from profile.profile import ProfilingData
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(episodes=50000, max_steps=20):
    # Dummy profiling data setup
    layers = [
        [0],           # dummy layer 0 (input)
        [0, 1, 2],     # Layer 0 with 3 nodes
        [0, 1],        # Layer 1 with 2 nodes
        [0, 1, 2, 3]   # Layer 2 with 4 nodes
    ]

    numberOfEdgeDevice = 2

    # Edge computation times (ms)
    node_edge_times = {
        (0,0): 1,  # dummy node
        (0, 0): 10, (0, 1): 4, (0, 2): 2,
        (1, 0): 15, (1, 1): 10,
        (2, 0): 4, (2, 1): 5, (2, 2): 8, (2, 3): 10
    }

    # Cloud computation times (ms)
    node_cloud_times = {
        (0,0):0,                    # dummy node   
        (0, 0): 5, (0, 1): 2, (0, 2): 1,
        (1, 0): 5, (1, 1): 6,
        (2, 0): 2, (2, 1): 1, (2, 2): 4, (2, 3): 5
    }

    # Edge power consumption (Watts)
    node_edge_powers = {
        (0,0):1.0,                      # dummy node
        (0, 0): 4.0, (0, 1): 4.5, (0, 2): 4.2,
        (1, 0): 5.0, (1, 1): 5.2,
        (2, 0): 4.6, (2, 1): 4.3, (2, 2): 5.0, (2, 3): 5.3
    }

    bandwidth = 15.0         # Mbps
    rtt = 0.5                # ms
    output_size = 1          # KB
    edge_idle_power = 1.2    # Watts
    edge_communication_power = 2.0  # Watts
    deadlines = np.linspace(15, 130, num=100)  # ms


    energy_results = []
    completion_time_results = []
    reward_results = []

    for deadline in deadlines:
        # Create profiling data instance
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
            deadline=deadline,
            edge_communication_power=edge_communication_power
        )

        agent = DoubleQLearningAgent(profiling_data)

        total_reward = 0.0
        total_edge_energy = 0.0
        total_completion_time = 0.0
        total_steps = 0

        # Run multiple episodes
        for ep in range(episodes):
            # Reset environment at the start of each episode
            current_state = (bandwidth, 0, 0, None)  # <-- prev_action = None at start
            print(f"Episode {ep+1}/{episodes}")
            for step in range(max_steps):
                action, reward, next_state, terminal, energy, completionTime = agent.train(current_state)

                total_reward += reward
                total_edge_energy += energy
                total_completion_time += completionTime
                total_steps += 1

                current_state = next_state
                if terminal:
                    break

        print(f"Average Reward per step: {total_reward/total_steps:.2f}")
        print(f"Total Edge Energy Consumption: {total_edge_energy:.2f} Joules")
        print(f"Average Completion Time per step: {((total_completion_time/episodes) * 10**3):.2f} ms")

        energy_results.append(total_edge_energy)
        completion_time_results.append((total_completion_time/episodes) * 10**3)
        reward_results.append(total_reward/total_steps)

    print("Energy Results:", np.average(energy_results))
    print("Completion Time Results:", np.average(completion_time_results))
    print("Reward Results:", np.average(reward_results))

    # 1. Energy vs Deadline
    plt.figure(figsize=(6,4))
    plt.plot(energy_results, deadlines, marker='o')
    plt.xlabel("Energy (Joules)")
    plt.ylabel("Deadline (ms)")
    plt.title("Energy vs Deadline")
    plt.grid(True)
    plt.show()

    # 2. Reward vs Energy
    plt.figure(figsize=(6,4))
    plt.plot(energy_results, reward_results, marker='s')
    plt.xlabel("Energy (Joules)")
    plt.ylabel("Reward")
    plt.title("Reward vs Energy")
    plt.grid(True)
    plt.show()

    # 3. Deadline vs Reward
    plt.figure(figsize=(6,4))
    plt.plot(deadlines, reward_results, marker='^')
    plt.xlabel("Deadline (ms)")
    plt.ylabel("Reward")
    plt.title("Deadline vs Reward")
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    run_simulation()
