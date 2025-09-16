from model.doubleQ import DoubleQLearningAgent
from profile.profile import ProfilingData


def run_simulation(episodes=5, max_steps=20):
    # Dummy profiling data setup
    layers = [
        [0, 1, 2],     # Layer 0 with 3 nodes
        [0, 1],        # Layer 1 with 2 nodes
        [0, 1, 2, 3]   # Layer 2 with 4 nodes
    ]

    numberOfEdgeDevice = 2

    # Edge computation times (ms)
    node_edge_times = {
        (0, 0): 30, (0, 1): 35, (0, 2): 32,
        (1, 0): 40, (1, 1): 45,
        (2, 0): 35, (2, 1): 33, (2, 2): 38, (2, 3): 40
    }

    # Cloud computation times (ms)
    node_cloud_times = {
        (0, 0): 10, (0, 1): 12, (0, 2): 11,
        (1, 0): 15, (1, 1): 16,
        (2, 0): 12, (2, 1): 11, (2, 2): 14, (2, 3): 15
    }

    # Edge power consumption (Watts)
    node_edge_powers = {
        (0, 0): 4.0, (0, 1): 4.5, (0, 2): 4.2,
        (1, 0): 5.0, (1, 1): 5.2,
        (2, 0): 4.6, (2, 1): 4.3, (2, 2): 5.0, (2, 3): 5.3
    }

    bandwidth = 15.0         # Mbps
    rtt = 8.0                # ms
    output_size = 512        # KB
    edge_idle_power = 1.2    # Watts
    deadline = 12           # ms

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
        deadline=deadline
    )

    agent = DoubleQLearningAgent(profiling_data)

    # Run multiple episodes
    for ep in range(episodes):
        # Reset environment at the start of each episode
        current_state = [bandwidth, 0, 0, len(layers[0])]
        print(f"\n===== Episode {ep + 1} =====")

        for step in range(max_steps):
            action, reward, next_state, terminal = agent.train(current_state)

            print(f"Step {step + 1}:")
            print(f"  Current State: {current_state}")
            print(f"  Action (0=edge, 1=cloudlet):\n{action}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Next State: {next_state}\n")

            current_state = next_state
            if terminal:
                print("  >>> Episode finished (last layer reached)\n")
                break


if __name__ == "__main__":
    run_simulation()
