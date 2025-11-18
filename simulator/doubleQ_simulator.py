from model.doubleQ import DoubleQLearningAgent
from profiling.profile import ProfilingData
import numpy as np

def run_simulation(profiling_data: ProfilingData, episodes=10000, max_steps=20):
    agent = DoubleQLearningAgent(profiling_data, is_test=True)
    edge_energy = []
    completion_time = []
    rewards = []

    agent.load_qtables()

    for ep in range(episodes):
        total_energy, total_time, total_reward = 0.0, 0.0, 0.0

        # ✅ reset env state EVERY EPISODE
        bandwidth = profiling_data.bandwidth
        cloud_time = 0.0
        current_state = (bandwidth, cloud_time, 0, None, 0, 0)

        for step in range(max_steps):
            _, reward, next_state, terminal, energy, completionTime, new_bandwidth, _, _ = agent.train(current_state)

            total_energy += energy
            total_time += (completionTime * 1000)  # ms
            total_reward += reward
            current_state = next_state
            if terminal:
                bandwidth = new_bandwidth
                break

            
        edge_energy.append(total_energy)
        completion_time.append(total_time)
        rewards.append(total_reward)
        # print(f"Episode {ep}, Energy: {total_energy:.3f}, Time: {total_time:.3f}, Reward: {total_reward:.3f}")
    try:
        agent.save_qtables()
    except Exception as e:
        print(f"Error saving Q-tables: {e}")   
    E = np.array(edge_energy)
    T = np.array(completion_time)

    print(f"DQ Avg Energy: {E.mean():.3f} J, Std: {E.std():.3f}, Lower Bound: {E.min():.3f} J, Upper Bound: {E.max():.3f} J , Lowest index: {np.argmin(E)}, Reward at that index: {rewards[np.argmin(E)]:.3f} time at that index: {T[np.argmin(E)]:.3f} ms")
    print(f"DQ Avg Time: {T.mean():.3f} ms, Std: {T.std():.3f}, Lower Bound: {T.min():.3f} ms, Upper Bound: {T.max():.3f} ms, Lowest index: {np.argmin(T)}, Reward at that index: {rewards[np.argmin(T)]:.3f} energy at that index: {E[np.argmin(T)]:.3f} J")
    return E.mean(), T.mean()



def run_simulation_all(profiling_data: ProfilingData, episodes=10000):
    """
    Run full-task Double Q-learning simulation.
    Each episode = one full task execution.
    The agent predicts one full assignment plan,
    simulator executes it, and we get a single reward for the task.
    """
    agent = DoubleQLearningAgent(profiling_data, is_test=False)
    agent.load_qtables()

    all_energies = []
    all_times = []
    all_rewards = []
    bandwidth = profiling_data.bandwidth
    cloud_time = 0



    for ep in range(episodes):
        # Each episode starts from the same initial state (no progression)
        surplus = 0.0
        neg_count = 0
        prev_action = None

        # The "state" represents the global condition before the task starts
        current_state = (bandwidth, cloud_time)

        # Perform one training step (predict whole plan + update Q-tables)
        reward, total_energy, total_time, bandwidth, cloud_time = agent.train_all(current_state)
        # print('cloud waiting', cloud_time)

        all_energies.append(total_energy)
        all_times.append(total_time)
        all_rewards.append(reward)
        if (ep + 1) % 1000 == 0:
            print(f"Episode {ep}, Energy: {total_energy:.3f}, Time: {total_time:.3f}, Reward: {reward:.3f}")

    # Save Q-tables after training
    try:
        agent.save_qtables()
    except Exception as e:
        print(f"Error saving Q-tables: {e}")

    E = np.array(all_energies)
    T = np.array(all_times)
    R = np.array(all_rewards)

    print(f"DQ Avg Energy: {E.mean():.3f} J, Std: {E.std():.3f}, Lower Bound: {E.min():.3f} J, Upper Bound: {E.max():.3f} J , Lowest index: {np.argmin(E)}, Reward at that index: {R[np.argmin(E)]:.3f} time at that index: {T[np.argmin(E)]:.3f} ms")
    print(f"DQ Avg Time: {T.mean():.3f} ms, Std: {T.std():.3f}, Lower Bound: {T.min():.3f} ms, Upper Bound: {T.max():.3f} ms, Lowest index: {np.argmin(T)}, Reward at that index: {R[np.argmin(T)]:.3f} energy at that index: {E[np.argmin(T)]:.3f} J")
    return E.mean(), T.mean(), R.mean()
