from model.doubleQ import DoubleQLearningAgent
from profiling.profile import ProfilingData
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

def run_simulation(profiling_data: ProfilingData, episodes=1, max_steps=20, is_test=False):
    agent = DoubleQLearningAgent(profiling_data, is_test=is_test)
    edge_energy = []
    completion_time = []
    rewards = []

    agent.load_qtables()

    total_episode_actions = []

    # ==========================================================
    # ✅ Counter for episode-level action sequence occurrences
    # ==========================================================
    action_sequence_counts = defaultdict(int)
    # ==========================================================

    timestamp = time.time()
    print(timestamp)

    for ep in range(episodes):
        total_energy, total_time, total_reward = 0.0, 0.0, 0.0

        cloud_time = 0.0
        current_state = (profiling_data.bandwidth, cloud_time, 0, None, 0, 0)
        overall_action = []

        for step in range(max_steps):
            action, reward, next_state, terminal, energy, completionTime, new_bandwidth, _, _ = agent.train(current_state)
            overall_action.append(tuple(map(tuple, action)))  # ensure hashable

            total_energy += energy
            total_time += (completionTime * 1000)  # ms
            total_reward += reward
            current_state = next_state

            if terminal:
                bandwidth = new_bandwidth
                break
            
        # Add the action sequence to the counter
        action_sequence_key = tuple(overall_action)
        action_sequence_counts[action_sequence_key] += 1

        edge_energy.append(total_energy)
        completion_time.append(total_time)
        total_episode_actions.append(overall_action)
        rewards.append(total_reward)
    print("Simulation Time:", time.time() - timestamp)

    if not is_test:    
        # Try saving Q-tables
        try:
            agent.save_qtables()
        except Exception as e:
            print(f"Error saving Q-tables: {e}")   

    E = np.array(edge_energy)
    T = np.array(completion_time)
    R = np.array(rewards)

    # print("----- Simulation Results -----")
    # print(f"DQ Avg Energy: {E.mean():.3f} J, Std: {E.std():.3f}, Lower Bound: {E.min():.3f} J, Upper Bound: {E.max():.3f} J , Lowest index: {np.argmin(E)}, Reward at that index: {R[np.argmin(E)]:.3f} time at that index: {T[np.argmin(E)]:.3f} ms")
    # print(f"DQ Avg Time: {T.mean():.3f} ms, Std: {T.std():.3f}, Lower Bound: {T.min():.3f} ms, Upper Bound: {T.max():.3f} ms, Lowest index: {np.argmin(T)}, Reward at that index: {R[np.argmin(T)]:.3f} energy at that index: {E[np.argmin(T)]:.3f} J")

    # ==================================================================
    #                 PRINT REPEATING ACTION SEQUENCES
    # ==================================================================
    # print("\n=== ACTION SEQUENCE REPETITION COUNT ===\n")

    # if len(action_sequence_counts) == 0:
    #         print("No episodes contained actions.")
    # else:
    #     sorted_sequences = sorted(action_sequence_counts.items(), key=lambda x: x[1], reverse=True)
    #     top_k = sorted_sequences[:5]

    #     for idx, (seq, count) in enumerate(top_k, start=1):
    #         print(f"#{idx}: Occurred {count} times")
    #         print("Sequence:")
    #         for step_action in seq:
    #             print(step_action)
    #         print("---")
    
    #     # --------------------------------------------------------------
    # #                HISTOGRAM OF ENERGY VALUES
    # # --------------------------------------------------------------
    # plt.figure(figsize=(8, 5))
    # plt.hist(edge_energy, bins=50)
    # plt.xlabel("Energy Consumption (Joules)")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Episode Energy Consumption")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.show()



    return E.mean(), T.mean()
