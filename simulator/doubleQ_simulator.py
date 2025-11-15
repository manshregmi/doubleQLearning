from model.doubleQ import DoubleQLearningAgent
from profiling.profile import ProfilingData
import numpy as np

def run_simulation(profiling_data: ProfilingData, episodes=1, max_steps=20):
    agent = DoubleQLearningAgent(profiling_data, is_test=True)
    edge_energy = []
    completion_time = []
    rewards = []
    number_of_layers = len(profiling_data.layers)

    agent.load_qtables()

    for ep in range(episodes):
        total_energy, total_time, total_reward = 0.0, 0.0, 0.0

        # ✅ reset env state EVERY EPISODE
        bandwidth = profiling_data.bandwidth
        cloud_time = 0.0
        current_state = (bandwidth, cloud_time, 0, None, 0, 0)
        overall_action = []
        percetage_of_optimum = []
        optimumCount = 0


        for step in range(max_steps):
            action, reward, next_state, terminal, energy, completionTime, new_bandwidth, _, _ = agent.train(current_state)
            overall_action.append(action)
            total_energy += energy
            total_time += (completionTime * 1000)  # ms
            total_reward += reward
            current_state = next_state
            if terminal:
                bandwidth = new_bandwidth
                break
        all_match = True
        
        optimum_actions = profiling_data.get_optimum_action_array()
        resemblance = 0
        for i, (act, opt) in enumerate(zip(overall_action, optimum_actions)):
            if not np.array_equal(act, opt):
                all_match = False
            else:
                resemblance += 1/6
        percetage_of_optimum.append(resemblance)
        if all_match:
            optimumCount += 1
        edge_energy.append(total_energy)
        completion_time.append(total_time)
        rewards.append(total_reward)
        if (ep + 1) % 1000 == 0:
            print(f"Episode {ep}, Energy: {total_energy:.3f}, Time: {total_time:.3f}, Reward: {total_reward:.3f}")
    try:
        agent.save_qtables()
    except Exception as e:
        print(f"Error saving Q-tables: {e}")   
    E = np.array(edge_energy)
    T = np.array(completion_time)
    print("----- Simulation Results -----")
    print("optimum action counts per layer:", agent.optimum_action_layer_count)
    print("last layer Not optimum count:", agent.last_layer_not_optimum)
    print("Total optimum actions selected:", optimumCount, "out of", episodes)
    print("Percentage of maximum resemblance :", np.max(percetage_of_optimum)*100 , "%")

    print(f"DQ Avg Energy: {E.mean():.3f} J, Std: {E.std():.3f}, Lower Bound: {E.min():.3f} J, Upper Bound: {E.max():.3f} J , Lowest index: {np.argmin(E)}, Reward at that index: {rewards[np.argmin(E)]:.3f} time at that index: {T[np.argmin(E)]:.3f} ms")
    print(f"DQ Avg Time: {T.mean():.3f} ms, Std: {T.std():.3f}, Lower Bound: {T.min():.3f} ms, Upper Bound: {T.max():.3f} ms, Lowest index: {np.argmin(T)}, Reward at that index: {rewards[np.argmin(T)]:.3f} energy at that index: {E[np.argmin(T)]:.3f} J")
    return E.mean(), T.mean()

