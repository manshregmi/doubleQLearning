import numpy as np
from collections import namedtuple
# Assuming the corrected A2CAgent is in 'a2c/actor_critic_agent.py'
from a2c.actor_critic_agent import A2CAgent
from profiling.profile import ProfilingData 

# --- PLACEHOLDER CLASSES (For environment consistency) ---

# Define the expected structure of the state (must match the agent's expected input)
State = namedtuple('State', ['bandwidth', 'cloud_time', 'layer', 'prev_action', 'surplus', 'negative_surplus_count'])
# --- A2C SIMULATION RUNNER ---
# --- A2C SIMULATION RUNNER ---

def run_a2c_simulation(profiling_data: ProfilingData, episodes=10000, max_steps=20):
    """
    Runs the Tabular Actor-Critic (A2C) training loop.
    A2C is an on-policy algorithm, meaning updates happen instantly after each step.
    
    NOTE: The A2CAgent is assumed to have a 'train' method that handles action 
    selection, environment interaction (using the simulator), reward calculation, 
    and policy/value updates in a single step for on-policy learning.
    """
    # 1. Initialization
    agent = A2CAgent(profiling_data)
    # The simulator object is initialized here, but the agent's internal 
    # 'train' method is responsible for using it for interaction.
    
    edge_energy, completion_time, rewards = [], [], []
    
    # Initial environment state parameters
    bandwidth = profiling_data.bandwidth
    cloud_time = 0.0
    
    # 2. Load Checkpoint (Tabular tables or Neural Net weights)
    # This call assumes the A2CAgent handles loading its necessary persistence data.
    agent.load_tables()
    
    print(f"Starting A2C simulation for {episodes} episodes...")

    for ep in range(episodes):
        total_reward = 0.0
        total_edge_energy = 0.0
        total_completion_time = 0.0
        
        # Initial state: (bandwidth, cloud_time, layer, prev_action, surplus, negative_surplus_count)
        current_state_tuple = State(bandwidth, 0, 0, None, 0.0, 0)

        for _ in range(max_steps):
            # 3. Step and Train (A2C is on-policy)
            # agent.train is expected to:
            # a) Select action based on current_state_tuple
            # b) Use the simulator to find next_state and compute reward, energy, time.
            # c) Perform the policy and value updates.
            _, reward, next_state_tuple, terminal, energy, completion_time_s = agent.train(current_state_tuple)

            # 4. Update tracking variables
            total_edge_energy += energy
            total_completion_time += (completion_time_s * 1000)  # s → ms
            total_reward += reward
            current_state_tuple = next_state_tuple

            if terminal:
                bandwidth = next_state_tuple[0]  
                break
        try:
            agent.save_tables()
        except Exception as e:
            print(f"Error saving A2C tables at episode {ep}: {e}")
        print(f"Episode {ep}, Energy: {total_edge_energy:.3f}, Time: {total_completion_time:.3f}, Reward: {total_reward:.3f}")
        
        # ✅ store episode stats
        edge_energy.append(total_edge_energy)
        completion_time.append(total_completion_time)
        rewards.append(total_reward)

    E = np.array(edge_energy)
    T = np.array(completion_time)

    print(f"A2C Avg Energy: {E.mean():.3f} J, Std: {E.std():.3f}")
    print(f"A2C Avg Time: {T.mean():.3f} ms, Std: {T.std():.3f}")
    return E.mean(), T.mean()
