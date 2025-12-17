import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from a2c.sac_agent import SoftActorCriticAgent  # Assuming your SAC agent is saved here
from profiling.profile import ProfilingData
from typing import Tuple, List, Dict, Any

def run_sac_simulation(
    profiling_data: ProfilingData,
    episodes: int = 1,
    max_steps: int = 20,
    is_test: bool = False,
    load_model: bool = True,
    model_path: str = "sac_models.pth",
    collect_stats: bool = True,
    visualize: bool = False,
    training_config: Dict = None,
    BW_bins: int = 15,
    CT_bins: int = 20,
    surplus_bins: int = 25
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Run SAC simulation for computational offloading.
    
    Args:
        profiling_data: ProfilingData instance
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        is_test: Whether to run in test mode (no training)
        load_model: Whether to load pre-trained model
        model_path: Path to saved model
        collect_stats: Whether to collect detailed statistics
        visualize: Whether to show visualizations
        training_config: Dictionary with training hyperparameters (if not is_test)
        
    Returns:
        Tuple of (avg_energy, avg_time, statistics_dict)
    """
    
    # Default training configuration
    if training_config is None:
        training_config = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'hidden_dim': 256,
            'buffer_size': 100000,
            'batch_size': 256,
            'auto_entropy': True
        }
    
    # Initialize SAC agent
    agent = SoftActorCriticAgent(
        profiling_data=profiling_data,
        is_test=is_test,
        learning_rate=training_config['learning_rate'],
        gamma=training_config['gamma'],
        tau=training_config['tau'],
        alpha=training_config['alpha'],
        hidden_dim=training_config['hidden_dim'],
        buffer_size=training_config['buffer_size'],
        batch_size=training_config['batch_size'],
        auto_entropy=training_config['auto_entropy'],
        BW_bins=BW_bins,
        CT_bins=CT_bins,
        surplus_bins=surplus_bins
    )
    
    # Load pre-trained model if specified
    if load_model:
        agent.load_models(model_path)
    
    # Statistics collection
    edge_energy = []
    completion_time = []
    rewards = []
    entropy_values = []
    q_values = []
    
    # Action sequence tracking
    action_sequence_counts = defaultdict(int)
    total_episode_actions = []
    
    # Layer-wise statistics
    layer_decisions = defaultdict(lambda: defaultdict(int))
    
    # Training progress (if training)
    training_losses = {
        'actor_loss': [],
        'critic_loss': [],
        'alpha_loss': [],
        'alpha_value': []
    }
    
    timestamp = time.time()
    print(f"Starting SAC {'evaluation' if is_test else 'training'} simulation at {timestamp}")
    print(f"Running {episodes} episodes with {max_steps} max steps")
    
    for ep in range(episodes):
        episode_start_time = time.time()
        
        # Reset episode statistics
        total_energy, total_time, total_reward = 0.0, 0.0, 0.0
        episode_entropy = []
        episode_q_values = []
        
        # Initialize state
        cloud_time = 0.0
        current_state = (profiling_data.bandwidth, cloud_time, 0, None, 0, 0)
        overall_action = []
        
        step = 0
        terminal = False
        
        while step < max_steps and not terminal:
            # Get action and transition from agent
            action, reward, next_state, terminal, energy, completionTime, new_bandwidth, surplus, fractional_deadline = agent.train(current_state)
            
            # Store action sequence (make it hashable)
            overall_action.append(tuple(map(tuple, action)))
            
            # Track layer decisions
            layer_idx = int(current_state[2])
            action_key = tuple(action[:, 1].tolist())  # Binary offloading decisions
            layer_decisions[layer_idx][action_key] += 1
            
            # Accumulate statistics
            total_energy += energy
            total_time += (completionTime * 1000)  # Convert to ms
            total_reward += reward
            
            # Store entropy and Q-values (if available during training)
            if not is_test and hasattr(agent, 'get_current_entropy'):
                # You might want to add this method to your SAC agent
                episode_entropy.append(agent.get_current_entropy())
                episode_q_values.append(agent.get_current_q_value())
            
            # Update state
            current_state = next_state
            step += 1
        
        # Store episode results
        edge_energy.append(total_energy)
        completion_time.append(total_time)
        rewards.append(total_reward)
        total_episode_actions.append(overall_action)
        
        # Track entropy and Q-values
        if episode_entropy:
            entropy_values.append(np.mean(episode_entropy))
            q_values.append(np.mean(episode_q_values))
        
        # Count action sequence occurrences
        action_sequence_key = tuple(overall_action)
        action_sequence_counts[action_sequence_key] += 1
        
        # Episode summary
        episode_time = time.time() - episode_start_time
        if ep % max(1, episodes // 10) == 0:  # Print progress every 10%
            print(f"Episode {ep}/{episodes}: "
                  f"Energy={total_energy:.2f}J, "
                  f"Time={total_time:.2f}ms, "
                  f"Reward={total_reward:.2f}, "
                  f"Duration={episode_time:.2f}s")
        
        # Notify agent of episode end (for potential adaptive updates)
        if not is_test:
            agent.notify_episode_end(total_reward)
            
            # Collect training losses (you might need to expose these from agent)
            if hasattr(agent, 'get_last_losses'):
                losses = agent.get_last_losses()
                for key in training_losses:
                    if key in losses:
                        training_losses[key].append(losses[key])
    
    total_simulation_time = time.time() - timestamp
    print(f"\nTotal simulation time: {total_simulation_time:.2f} seconds")
    
    # Convert to numpy arrays for analysis
    E = np.array(edge_energy)
    T = np.array(completion_time)
    R = np.array(rewards)
    
    # Calculate statistics
    stats = {
        'energy_mean': E.mean(),
        'energy_std': E.std(),
        'energy_min': E.min(),
        'energy_max': E.max(),
        'energy_argmin': int(np.argmin(E)),
        'time_mean': T.mean(),
        'time_std': T.std(),
        'time_min': T.min(),
        'time_max': T.max(),
        'time_argmin': int(np.argmin(T)),
        'reward_mean': R.mean(),
        'reward_std': R.std(),
        'reward_min': R.min(),
        'reward_max': R.max(),
        'num_unique_sequences': len(action_sequence_counts),
        'most_common_sequence_count': max(action_sequence_counts.values()) if action_sequence_counts else 0,
        'simulation_time': total_simulation_time,
        'avg_episode_time': total_simulation_time / episodes
    }
    
    # Add training statistics if applicable
    if not is_test:
        stats.update({
            'avg_entropy': np.mean(entropy_values) if entropy_values else 0,
            'avg_q_value': np.mean(q_values) if q_values else 0,
            'training_step': agent.training_step,
            'buffer_size': len(agent.replay_buffer) if hasattr(agent, 'replay_buffer') else 0
        })
    
    # Print summary
    print("\n" + "="*60)
    print("SAC SIMULATION SUMMARY")
    print("="*60)
    print(f"Mode: {'TEST' if is_test else 'TRAINING'}")
    print(f"Episodes: {episodes}")
    print(f"Average Energy: {E.mean():.3f} ± {E.std():.3f} J")
    print(f"Energy Range: [{E.min():.3f}, {E.max():.3f}] J")
    print(f"Best Energy Episode: {np.argmin(E)} ({E.min():.3f} J)")
    print(f"Average Time: {T.mean():.3f} ± {T.std():.3f} ms")
    print(f"Time Range: [{T.min():.3f}, {T.max():.3f}] ms")
    print(f"Best Time Episode: {np.argmin(T)} ({T.min():.3f} ms)")
    print(f"Average Reward: {R.mean():.3f} ± {R.std():.3f}")
    print(f"Unique Action Sequences: {len(action_sequence_counts)}")
    print(f"Most Common Sequence Count: {max(action_sequence_counts.values()) if action_sequence_counts else 0}")
    print("="*60)
    
    # Visualization
    if visualize:
        visualize_results(
            edge_energy, completion_time, rewards,
            action_sequence_counts, layer_decisions,
            training_losses if not is_test else None
        )
    
    # Print top action sequences
    print("\nTOP 5 ACTION SEQUENCES:")
    if action_sequence_counts:
        sorted_sequences = sorted(action_sequence_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (seq, count) in enumerate(sorted_sequences[:5], 1):
            print(f"\n#{idx}: Occurred {count} times")
            for step_num, step_action in enumerate(seq):
                print(f"  Step {step_num}: {step_action}")
    else:
        print("No action sequences recorded.")
    
    # Save model if training
    if not is_test:
        try:
            agent.save_models(f"sac_model_ep{episodes}.pth")
            print(f"Model saved to sac_model_ep{episodes}.pth")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    return E.mean(), T.mean()

def visualize_results(
    edge_energy: List[float],
    completion_time: List[float],
    rewards: List[float],
    action_sequence_counts: Dict,
    layer_decisions: Dict,
    training_losses: Dict = None
):
    """Create visualizations for simulation results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. Energy histogram
    axes[0].hist(edge_energy, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Energy Consumption (Joules)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram of Episode Energy Consumption")
    axes[0].axvline(np.mean(edge_energy), color='r', linestyle='--', label=f'Mean: {np.mean(edge_energy):.2f}J')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Time histogram
    axes[1].hist(completion_time, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel("Completion Time (ms)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Histogram of Episode Completion Time")
    axes[1].axvline(np.mean(completion_time), color='r', linestyle='--', label=f'Mean: {np.mean(completion_time):.2f}ms')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Energy vs Time scatter
    axes[2].scatter(edge_energy, completion_time, alpha=0.6)
    axes[2].set_xlabel("Energy (J)")
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_title("Energy vs Completion Time Trade-off")
    axes[2].grid(True, alpha=0.3)
    
    # 4. Reward progression
    axes[3].plot(rewards, marker='.', linestyle='-', alpha=0.7)
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Total Reward")
    axes[3].set_title("Reward Progression")
    axes[3].grid(True, alpha=0.3)
    
    # 5. Action sequence distribution
    if action_sequence_counts:
        seq_counts = list(action_sequence_counts.values())
        axes[4].hist(seq_counts, bins=min(20, len(set(seq_counts))), edgecolor='black', alpha=0.7, color='green')
        axes[4].set_xlabel("Sequence Occurrence Count")
        axes[4].set_ylabel("Number of Sequences")
        axes[4].set_title("Action Sequence Frequency Distribution")
        axes[4].grid(True, alpha=0.3)
    
    # 6. Training losses or layer decisions
    if training_losses and training_losses['actor_loss']:
        # Plot training losses
        ax6 = axes[5]
        episodes_range = range(len(training_losses['actor_loss']))
        ax6.plot(episodes_range, training_losses['actor_loss'], label='Actor Loss', alpha=0.7)
        ax6.plot(episodes_range, training_losses['critic_loss'], label='Critic Loss', alpha=0.7)
        if training_losses['alpha_loss']:
            ax6.plot(episodes_range, training_losses['alpha_loss'], label='Alpha Loss', alpha=0.7)
        ax6.set_xlabel("Training Episode")
        ax6.set_ylabel("Loss Value")
        ax6.set_title("Training Loss Progression")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        # Plot layer decision diversity
        ax6 = axes[5]
        layers = sorted(layer_decisions.keys())
        diversities = [len(layer_decisions[layer]) for layer in layers]
        ax6.bar(layers, diversities, alpha=0.7, color='purple')
        ax6.set_xlabel("Layer Index")
        ax6.set_ylabel("Number of Unique Decisions")
        ax6.set_title("Decision Diversity per Layer")
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
