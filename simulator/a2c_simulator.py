from matplotlib.patches import Patch
from a2c.actor_critic_agent import TabularActorCriticAgent
from profiling.profile import ProfilingData
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def run_a2c_simulation(
    profiling_data: ProfilingData,
    episodes=1,
    max_steps=20,
    is_test=False,
    visualize_stats=False,
    plot_rewards=False,
    smoothing_window=50,
):
    agent = TabularActorCriticAgent(profiling_data, is_test=is_test)
    agent.load()

    edge_energy = []
    completion_time = []
    rewards = []
    cumulative_rewards = []
    
    # Track rewards per episode for plotting
    episode_rewards = []
    episode_modified_rewards = []

    deadline_missed_count = 0
    deadline_met_count = 0
    layer_violation_stats = defaultdict(int)
    
    if visualize_stats:
        edge_execution_stats = defaultdict(int)
        cloud_execution_stats = defaultdict(int)

    start_time = time.time()
    print(f"Starting A2C simulation at: {start_time}")
    bandwidth = profiling_data.bandwidth

    for ep in range(episodes):
        total_energy = 0.0
        total_time = 0.0
        total_reward = 0.0

        cloud_time = 0.0
        current_state = (bandwidth, cloud_time, 0, None, 0, 0)

        trajectory = []
        step_surpluses = []

        for step in range(max_steps):
            state_key = agent._state_to_key(current_state)

            (
                action,
                reward,
                next_state,
                terminal,
                energy,
                completion_time_s,
                new_bandwidth,
                surplus,
                fractional_deadline,
                neg_count,
            ) = agent.train(current_state)
            
            if visualize_stats:
                current_layer = int(current_state[2])
                agent.track_action_execution(action, current_layer)
                for node_idx, (_, location) in enumerate(action):
                    key = (current_layer, node_idx)
                    if location == 0:
                        edge_execution_stats[key] += 1
                    else:
                        cloud_execution_stats[key] += 1

            bandwidth = next_state[0]

            prev_neg = current_state[5]
            neg_increased = neg_count > prev_neg

            trajectory.append({
                "state_key": state_key,
                "action_key": agent._action_to_key(action),
                "original_reward": reward,
                "reward": reward,
                "surplus": surplus,
                "negative_surplus_increased": neg_increased,
            })

            if neg_increased:
                layer_violation_stats[int(current_state[2])] += 1

            step_surpluses.append(surplus)

            total_energy += energy
            total_time += completion_time_s * 1000.0
            total_reward += reward

            current_state = next_state

            if terminal:
                break

        # Deadline reward reshaping
        avg_step_reward = np.clip(
            np.mean([abs(s["original_reward"]) for s in trajectory]),
            300.0, 3000.0
        )

        deadline_violated = total_time > profiling_data.deadline

        if deadline_violated:
            deadline_missed_count += 1
            excess = total_time - profiling_data.deadline
            base_penalty = -0.6 * avg_step_reward
            scale = np.clip(excess / profiling_data.deadline, 0.0, 1.5)
            penalty = base_penalty * (1.0 + scale)

            for step in trajectory:
                step["reward"] += penalty / len(trajectory)
        else:
            deadline_met_count += 1
            saved = profiling_data.deadline - total_time
            bonus = 0.25 * avg_step_reward * (1.0 + saved / profiling_data.deadline)

            for step in trajectory:
                step["reward"] += bonus / len(trajectory)

        for step in trajectory:
            step["reward"] = np.clip(step["reward"], -500.0, 50.0)

        if not is_test:
            agent.update_trajectory(trajectory)
            agent.notify_episode_end(sum(step["reward"] for step in trajectory))

        modified_reward = sum(step["reward"] for step in trajectory)

        edge_energy.append(total_energy)
        completion_time.append(total_time)
        rewards.append(total_reward)
        cumulative_rewards.append(modified_reward)
        
        # Store episode rewards for plotting
        episode_rewards.append(total_reward)
        episode_modified_rewards.append(modified_reward)

    # Plot rewards if requested
    if plot_rewards and episodes > 1:
        R = np.array(episode_modified_rewards)
        smoothed = np.convolve(R, np.ones(50)/50, mode='valid')
        
        plt.figure(figsize=(8,5))
        plt.plot(smoothed, color='#0066CC', linewidth=2)
        plt.xlabel("Episodes")
        plt.ylabel("Smoothed Reward")
        plt.title("Actor-Critic (A2C) Convergence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        pdf_path = f"a2c_convergence.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"A2C convergence plot saved to: {pdf_path}")

    agent.save()
    
    # Return episode data for further analysis if needed
    return (
        np.mean(edge_energy),
        np.mean(completion_time),
        deadline_missed_count,
    )

def plot_smooth_reward_curve_paper_quality(rewards, sigma=2.0):
    """
    Create a publication-quality smoothed reward curve.
    sigma: controls the smoothing amount (higher = smoother)
    """
    episodes = np.arange(1, len(rewards) + 1)
    
    # Apply Gaussian smoothing (better than moving average)
    smoothed_rewards = gaussian_filter1d(rewards, sigma=sigma)
    
    # Create figure with publication quality
    plt.figure(figsize=(8, 4.5), dpi=300)  # Higher DPI for publication
    
    # Plot with gradient fill under the curve
    plt.plot(episodes, smoothed_rewards, 
             color='#2E86AB',  # Professional blue color
             linewidth=2.5,
             alpha=0.9,
             label='Smoothed Reward')
    # Fill under the curve with gradient
    plt.fill_between(episodes, smoothed_rewards, 
                     alpha=0.15, 
                     color='#2E86AB',
                     linewidth=0)
    
    # Style like Nature/Science papers
    plt.xlabel('Episode', fontsize=12, fontweight='medium', labelpad=10)
    plt.ylabel('Reward', fontsize=12, fontweight='medium', labelpad=10)
    
    # Add grid (subtle)
    plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add legend (simple, clean)
    plt.legend(frameon=False, fontsize=11, loc='lower right')
    
    # Tight layout for clean appearance

    plt.tight_layout()
    plt.show()





def plot_reward_curve_paper_style(rewards):
    """
    Plot the reward curve exactly like Figure 7 in the paper.
    Simple, clean, single line showing reward convergence.
    """
    episodes = np.arange(1, len(rewards) + 1)
    
    # Create figure - exactly like academic paper figure
    plt.figure(figsize=(8, 4))
    
    # Plot raw reward curve (no smoothing)
    plt.plot(episodes, rewards, 
             color='black', 
             linewidth=1.5)
    
    # Style exactly like paper
    plt.xlabel('Episode', fontsize=11)
    plt.ylabel('Reward', fontsize=11)
    
    # Remove title (papers often don't have titles on figures)
    # plt.title('Reward Convergence', fontsize=12, fontweight='bold')
    
    # Add subtle grid
    plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Make sure axis labels are visible and in frame
    plt.tight_layout()
    plt.show()


def plot_paper_style_comparison(rewards_list, labels=None, smoothing_window=50, deadline_ms=500):
    """
    Create a comparison plot similar to Figure 7 in the paper.
    Shows smoothed reward convergence for different methods/algorithms.
    
    Parameters:
    - rewards_list: List of reward lists for different methods
    - labels: List of labels for each method
    - smoothing_window: Smoothing window size
    - deadline_ms: Deadline constraint (for title/context)
    """
    
    if labels is None:
        labels = [f'Method {i+1}' for i in range(len(rewards_list))]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Set clean background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    for i, rewards in enumerate(rewards_list):
        episodes = np.arange(1, len(rewards) + 1)
        
        # Apply smoothing
        if len(rewards) >= smoothing_window:
            smoothed = pd.Series(rewards).rolling(
                window=smoothing_window, center=True, min_periods=1
            ).mean().values
        else:
            smoothed = rewards
        
        # Plot smoothed curve
        color = colors[i % len(colors)]
        ax.plot(episodes, smoothed, 
               color=color, 
               linewidth=2,
               label=labels[i])
    
    # Customize plot (academic paper style)
    ax.set_xlabel('Episode', fontsize=11, fontweight='medium')
    ax.set_ylabel('Smoothed Reward', fontsize=11, fontweight='medium')
    ax.set_title(f'RL Agent Convergence Comparison ({deadline_ms}ms Deadline)', 
                fontsize=12, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white')
    
    # Add smoothing note
    ax.text(0.98, 0.02, f'Moving average (window={smoothing_window})', 
            transform=ax.transAxes, fontsize=9, 
            ha='right', va='bottom', color='gray', alpha=0.7)
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.show()


# Alternative: Simple single-line plot (most similar to paper Figure 7)
def plot_simple_smoothed_reward(rewards, smoothing_window=1, color='black', linewidth=2):
    """
    Minimalist plot showing only smoothed reward curve.
    """
    episodes = np.arange(1, len(rewards) + 1)
    
    # Smooth the rewards
    smoothed = pd.Series(rewards).rolling(
        window=smoothing_window, center=True, min_periods=1
    ).mean()
    
    # Create plot
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, smoothed, color=color, linewidth=linewidth)
    
    # Style like academic figure
    plt.xlabel('Episode', fontsize=11)
    plt.ylabel('Reward', fontsize=11)
    plt.title('Actor-Critic Agent: Reward Curve', fontsize=12, fontweight='bold')
    
    plt.grid(True, alpha=0.2, linestyle='-')
    plt.tight_layout()
    plt.show()
    



def plot_reward_curves(original_rewards, modified_rewards, energy, times, deadline_status, 
                      deadline_constraint, plot_type='combined', smoothing_window=50):
    """
    Plot reward progression across episodes.
    
    Parameters:
    - original_rewards: List of original rewards per episode
    - modified_rewards: List of modified (shaped) rewards per episode
    - energy: List of energy consumption per episode (J)
    - times: List of completion times per episode (ms)
    - deadline_status: List of booleans indicating deadline status
    - deadline_constraint: The deadline constraint in ms
    - plot_type: 'combined', 'separate', or 'smoothed'
    - smoothing_window: Window size for moving average smoothing
    """
    
    episodes = np.arange(1, len(original_rewards) + 1)
    
    if plot_type == 'separate':
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Actor-Critic Agent Learning Progress Across Episodes', fontsize=16, fontweight='bold')
        
        # Plot 1: Original vs Modified Rewards
        axes[0, 0].plot(episodes, original_rewards, 'b-', alpha=0.6, linewidth=1.5, label='Original Reward')
        axes[0, 0].plot(episodes, modified_rewards, 'r-', alpha=0.8, linewidth=1.5, label='Modified Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Reward Progression')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Energy Consumption
        axes[0, 1].plot(episodes, energy, 'g-', linewidth=1.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Energy (J)')
        axes[0, 1].set_title('Energy Consumption per Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Completion Time with Deadline
        axes[1, 0].plot(episodes, times, 'm-', linewidth=1.5, label='Completion Time')
        axes[1, 0].axhline(y=deadline_constraint, color='r', linestyle='--', linewidth=2, label=f'Deadline ({deadline_constraint}ms)')
        # Color points based on deadline status
        for ep, time_val, met in zip(episodes, times, deadline_status):
            color = 'green' if met else 'red'
            axes[1, 0].scatter(ep, time_val, color=color, s=20, alpha=0.6)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Completion Time vs Deadline')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Moving average of modified reward
        if len(modified_rewards) >= smoothing_window:
            smoothed_rewards = pd.Series(modified_rewards).rolling(window=smoothing_window, center=True).mean()
            axes[1, 1].plot(episodes, modified_rewards, 'r-', alpha=0.3, linewidth=0.8, label='Raw')
            axes[1, 1].plot(episodes, smoothed_rewards, 'k-', linewidth=2, label=f'Moving Avg (window={smoothing_window})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].set_title('Smoothed Reward Progression')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].plot(episodes, modified_rewards, 'r-', linewidth=1.5)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].set_title('Modified Reward Progression')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    elif plot_type == 'smoothed':
        # Single plot with smoothed rewards
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if len(modified_rewards) >= smoothing_window:
            smoothed_rewards = pd.Series(modified_rewards).rolling(window=smoothing_window, center=True).mean()
            ax.plot(episodes, modified_rewards, 'r-', alpha=0.3, linewidth=0.8, label='Raw Modified Reward')
            ax.plot(episodes, smoothed_rewards, 'k-', linewidth=2.5, 
                   label=f'Smoothed Modified Reward (window={smoothing_window})')
        else:
            ax.plot(episodes, modified_rewards, 'r-', linewidth=1.5, label='Modified Reward')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Actor-Critic Agent: Smoothed Reward Progression', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add deadline performance annotation
        deadline_met_rate = sum(deadline_status) / len(deadline_status) * 100
        ax.text(0.02, 0.98, f'Deadline Met: {deadline_met_rate:.1f}%', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    else:  # 'combined' - default
        # Combined plot with rewards and deadline status
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Top plot: Rewards
        ax1.plot(episodes, original_rewards, 'b-', alpha=0.6, linewidth=1.5, label='Original Reward')
        ax1.plot(episodes, modified_rewards, 'r-', alpha=0.8, linewidth=1.5, label='Modified Reward')
        
        # Add moving average if enough episodes
        if len(modified_rewards) >= smoothing_window:
            smoothed = pd.Series(modified_rewards).rolling(window=smoothing_window, center=True).mean()
            ax1.plot(episodes, smoothed, 'k--', linewidth=2, label=f'Moving Avg (window={smoothing_window})')
        
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Actor-Critic Agent: Reward Progression Across Episodes', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Completion time with deadline status
        colors = ['green' if met else 'red' for met in deadline_status]
        ax2.scatter(episodes, times, c=colors, s=30, alpha=0.7, label='Completion Time')
        ax2.axhline(y=deadline_constraint, color='r', linestyle='--', linewidth=2, label=f'Deadline ({deadline_constraint}ms)')
        
        # Add a line connecting the points
        ax2.plot(episodes, times, 'gray', alpha=0.3, linewidth=0.5)
        
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Completion Time (ms)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics annotation
        deadline_met = sum(deadline_status)
        deadline_missed = len(deadline_status) - deadline_met
        avg_energy = np.mean(energy)
        avg_time = np.mean(times)
        
        stats_text = f'Episodes: {len(episodes)}\nDeadline Met: {deadline_met} ({deadline_met/len(episodes)*100:.1f}%)\nAvg Energy: {avg_energy:.2f} J\nAvg Time: {avg_time:.1f} ms'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
    


# ==================================================
# Color-blind safe palette (Okabe–Ito inspired)
# ==================================================
EDGE_COLOR = '#0072B2'     # Blue
CLOUD_COLOR = '#E69F00'    # Orange
BALANCED_COLOR = '#999999'


def create_node_execution_visualizations(edge_stats, cloud_stats, profiling_data, total_episodes):
    """
    Create comprehensive visualizations of node execution statistics
    using color-blind safe colors only.
    """

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------
    nodes_data = []
    edge_counts = []
    cloud_counts = []
    edge_percentages = []
    cloud_percentages = []

    layer_names = [
        'v1', 'v2', 'v3',
        ['v4', 'v7', 'v10'],
        ['v5', 'v8', 'v11'],
        ['v6', 'v9', 'v12'],
        'v13'
    ]

    for layer_idx, layer_nodes in enumerate(profiling_data.layers):
        for node_idx in range(len(layer_nodes)):
            node_key = (layer_idx, node_idx)

            edge_count = edge_stats.get(node_key, 0)
            cloud_count = cloud_stats.get(node_key, 0)
            total_count = edge_count + cloud_count

            if total_count > 0:
                edge_pct = (edge_count / total_count) * 100
                cloud_pct = (cloud_count / total_count) * 100
            else:
                edge_pct = cloud_pct = 0

            if len(layer_nodes) == 1:
                node_label = f"{layer_names[layer_idx]}"
            else:
                node_label = layer_names[layer_idx][node_idx]

            nodes_data.append(node_label)
            edge_counts.append(edge_count)
            cloud_counts.append(cloud_count)
            edge_percentages.append(edge_pct)
            cloud_percentages.append(cloud_pct)

    df = pd.DataFrame({
        'Node': nodes_data,
        'Edge_Executions': edge_counts,
        'Cloud_Executions': cloud_counts,
        'Edge_Percentage': edge_percentages,
        'Cloud_Percentage': cloud_percentages
    })

    # ==================================================
    # FIGURE 1: Execution Counts & Percentages
    # ==================================================
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    x = range(len(df))
    width = 0.6

    # --- Execution counts (stacked)
    ax1.bar(
        x,
        df['Edge_Executions'],
        width,
        label='Edge Executions',
        color=EDGE_COLOR
    )

    ax1.bar(
        x,
        df['Cloud_Executions'],
        width,
        bottom=df['Edge_Executions'],
        label='Cloud Executions',
        color=CLOUD_COLOR
    )

    ax1.set_xlabel('Node')
    ax1.set_ylabel('Execution Count')
    ax1.set_title(f'Edge vs Cloud Execution Counts ({total_episodes} Episodes)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Node'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # --- Execution percentages (stacked)
    ax2.bar(
        x,
        df['Edge_Percentage'],
        width,
        label='Edge %',
        color=EDGE_COLOR
    )

    ax2.bar(
        x,
        df['Cloud_Percentage'],
        width,
        bottom=df['Edge_Percentage'],
        label='Cloud %',
        color=CLOUD_COLOR
    )

    ax2.set_xlabel('Node')
    ax2.set_ylabel('Execution Percentage (%)')
    ax2.set_title('Execution Location Percentages')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Node'], rotation=45, ha='right')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f'node_execution_comparison_{total_episodes}_episodes.png',
        dpi=150,
        bbox_inches='tight'
    )
    plt.show()

    # ==================================================
    # FIGURE 2: Edge Preference by Node
    # ==================================================
    fig2, ax = plt.subplots(figsize=(14, 6))

    colors = []
    for pct in edge_percentages:
        if pct > 70:
            colors.append(EDGE_COLOR)
        elif pct >= 30:
            colors.append(BALANCED_COLOR)
        else:
            colors.append(CLOUD_COLOR)

    bars = ax.bar(nodes_data, edge_percentages, color=colors)

    ax.set_xlabel('Node')
    ax.set_ylabel('Edge Execution Percentage (%)')
    ax.set_title(f'Edge Execution Preference by Node ({total_episodes} Episodes)')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    for bar, pct in zip(bars, edge_percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{pct:.0f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )

    legend_elements = [
        Patch(facecolor=EDGE_COLOR, label='Prefers Edge (>70%)'),
        Patch(facecolor=BALANCED_COLOR, label='Balanced (30–70%)'),
        Patch(facecolor=CLOUD_COLOR, label='Prefers Cloud (<30%)')
    ]

    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(
        f'edge_preference_{total_episodes}_episodes.png',
        dpi=150,
        bbox_inches='tight'
    )
    plt.show()

    # ==================================================
    # TEXT STATISTICS
    # ==================================================
    print("\n" + "=" * 80)
    print("DETAILED NODE EXECUTION STATISTICS:")
    print("=" * 80)
    print(f"{'Node':<10} {'Edge':<10} {'Cloud':<10} {'Total':<10} {'Edge %':<10} {'Cloud %':<10} {'Preference':<12}")
    print("-" * 80)

    for _, row in df.iterrows():
        total = row['Edge_Executions'] + row['Cloud_Executions']
        if row['Edge_Percentage'] > 70:
            pref = "EDGE"
        elif row['Edge_Percentage'] >= 30:
            pref = "BALANCED"
        else:
            pref = "CLOUD"

        print(f"{row['Node']:<10} {row['Edge_Executions']:<10} {row['Cloud_Executions']:<10} "
              f"{total:<10} {row['Edge_Percentage']:<10.1f} {row['Cloud_Percentage']:<10.1f} {pref:<12}")

    print("=" * 80)

    return df


# ==================================================
# SIMPLE EDGE-ONLY VIEW (color-blind safe)
# ==================================================
def plot_simple_edge_execution(edge_stats, cloud_stats, profiling_data, total_episodes):
    fig, ax = plt.subplots(figsize=(12, 6))

    nodes = []
    edge_percentages = []

    for layer_idx, layer_nodes in enumerate(profiling_data.layers):
        for node_idx in range(len(layer_nodes)):
            key = (layer_idx, node_idx)
            edge = edge_stats.get(key, 0)
            cloud = cloud_stats.get(key, 0)
            total = edge + cloud

            pct = (edge / total) * 100 if total > 0 else 0

            nodes.append(f"L{layer_idx}_N{node_idx}")
            edge_percentages.append(pct)

    colors = [
        EDGE_COLOR if p > 70 else CLOUD_COLOR if p < 30 else BALANCED_COLOR
        for p in edge_percentages
    ]

    bars = ax.bar(nodes, edge_percentages, color=colors)

    ax.set_xlabel('Node')
    ax.set_ylabel('Edge Execution Percentage (%)')
    ax.set_title(f'Edge Execution Percentage by Node ({total_episodes} Episodes)')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(
        f'simple_edge_stats_{total_episodes}_episodes.png',
        dpi=150,
        bbox_inches='tight'
    )
    plt.show()
