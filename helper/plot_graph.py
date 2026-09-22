import numpy as np
import matplotlib.pyplot as plt

def create_energy_bar_graph_clean():
    """
    Create a clean bar graph with better spacing and layout.
    """
    # Set global font settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['figure.titlesize'] = 28
    
    # Your data
    energy_values = [3.62, 3.09, 5.37, 4.62, 7.12, 7.37]
    
    # Updated labels
    labels = [
        'Level-wise DQ',
        'Level-wise A2C', 
        'Coarse-grained',
        'All U',
        'All C',
        'Random'
    ]
    
    # Colors
    colors = [
        'tab:blue',    # Level-wise DQ
        'tab:green',   # Level-wise A2C
        'tab:cyan',    # Coarse-grained
        'tab:olive',   # All U
        'tab:gray',    # All C
        'tab:brown'    # Random
    ]
    
    # Create figure with better aspect ratio
    fig, ax = plt.subplots(figsize=(18, 10))
    
    x = np.arange(len(labels))
    width = 0.65
    
    # Create bars with more spacing
    bars = ax.bar(x, energy_values, width, color=colors, 
                  edgecolor='black', linewidth=2, alpha=0.85)
    
    # Add value labels with better positioning
    for i, (bar, value) in enumerate(zip(bars, energy_values)):
        height = bar.get_height()
        # Position text inside the top of the bar for cleaner look
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.3,
                f'{value:.2f}', ha='center', va='top',
                fontsize=26, fontweight='bold', color='white',
                fontfamily='Times New Roman',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
    
    # Set x-ticks with better spacing
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=26, fontfamily='Times New Roman')
    
    # Rotate labels if they're overlapping (uncomment if needed)
    # ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=26, fontfamily='Times New Roman')
    
    # Set y-axis with proper limits
    ax.set_ylim(0, max(energy_values) * 1.2)  # 20% headroom
    ax.set_yticks(np.arange(0, max(energy_values) * 1.2, 1.0))
    ax.set_ylabel("Average Energy (Joules)", fontsize=28, 
                  fontfamily='Times New Roman', labelpad=15)
    
    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=1)
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add title with padding
    ax.set_title("Energy Consumption Comparison at 500ms Deadline", fontsize=32, 
                 fontfamily='Times New Roman', pad=25)
    
    # Adjust layout to prevent bottom clutter
    plt.tight_layout(pad=3.0)
    
    # Save with high resolution
    plt.savefig("energy_comparison_clean.png", dpi=600, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax

# Alternative: Version with vertical bars (if horizontal labels are too crowded)
def create_energy_bar_graph_horizontal():
    """
    Create horizontal bar graph to prevent bottom clutter.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.titlesize': 28,
        'axes.labelsize': 28,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'figure.titlesize': 28
    })
    
    energy_values = [3.62, 3.09, 5.37, 4.62, 7.12, 7.37]
    labels = ['Level-wise DQ', 'Level-wise A2C', 'Coarse-grained', 'All U', 'All C', 'Random']
    colors = ['tab:blue', 'tab:green', 'tab:cyan', 'tab:olive', 'tab:gray', 'tab:brown']
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(labels))
    
    # Create horizontal bars
    bars = ax.barh(y_pos, energy_values, height=0.6, color=colors, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels at the end of each bar
    for i, (bar, value) in enumerate(zip(bars, energy_values)):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{value:.2f}', va='center', fontsize=24,
                fontweight='bold', fontfamily='Times New Roman')
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=24, fontfamily='Times New Roman')
    
    # Set x-axis
    ax.set_xlim(0, max(energy_values) * 1.2)
    ax.set_xlabel("Average Energy (Joules)", fontsize=28, 
                  fontfamily='Times New Roman', labelpad=15)
    
    # Add vertical grid lines
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_title("Energy Consumption Comparison at 500ms Deadline", fontsize=30, 
                 fontfamily='Times New Roman', pad=20)
    
    plt.tight_layout()
    plt.savefig("energy_comparison_horizontal.png", dpi=600, bbox_inches='tight')
    plt.show()

# Option 3: Minimalist version with maximum space
def create_energy_bar_graph_minimal():
    """
    Minimalist version with maximum space and no clutter.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.titlesize': 26,
        'axes.labelsize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22
    })
    
    energy_values = [3.62, 3.09, 5.37, 4.62, 7.12, 7.37]
    labels = ['Level-wise DQ', 'Level-wise A2C', 'Coarse-grained', 'All U', 'All C', 'Random']
    colors = ['tab:blue', 'tab:green', 'tab:cyan', 'tab:olive', 'tab:gray', 'tab:brown']
    
    # Wider figure for more space
    fig, ax = plt.subplots(figsize=(20, 8))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, energy_values, width=0.5, color=colors, 
                  edgecolor='black', linewidth=1.5)
    
    # Values inside bars (minimal)
    for bar, value in zip(bars, energy_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{value:.2f}', ha='center', va='center',
                fontsize=20, fontweight='bold', color='white',
                fontfamily='Times New Roman')
    
    # X-axis labels with more space
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22, fontfamily='Times New Roman')
    
    # Y-axis
    ax.set_ylim(0, max(energy_values) * 1.15)
    ax.set_ylabel("Energy (J)", fontsize=26, fontfamily='Times New Roman')
    
    # Minimal grid
    ax.yaxis.grid(True, linestyle=':', alpha=0.3)
    
    # Remove all spines except bottom
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    
    ax.set_title("Energy Consumption Comparison at 500ms Deadline", fontsize=28, 
                 fontfamily='Times New Roman', pad=20)
    
    # Maximum space
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.08, right=0.95)
    
    plt.savefig("energy_comparison_minimal.png", dpi=600)
    plt.show()

# Run different options
if __name__ == "__main__":
    print("Creating clean vertical bar graph...")
    create_energy_bar_graph_clean()
    
    print("\nCreating horizontal bar graph (best for avoiding bottom clutter)...")
    create_energy_bar_graph_horizontal()
    
    print("\nCreating minimalist version...")
    create_energy_bar_graph_minimal()