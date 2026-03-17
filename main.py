import random
from profiling.initialize_profiling import get_profiling_data
import matplotlib.pyplot as plt
from reference_schedulers.random_scheduler import run_random_scheduler
from simulator.a2c_simulator import run_a2c_simulation
from simulator.doubleQ_simulator import run_simulation
from a2c.coarse_grained_dq import run_oneshot_doubleQ_simulation
from a2c.coarse_grained_a2c import run_oneshot_a2c_simulation
import numpy as np

from simulator.ppo_simulator import run_ppo_simulation

if __name__ == "__main__":
    # Set global font settings for all plots - Times New Roman with conference paper sizes
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 10
    
    is_test = False
    episodes = 100
    # episodes = 100
    max_steps = 10
    deadlines = list(range(400, 601, 10))  # 400ms to 600ms
    
    # existing methods
    dq_energy, dq_time, dq_deadline_misses = [], [], []
    a2c_energy, a2c_time, a2c_deadline_misses = [], [], []
    ppo_energy, ppo_time, ppo_deadline_misses = [], [], []
    
    # NEW: One-shot baselines
    oneshot_dq_energy, oneshot_dq_time, oneshot_dq_deadline_misses = [], [], []
    oneshot_a2c_energy, oneshot_a2c_time, oneshot_a2c_deadline_misses = [], [], []
    
    # Baseline methods
    random_energy, random_time, random_deadline_misses = [], [], []
    edge_energy, edge_time, edge_deadline_misses = [], [], []
    cloud_energy, cloud_time, cloud_dadline_misses = [], [], []

    for d in deadlines:
        print(f"\n{'='*60}")
        print(f"Running simulations for deadline: {d} ms")
        print(f"{'='*60}")
        
        profiling_data = get_profiling_data(d, 8)

        # Your level-wise methods
        e, t, dm = run_simulation(profiling_data, episodes, max_steps, is_test)
        dq_energy.append(e)
        dq_time.append(t)
        dq_deadline_misses.append(dm/episodes)

        a2c_e, a2c_t, a2c_dm = run_a2c_simulation(profiling_data, episodes, max_steps, is_test)
        a2c_energy.append(a2c_e)
        a2c_time.append(a2c_t)
        a2c_deadline_misses.append(a2c_dm/episodes)
        print("average energy is ", a2c_e, a2c_t)
        
        # NEW: One-shot baselines
        try:
            oneshot_dq_e, oneshot_dq_t, oneshot_dq_dm = run_oneshot_doubleQ_simulation(
                profiling_data, episodes, max_steps, is_test
            )
            oneshot_dq_energy.append(oneshot_dq_e)
            oneshot_dq_time.append(oneshot_dq_t)
            oneshot_dq_deadline_misses.append(oneshot_dq_dm/episodes)
        except Exception as e:
            print(f"Error running one-shot DQ: {e}")
            oneshot_dq_energy.append(float('nan'))
            oneshot_dq_time.append(float('nan'))
            oneshot_dq_deadline_misses.append(float('nan'))

        # PPO 
        ppo_e, ppo_t, ppo_dm = run_ppo_simulation(profiling_data, episodes, max_steps, is_test)
        ppo_energy.append(ppo_e)
        ppo_time.append(ppo_t)
        ppo_deadline_misses.append(ppo_dm/episodes)
    
        
    #     # try:
    #     #     oneshot_a2c_e, oneshot_a2c_t, oneshot_a2c_dm = run_oneshot_a2c_simulation(
    #     #         profiling_data, episodes, max_steps, is_test
    #     #     )
    #     #     oneshot_a2c_energy.append(oneshot_a2c_e)
    #     #     oneshot_a2c_time.append(oneshot_a2c_t)
    #     #     oneshot_a2c_deadline_misses.append(oneshot_a2c_dm/episodes)
    #     # except Exception as e:
    #     #     print(f"Error running one-shot A2C: {e}")
    #     #     oneshot_a2c_energy.append(float('nan'))
    #     #     oneshot_a2c_time.append(float('nan'))
    #     #     oneshot_a2c_deadline_misses.append(float('nan'))

        # Baseline methods
        re, rt, random_deadline_missed = run_random_scheduler(profiling_data, episodes, max_steps, is_random=True, is_all_cloud=False)
        random_energy.append(re)
        random_time.append(rt)
        random_deadline_misses.append(random_deadline_missed/episodes)

        # ee, et, edge_deadline_missed = run_random_scheduler(profiling_data, 1000, max_steps, is_random=False, is_all_cloud=False)
        # edge_energy.append(ee)
        # edge_time.append(et)
        # edge_deadline_misses.append(edge_deadline_missed/1000)

        ce, ct, cloud_deadline_missed = run_random_scheduler(profiling_data, 1000, max_steps, is_random=False, is_all_cloud=True)
        cloud_energy.append(ce)
        cloud_time.append(ct)
        cloud_dadline_misses.append(cloud_deadline_missed/1000)


    # FIX: Remove NaN values for plotting
    def remove_nan(x_data, y_data):
        """Remove NaN values from data for plotting."""
        clean_x, clean_y = [], []
        for x, y in zip(x_data, y_data):
            if not np.isnan(y):
                clean_x.append(x)
                clean_y.append(y)
        return clean_x, clean_y

    # # Plot 1: Energy vs Deadline (Comparison)
    # plt.figure(figsize=(14, 8))
    # plt.plot(deadlines, dq_energy, label="Level-wise DQ", marker='o', linewidth=3)
    # plt.plot(deadlines, a2c_energy, label="Level-wise A2C", marker='*', linewidth=3)
    
    # # Plot one-shot methods only if they have data
    # if any(not np.isnan(v) for v in oneshot_dq_energy):
    #     clean_deadlines, clean_oneshot_dq = remove_nan(deadlines, oneshot_dq_energy)
    #     if clean_deadlines:
    #         plt.plot(clean_deadlines, clean_oneshot_dq, label="Coarse grained DQ", marker='s', linestyle='--', linewidth=2)
    
    # if any(not np.isnan(v) for v in oneshot_a2c_energy):
    #     clean_deadlines, clean_oneshot_a2c = remove_nan(deadlines, oneshot_a2c_energy)
    #     if clean_deadlines:
    #         plt.plot(clean_deadlines, clean_oneshot_a2c, label="Coarse grained A2C", marker='^', linestyle='--', linewidth=2)
    
    # plt.plot(deadlines, edge_energy, label="All Edge", marker='x', color='gray', alpha=0.7, linewidth=2)
    # plt.plot(deadlines, cloud_energy, label="All Cloud", marker='+', color='gray', alpha=0.7, linewidth=2)
    
    # plt.xlabel("Deadline (ms)", fontsize=28, fontfamily='Times New Roman')
    # plt.ylabel("Average Energy (Joules)", fontsize=28, fontfamily='Times New Roman')
    # plt.title("Energy vs Deadline: Level-wise vs One-shot RL", fontsize=28, fontfamily='Times New Roman')
    # plt.legend(fontsize=24, loc='best')
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.xticks(fontsize=24, fontfamily='Times New Roman')
    # plt.yticks(fontsize=24, fontfamily='Times New Roman')
    # plt.tight_layout()
    # plt.savefig("energy_vs_deadline_comparison.png", dpi=600)
    # plt.show()

    # Plot 2: Deadline Miss Rate vs Deadline
    plt.figure(figsize=(14, 8))
    plt.plot(deadlines, dq_deadline_misses, label="Level-wise DQ", marker='o', linewidth=3)
    plt.plot(deadlines, a2c_deadline_misses, label="Level-wise A2C", marker='*', linewidth=3)
    plt.plot(deadlines, ppo_deadline_misses, label="Level-wise PPO", marker='x', linewidth=3)
    plt.plot(deadlines, cloud_dadline_misses, label="All Cloud", marker='+', linewidth=3)
    plt.plot(deadlines, random_deadline_misses, label="Random", marker='^', linewidth=3)

    # Plot one-shot methods only if they have data
    if any(not np.isnan(v) for v in oneshot_dq_deadline_misses):
        clean_deadlines, clean_oneshot_dq = remove_nan(deadlines, oneshot_dq_deadline_misses)
        if clean_deadlines:
            plt.plot(clean_deadlines, clean_oneshot_dq, label="Coarse grained", marker='s', linestyle='--', linewidth=2)
    
    # if any(not np.isnan(v) for v in oneshot_a2c_deadline_misses):
    #     clean_deadlines, clean_oneshot_a2c = remove_nan(deadlines, oneshot_a2c_deadline_misses)
    #     if clean_deadlines:
    #         plt.plot(clean_deadlines, clean_oneshot_a2c, label="Coarse grained A2C", marker='^', linestyle='--', linewidth=2)
    
    plt.xlabel("Deadline (ms)", fontsize=28, fontfamily='Times New Roman')
    plt.ylabel("Deadline Miss Rate (%)", fontsize=28, fontfamily='Times New Roman')
    # plt.title("Deadline Miss Rate vs Deadline", fontsize=28, fontfamily='Times New Roman')
    # plt.legend(fontsize=24, loc='best')
    plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,   # increase if you have many entries
    frameon=False,
    prop={'family': 'Times New Roman', 'size': 24}
)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=24, fontfamily='Times New Roman')
    plt.yticks(fontsize=24, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig("deadline_misses_comparison.png", dpi=600)
    plt.show()

    # # Helper function for bar charts
    # def create_bar_chart(deadline_value, deadline_idx, suffix=""):
    #     """Create bar chart for specific deadline."""
    #     # FIX: Check if deadline_idx exists
    #     if deadline_idx >= len(deadlines) or deadline_idx < 0:
    #         print(f"Warning: Deadline {deadline_value}ms not in deadlines list")
    #         return
        
    #     # FIX: Check array bounds
    #     energies = []
    #     labels = []
        
    #     # Collect available data
    #     if deadline_idx < len(dq_energy):
    #         energies.append(dq_energy[deadline_idx])
    #         labels.append('Level DQ')
        
    #     if deadline_idx < len(a2c_energy):
    #         energies.append(a2c_energy[deadline_idx])
    #         labels.append('Level A2C')
        
    #     if deadline_idx < len(oneshot_dq_energy) and not np.isnan(oneshot_dq_energy[deadline_idx]):
    #         energies.append(oneshot_dq_energy[deadline_idx])
    #         labels.append('Coarse-grained')
        
    #     # if deadline_idx < len(oneshot_a2c_energy) and not np.isnan(oneshot_a2c_energy[deadline_idx]):
    #     #     energies.append(oneshot_a2c_energy[deadline_idx])
    #     #     labels.append('Coarse-grained A2C')
        
    #     if deadline_idx < len(edge_energy):
    #         energies.append(edge_energy[deadline_idx])
    #         labels.append('All Edge')
        
    #     if deadline_idx < len(cloud_energy):
    #         energies.append(cloud_energy[deadline_idx])
    #         labels.append('All Cloud')

    #     if deadline_idx < len(random_energy):
    #         energies.append(random_energy[deadline_idx])
    #         labels.append('Random')
        
    #     if not energies:
    #         print(f"No data available for {deadline_value}ms deadline")
    #         return
        
    #     # Create bar chart
    #     x = np.arange(len(labels))
    #     width = 0.7
        
    #     plt.figure(figsize=(16, 9))  # Larger figure for better text fitting
    #     colors = [
    #             'tab:blue',
    #             'tab:green',
    #             'tab:cyan',
    #             'tab:olive',
    #             'tab:gray',
    #             'tab:brown',
    #             # 'tab:orange'
    #         ]
        
    #     # Use only as many colors as we have bars
    #     bars = plt.bar(x, energies, width, color=colors[:len(energies)], edgecolor='black', linewidth=1.5)
        
    #     # Add value labels on bars with Times New Roman and larger font
    #     for i, v in enumerate(energies):
    #         plt.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', 
    #                  fontsize=22, fontfamily='Times New Roman', fontweight='bold')
        
    #     plt.xticks(x, labels, rotation=45, ha='right', fontsize=24, fontfamily='Times New Roman')
    #     plt.yticks(fontsize=24, fontfamily='Times New Roman')
    #     plt.ylabel("Average Energy (Joules)", fontsize=28, fontfamily='Times New Roman')
    #     plt.title(f"Energy Comparison at {deadline_value} ms Deadline", fontsize=28, fontfamily='Times New Roman')
    #     plt.grid(axis='y', linestyle='--', alpha=0.6)
    #     plt.tight_layout()
    #     plt.savefig(f"energy_bargraph_{deadline_value}ms_comparison{suffix}.png", 
    #                 dpi=600, bbox_inches='tight')
    #     plt.show()
        
    #     return energies, labels

    # # Create bar charts for each deadline
    # for deadline_value in [400, 450, 500, 550, 600]:
    #     try:
    #         deadline_idx = deadlines.index(deadline_value)
    #         create_bar_chart(deadline_value, deadline_idx)
    #     except ValueError:
    #         print(f"Deadline {deadline_value}ms not in deadlines list")

    # # Print results for each deadline
    # print("\n" + "="*60)
    # print("SIMULATION RESULTS SUMMARY")
    # print("="*60)
    
    # for deadline_value in [450, 500, 550, 600]:
    #     try:
    #         deadline_idx = deadlines.index(deadline_value)
    #         print(f"\nResults at {deadline_value}ms deadline:")
            
    #         if deadline_idx < len(dq_energy):
    #             print(f"  Level-wise DQ: Energy={dq_energy[deadline_idx]:.2f}J, "
    #                   f"Miss={dq_deadline_misses[deadline_idx]*100:.1f}%")
            
    #         if deadline_idx < len(a2c_energy):
    #             print(f"  Level-wise A2C: Energy={a2c_energy[deadline_idx]:.2f}J, "
    #                   f"Miss={a2c_deadline_misses[deadline_idx]*100:.1f}%")
            
    #         if (deadline_idx < len(oneshot_dq_energy) and 
    #             not np.isnan(oneshot_dq_energy[deadline_idx])):
    #             print(f"  Coarse-grained DQ: Energy={oneshot_dq_energy[deadline_idx]:.2f}J, "
    #                   f"Miss={oneshot_dq_deadline_misses[deadline_idx]*100:.1f}%")
            
    #         # if (deadline_idx < len(oneshot_a2c_energy) and 
    #         #     not np.isnan(oneshot_a2c_energy[deadline_idx])):
    #         #     print(f"  Coarse-grained A2C: Energy={oneshot_a2c_energy[deadline_idx]:.2f}J, "
    #         #           f"Miss={oneshot_a2c_deadline_misses[deadline_idx]*100:.1f}%")
            
    #         if deadline_idx < len(edge_energy):
    #             print(f"  All Edge: Energy={edge_energy[deadline_idx]:.2f}J")
            
    #         if deadline_idx < len(cloud_energy):
    #             print(f"  All Cloud: Energy={cloud_energy[deadline_idx]:.2f}J")
                
    #     except ValueError:
    #         print(f"\nDeadline {deadline_value}ms not in deadlines list")
    
    # print("\n" + "="*60)
    # print("CONCLUSION: Level-wise RL outperforms coarse-grained RL!")
    # print("="*60)