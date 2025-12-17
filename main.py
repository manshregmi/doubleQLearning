import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from profiling.initialize_profiling import get_profiling_data
from simulator.a2c_simulator import run_a2c_simulation
from simulator.doubleQ_simulator import run_simulation
from simulator.sac_simulator import run_sac_simulation

def clean_up_files():
    """Deletes existing model/table files to ensure fresh training each iteration."""
    files = ["q_tables.pkl", "sac_model_ep500.pth", "value_table.npy", "policy_table.npy"]
    for f in files:
        if os.path.exists(f):
            os.remove(f)

def run_bin_analysis():
    # Simulation Parameters
    train_episodes = 500
    test_episodes = 100
    max_steps = 10
    target_deadline = 500 
    profiling_data = get_profiling_data(target_deadline)

    # Bin Ranges (Adjust as needed)
    bw_bin_range = [5, 10, 15]
    ct_bin_range = [5, 10, 20]
    surplus_bin_range = [5, 15, 25]

    combinations = list(product(bw_bin_range, ct_bin_range, surplus_bin_range))
    results_matrix = []

    for (bw, ct, surp) in combinations:
        print(f"\n>>> Analyzing Combination: BW={bw}, CT={ct}, Surplus={surp}")
        
        # --- PHASE 1: TRAINING ---
        # Ensure your simulators handle training when is_test=False
        run_simulation(profiling_data, train_episodes, max_steps, is_test=False, 
                       BW_bins=bw, CT_bins=ct, surplus_bins=surp)
        run_a2c_simulation(profiling_data, train_episodes, max_steps, is_test=False, 
                           BW_bins=bw, CT_bins=ct, surplus_bins=surp)
        run_sac_simulation(profiling_data, train_episodes, max_steps, is_test=False, 
                           BW_bins=bw, CT_bins=ct, surplus_bins=surp)

        # --- PHASE 2: TESTING ---
        # Evaluate performance using the weights/tables trained above
        dq_e, _ = run_simulation(profiling_data, test_episodes, max_steps, is_test=True, 
                                 BW_bins=bw, CT_bins=ct, surplus_bins=surp)
        a2c_e, _ = run_a2c_simulation(profiling_data, test_episodes, max_steps, is_test=True, 
                                      BW_bins=bw, CT_bins=ct, surplus_bins=surp)
        sac_e, _ = run_sac_simulation(profiling_data, test_episodes, max_steps, is_test=True, 
                                      BW_bins=bw, CT_bins=ct, surplus_bins=surp)

        # Store test results
        results_matrix.append([bw, ct, surp, dq_e, a2c_e, sac_e])
        
        # --- CLEANUP ---
        clean_up_files()

    # Final Plotting Logic (3D Scatter)
    results_matrix = np.array(results_matrix)
    plot_results(results_matrix)

def plot_results(data):
    fig = plt.figure(figsize=(15, 5))
    algos = [("Double Q", 3), ("A2C", 4), ("SAC", 5)]
    for i, (name, col) in enumerate(algos, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        img = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,col], cmap='viridis_r', s=80)
        ax.set_title(f"Test Energy: {name}")
        ax.set_xlabel('BW'); ax.set_ylabel('CT'); ax.set_zlabel('Surplus')
        fig.colorbar(img, ax=ax, shrink=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    clean_up_files()
    run_bin_analysis()