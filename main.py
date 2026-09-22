import random

from matplotlib import pyplot as plt
from profiling.initialize_profiling import get_profiling_data
from reference_schedulers.random_scheduler import run_random_scheduler
from simulator.a2c_simulator import run_a2c_simulation
# from simulator.doubleQ_simulator import run_simulation       # if you want DQ
# from a2c.coarse_grained_dq import run_oneshot_doubleQ_simulation
# from a2c.coarse_grained_a2c import run_oneshot_a2c_simulation
import numpy as np

# ---------- CONFIGURATION ----------
# Set this to your actual trace CSV path, or None to use stochastic bandwidth/RTT
TRACE_CSV_PATH = "C:\\Users\\SIU856613348\\OneDrive - Southern Illinois University\\doubleQLearning\\simulator\\data\\bw_data_trace.csv"   # <--- UPDATE THIS
TIMEOUT_THRESHOLD_MS = 25                  # timeout threshold in ms

# ---------- PLOTTING STYLE ----------
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

if __name__ == "__main__":
    is_test = False
    episodes = 1000000
    max_steps = 10
    deadlines = list(range(500, 501, 50)) 

    # Results containers
    a2c_energy, a2c_time, a2c_deadline_misses = [], [], []
    random_energy, random_time, random_deadline_misses = [], [], []
    cloud_energy, cloud_time, cloud_deadline_misses = [], [], []

    for d in deadlines:
        print(f"\n{'='*60}")
        print(f"Running simulations for deadline: {d} ms")
        print(f"{'='*60}")

        profiling_data = get_profiling_data(d, 8)

        # ----- EdgeWise A2C (level-wise, ternary actions, timeout) -----
        print("Running EdgeWise A2C...")
        a2c_e, a2c_t, a2c_dm = run_a2c_simulation(
            profiling_data,
            episodes=episodes,
            max_steps=max_steps,
            is_test=is_test,
            visualize_stats=False,
            plot_rewards=False,
            smoothing_window=50,
            trace_csv_path=TRACE_CSV_PATH,
            timeout_threshold_ms=TIMEOUT_THRESHOLD_MS
        )
        a2c_energy.append(a2c_e)
        a2c_time.append(a2c_t)
        a2c_deadline_misses.append(a2c_dm / episodes)
        print(f"  A2C: Energy={a2c_e:.3f}J, Time={a2c_t:.1f}ms, Miss={a2c_dm/episodes*100:.2f}%")

        # # ----- Baselines (binary actions) -----
        # # All Cloud
        # ce, ct, cloud_missed = run_random_scheduler(
        #     profiling_data, 1000, max_steps,
        #     is_random=False, is_all_cloud=True
        # )
        # cloud_energy.append(ce)
        # cloud_time.append(ct)
        # cloud_deadline_misses.append(cloud_missed / 1000)

        # # Random scheduler
        # re, rt, random_missed = run_random_scheduler(
        #     profiling_data, 1000, max_steps,
        #     is_random=True, is_all_cloud=False
        # )
        # random_energy.append(re)
        # random_time.append(rt)
        # random_deadline_misses.append(random_missed / 1000)

        # (Optional) All Edge – can be added similarly if desired
        # ee, et, edge_missed = run_random_scheduler(... is_all_cloud=False, is_random=False)
        # edge_energy.append(ee) ...

    # # ---------- PLOT: Deadline Miss Rate ----------
    # plt.figure(figsize=(14, 8))
    # plt.plot(deadlines, a2c_deadline_misses, label="EdgeWise A2C (level-wise)", marker='o', linewidth=3)
    # plt.plot(deadlines, cloud_deadline_misses, label="All Cloud", marker='x', linewidth=3)
    # plt.plot(deadlines, random_deadline_misses, label="Random", marker='^', linewidth=3)

    # plt.xlabel("Deadline (ms)", fontsize=28, fontfamily='Times New Roman')
    # plt.ylabel("Deadline Miss Rate (%)", fontsize=28, fontfamily='Times New Roman')
    # plt.legend(
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 1.02),
    #     ncol=3,
    #     frameon=False,
    #     prop={'family': 'Times New Roman', 'size': 24}
    # )
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.xticks(fontsize=24, fontfamily='Times New Roman')
    # plt.yticks(fontsize=24, fontfamily='Times New Roman')
    # plt.tight_layout()
    # plt.savefig("deadline_misses_edgewise_comparison.png", dpi=600)
    # plt.show()

    # # Optional: Energy vs Deadline plot
    # plt.figure(figsize=(14, 8))
    # plt.plot(deadlines, a2c_energy, label="EdgeWise A2C", marker='o', linewidth=3)
    # plt.plot(deadlines, cloud_energy, label="All Cloud", marker='x', linewidth=3)
    # plt.plot(deadlines, random_energy, label="Random", marker='^', linewidth=3)
    # plt.xlabel("Deadline (ms)", fontsize=28, fontfamily='Times New Roman')
    # plt.ylabel("Average Energy (J)", fontsize=28, fontfamily='Times New Roman')
    # plt.legend(
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 1.02),
    #     ncol=3,
    #     frameon=False,
    #     prop={'family': 'Times New Roman', 'size': 24}
    # )
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.xticks(fontsize=24, fontfamily='Times New Roman')
    # plt.yticks(fontsize=24, fontfamily='Times New Roman')
    # plt.tight_layout()
    # plt.savefig("energy_edgewise_comparison.png", dpi=600)
    # plt.show()

    # # Print summary
    # print("\n" + "="*60)
    # print("SUMMARY OF RESULTS (EdgeWise A2C)")
    # print("="*60)
    # for idx, d in enumerate(deadlines):
    #     print(f"\nDeadline {d}ms:")
    #     print(f"  A2C Energy: {a2c_energy[idx]:.3f} J, Miss: {a2c_deadline_misses[idx]*100:.1f}%")
    #     print(f"  Cloud Energy: {cloud_energy[idx]:.3f} J, Miss: {cloud_deadline_misses[idx]*100:.1f}%")
    #     print(f"  Random Energy: {random_energy[idx]:.3f} J, Miss: {random_deadline_misses[idx]*100:.1f}%")
    # print("="*60)