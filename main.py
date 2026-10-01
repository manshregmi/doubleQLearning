import random
import numpy as np
import matplotlib.pyplot as plt

from profiling.initialize_profiling import get_profiling_data
from simulator.a2c_simulator import run_a2c_simulation


# ---------- CONFIGURATION ----------
TRACE_CSV_PATH = ("C:\\Users\\SIU856613348\\OneDrive - Southern Illinois University\\"
                  "doubleQLearning\\simulator\\data\\bw_data_trace.csv")

DEADLINE_MS = 500
N_DEVICES = 8

# Thresholds to sweep over
TIMEOUT_THRESHOLDS_MS = [25, 50, 75, 100, 150, 200, 300]

# Training settings
TRAIN_EPISODES = 1000000
EVAL_EPISODES = 10000
MAX_STEPS = 10
PACKET_LOSS = 0.10
MODEL_PATH = "a2c_tables.pkl"
SEED = 42

# ---------- PLOTTING STYLE ----------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10


if __name__ == "__main__":
    # Set up profiling (deadline fixed at 500 ms)
    profiling_data = get_profiling_data(DEADLINE_MS, N_DEVICES)

    # ---------- 1) TRAIN ONCE with threshold in state, randomized across sweep ----------
    print("\n" + "=" * 70)
    print("TRAINING: agent sees all thresholds during training")
    print("=" * 70)
    run_a2c_simulation(
        profiling_data,
        episodes=TRAIN_EPISODES,
        max_steps=MAX_STEPS,
        is_test=False,
        visualize_stats=False,
        plot_rewards=False,
        trace_csv_path=TRACE_CSV_PATH,
        timeout_threshold_ms=TIMEOUT_THRESHOLDS_MS[0],
        packet_loss_prob=PACKET_LOSS,
        model_path=MODEL_PATH,
        train_thresholds=TIMEOUT_THRESHOLDS_MS,  # sample threshold each episode
        seed=SEED,
    )

    # ---------- 2) EVALUATE AT EACH THRESHOLD ----------
    print("\n" + "=" * 70)
    print("EVALUATION: sweeping timeout thresholds")
    print("=" * 70)

    energies, times, miss_rates, timeouts = [], [], [], []

    for th in TIMEOUT_THRESHOLDS_MS:
        result = run_a2c_simulation(
            profiling_data,
            episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS,
            is_test=False,
            visualize_stats=False,
            plot_rewards=False,
            trace_csv_path=TRACE_CSV_PATH,
            timeout_threshold_ms=th,
            packet_loss_prob=PACKET_LOSS,
            model_path=MODEL_PATH,
            train_thresholds=None,
            seed=SEED,  # same seed for paired evaluation
        )
        energies.append(result["avg_energy"])
        times.append(result["avg_time_ms"])
        miss_rates.append(result["deadline_miss_rate"] * 100.0)
        timeouts.append(result["avg_timeouts_per_ep"])
        print(f"  τ_to = {th:>4} ms -> "
              f"E = {result['avg_energy']:.3f} J, "
              f"T = {result['avg_time_ms']:.1f} ms, "
              f"DL-miss = {result['deadline_miss_rate']*100:.1f}%, "
              f"timeouts/ep = {result['avg_timeouts_per_ep']:.2f}")

    # ---------- 3) PLOTS ----------
    ths = np.array(TIMEOUT_THRESHOLDS_MS)

    # Energy vs threshold
    plt.figure(figsize=(10, 7))
    plt.plot(ths, energies, marker='o', color='#1f77b4')
    plt.xlabel("Timeout Threshold (ms)", fontfamily='Times New Roman')
    plt.ylabel("Average Energy (J)", fontfamily='Times New Roman')
    plt.title(f"Energy vs Timeout Threshold (Deadline={DEADLINE_MS} ms)",
              fontfamily='Times New Roman')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("energy_vs_threshold.png", dpi=300)
    plt.show()

    # Completion time vs threshold
    plt.figure(figsize=(10, 7))
    plt.plot(ths, times, marker='s', color='#2ca02c')
    plt.axhline(y=DEADLINE_MS, color='red', linestyle='--',
                linewidth=2, label=f"Deadline = {DEADLINE_MS} ms")
    plt.xlabel("Timeout Threshold (ms)", fontfamily='Times New Roman')
    plt.ylabel("Average Completion Time (ms)", fontfamily='Times New Roman')
    plt.title(f"Completion Time vs Timeout Threshold (Deadline={DEADLINE_MS} ms)",
              fontfamily='Times New Roman')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("time_vs_threshold.png", dpi=300)
    plt.show()

    # Deadline miss rate vs threshold
    plt.figure(figsize=(10, 7))
    plt.plot(ths, miss_rates, marker='^', color='#d62728')
    plt.xlabel("Timeout Threshold (ms)", fontfamily='Times New Roman')
    plt.ylabel("Deadline Miss Rate (%)", fontfamily='Times New Roman')
    plt.title(f"Deadline Miss Rate vs Timeout Threshold (Deadline={DEADLINE_MS} ms)",
              fontfamily='Times New Roman')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("miss_rate_vs_threshold.png", dpi=300)
    plt.show()

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Threshold (ms)':<16} {'Energy (J)':<12} {'Time (ms)':<12} "
          f"{'DL-miss %':<12} {'TO/ep':<10}")
    for i, th in enumerate(TIMEOUT_THRESHOLDS_MS):
        print(f"{th:<16} {energies[i]:<12.3f} {times[i]:<12.1f} "
              f"{miss_rates[i]:<12.2f} {timeouts[i]:<10.2f}")
    print("=" * 70)