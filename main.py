import random
import numpy as np
import matplotlib.pyplot as plt

from profiling.initialize_profiling import get_profiling_data
from simulator.a2c_simulator import run_a2c_simulation


# ---------- CONFIGURATION ----------
TRACE_CSV_PATH = (
    "C:\\Users\\SIU856613348\\OneDrive - Southern Illinois University\\"
    "doubleQLearning\\simulator\\data\\bw_data_trace.csv"
)

DEADLINE_MS = 500
N_DEVICES = 8

TIMEOUT_THRESHOLDS_MS = [25, 50, 75, 100, 150, 200, 300]

# Training settings
TRAIN_EPISODES = 1_000_000

# Use substantially more than 100 for final evaluation.
EVAL_EPISODES = 2_000

MAX_STEPS = 10
PACKET_LOSS = 0.10
SEED = 42

DO_TRAIN = True


# ---------- AGENT VARIANTS ----------
VARIANTS = {
    "Ternary A2C": {
        "modes": (0, 1, 2),
        "threshold_aware": True,

        # Keep False for the first controlled experiment.
        # After establishing the fixed agent, test True as a separate ablation.
        "threshold_sharing": False,

        # IMPORTANT: use a NEW filename so old tables produced by the
        # previous actor update are not reused.
        "model_path": "a2c_ternary_fixed.pkl",

        "color": "#1f77b4",
        "marker": "o",
    },

    "Ternary A2C (no threshold)": {
        "modes": (0, 1, 2),
        "threshold_aware": False,
        "threshold_sharing": False,
        "model_path": "a2c_ternary_nothreshold_fixed.pkl",
        "color": "#9467bd",
        "marker": "D",
    },

    "Binary A2C (Optimistic)": {
        "modes": (0, 1),
        "threshold_aware": False,
        "threshold_sharing": False,
        "model_path": "a2c_binary_optimistic_fixed.pkl",
        "color": "#ff7f0e",
        "marker": "s",
    },

    "Binary A2C (Conservative)": {
        "modes": (0, 2),
        "threshold_aware": False,
        "threshold_sharing": False,
        "model_path": "a2c_binary_conservative_fixed.pkl",
        "color": "#2ca02c",
        "marker": "^",
    },
}


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 24
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 10


def plot_metric(
    ths, results, key, ylabel, title, filename, deadline_line=False
):
    plt.figure(figsize=(10, 7))

    for name, cfg in VARIANTS.items():
        plt.plot(
            ths,
            results[name][key],
            marker=cfg["marker"],
            color=cfg["color"],
            label=name,
        )

    if deadline_line:
        plt.axhline(
            y=DEADLINE_MS,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Deadline = {DEADLINE_MS} ms",
        )

    plt.xlabel("Timeout Threshold (ms)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    profiling_data = get_profiling_data(DEADLINE_MS, N_DEVICES)

    # ---------- 1) TRAIN ----------
    if DO_TRAIN:
        for name, cfg in VARIANTS.items():
            print("\n" + "=" * 70)
            print(
                f"TRAINING: {name}  modes={cfg['modes']}"
            )
            print("=" * 70)

            run_a2c_simulation(
                profiling_data,
                episodes=TRAIN_EPISODES,
                max_steps=MAX_STEPS,
                is_test=False,
                trace_csv_path=TRACE_CSV_PATH,

                # Initial threshold. During training the simulator samples
                # from train_thresholds below.
                timeout_threshold_ms=TIMEOUT_THRESHOLDS_MS[0],

                packet_loss_prob=PACKET_LOSS,
                model_path=cfg["model_path"],
                train_thresholds=TIMEOUT_THRESHOLDS_MS,
                seed=SEED,

                action_modes=cfg["modes"],
                threshold_aware=cfg["threshold_aware"],
                threshold_levels=TIMEOUT_THRESHOLDS_MS,
                threshold_sharing=cfg.get(
                    "threshold_sharing", False
                ),
                label=name,
            )

    # ---------- 2) EVALUATION ----------
    results = {
        name: {
            "energy": [],
            "time": [],
            "miss": [],
            "timeouts": [],
            "mix": [],
            "unseen": [],
            "time_ci": [],
            "miss_ci": [],
            "reward": [],
        }
        for name in VARIANTS
    }

    for name, cfg in VARIANTS.items():
        print("\n" + "=" * 70)
        print(f"EVALUATION: {name}")
        print("=" * 70)

        for th in TIMEOUT_THRESHOLDS_MS:
            r = run_a2c_simulation(
                profiling_data,
                episodes=EVAL_EPISODES,
                max_steps=MAX_STEPS,
                is_test=True,
                trace_csv_path=TRACE_CSV_PATH,
                timeout_threshold_ms=th,
                packet_loss_prob=PACKET_LOSS,
                model_path=cfg["model_path"],
                train_thresholds=None,
                seed=SEED,
                action_modes=cfg["modes"],
                threshold_aware=cfg["threshold_aware"],
                threshold_levels=TIMEOUT_THRESHOLDS_MS,
                threshold_sharing=cfg.get(
                    "threshold_sharing", False
                ),
                label=f"{name} @ {th}ms",
            )

            results[name]["energy"].append(r["avg_energy"])
            results[name]["time"].append(r["avg_time_ms"])
            results[name]["miss"].append(
                r["deadline_miss_rate"] * 100.0
            )
            results[name]["timeouts"].append(
                r["avg_timeouts_per_ep"]
            )

            ac = r["action_counts"]
            total_actions = max(1, sum(ac.values()))

            results[name]["mix"].append(
                f"{100 * ac[0] / total_actions:.0f}/"
                f"{100 * ac[1] / total_actions:.0f}/"
                f"{100 * ac[2] / total_actions:.0f}"
            )

            results[name]["unseen"].append(
                r["unseen_state_rate"] * 100.0
            )

            results[name]["reward"].append(
                float(np.mean(r["rewards"]))
            )

            n = max(1, r["n_episodes"])
            p_miss = r["deadline_miss_rate"]

            # Approximate 95% normal CI for the proportion.
            results[name]["miss_ci"].append(
                1.96
                * np.sqrt(p_miss * (1.0 - p_miss) / n)
                * 100.0
            )

            results[name]["time_ci"].append(
                1.96 * np.std(r["times"], ddof=1) / np.sqrt(n)
            )

    # ---------- 3) PLOTS ----------
    ths = np.array(TIMEOUT_THRESHOLDS_MS)

    plot_metric(
        ths,
        results,
        "energy",
        "Average Energy (J)",
        f"Energy vs Timeout Threshold (Deadline={DEADLINE_MS} ms)",
        "cmp_energy_vs_threshold_fixed.png",
    )

    plot_metric(
        ths,
        results,
        "time",
        "Average Completion Time (ms)",
        f"Completion Time vs Timeout Threshold "
        f"(Deadline={DEADLINE_MS} ms)",
        "cmp_time_vs_threshold_fixed.png",
        deadline_line=True,
    )

    plot_metric(
        ths,
        results,
        "miss",
        "Deadline Miss Rate (%)",
        f"Deadline Miss Rate vs Timeout Threshold "
        f"(Deadline={DEADLINE_MS} ms)",
        "cmp_miss_rate_vs_threshold_fixed.png",
    )

    plot_metric(
        ths,
        results,
        "timeouts",
        "Timeouts per Episode",
        f"Timeouts vs Timeout Threshold "
        f"(Deadline={DEADLINE_MS} ms)",
        "cmp_timeouts_vs_threshold_fixed.png",
    )

    # ---------- SUMMARY ----------
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for name in VARIANTS:
        print(f"\n{name}")
        print(
            f"{'Threshold (ms)':<16} "
            f"{'Energy (J)':<12} "
            f"{'Time (ms)':<18} "
            f"{'DL-miss %':<16} "
            f"{'TO/ep':<8} "
            f"{'E/O/C %':<12} "
            f"{'Unseen %':<10} "
            f"{'Reward':<10}"
        )

        for i, th in enumerate(TIMEOUT_THRESHOLDS_MS):
            t_str = (
                f"{results[name]['time'][i]:.1f}"
                f"+/-{results[name]['time_ci'][i]:.1f}"
            )

            m_str = (
                f"{results[name]['miss'][i]:.1f}"
                f"+/-{results[name]['miss_ci'][i]:.1f}"
            )

            print(
                f"{th:<16} "
                f"{results[name]['energy'][i]:<12.3f} "
                f"{t_str:<18} "
                f"{m_str:<16} "
                f"{results[name]['timeouts'][i]:<8.2f} "
                f"{results[name]['mix'][i]:<12} "
                f"{results[name]['unseen'][i]:<10.1f} "
                f"{results[name]['reward'][i]:<10.1f}"
            )

    print("=" * 90)
