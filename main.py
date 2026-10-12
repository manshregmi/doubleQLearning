import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from profiling.initialize_profiling import get_profiling_data
from simulator.a2c_simulator import run_a2c_simulation


TRACE_CSV_PATH = ("C:\\Users\\SIU856613348\\OneDrive - Southern Illinois University\\"
                  "doubleQLearning\\simulator\\data\\bw_data_trace.csv")

N_DEVICES = 8

DEADLINE_FACTORS = [0.8, 1.0, 1.25, 1.5]
DEADLINES_MS_OVERRIDE = [400, 450, 500, 550, 600, 650, 700]

TRAIN_EPISODES = 3000000
EVAL_EPISODES = 10000
MAX_STEPS = 10
SEED = 42

DO_TRAIN = True

VARIANTS = {
    "Ternary A2C": {
        "modes": (0, 1, 2),
        "model_path": "a2c_ternary_dl.pkl",
        "color": "#1f77b4", "marker": "o",
    },
    "Binary A2C (Optimistic)": {
        "modes": (0, 1),
        "model_path": "a2c_binary_optimistic_dl.pkl",
        "color": "#ff7f0e", "marker": "s",
    },
    "Binary A2C (Conservative)": {
        "modes": (0, 2),
        "model_path": "a2c_binary_conservative_dl.pkl",
        "color": "#2ca02c", "marker": "^",
    },
}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8


def all_local_time_ms(p):
    total = 0.0
    for l in range(len(p.layers)):
        t = [p.get_node_edge_time(l, i) for i in range(p.get_num_nodes(l))]
        total += max(t) if l in (3, 5) else sum(t)
    return total


def plot_vs_deadline(df, metric, ylabel, filename, variant_cfgs,
                     deadline_line=False, logy=False):
    """One line per variant, x-axis = deadline (ms)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ci_col = metric + "_ci"

    for name, cfg in variant_cfgs.items():
        s = df[df["variant"] == name].sort_values("deadline_ms")
        if s.empty:
            continue
        ax.errorbar(
            s["deadline_ms"], s[metric],
            yerr=s[ci_col] if ci_col in s.columns else None,
            marker=cfg["marker"], color=cfg["color"], label=name, capsize=4,
        )

    if deadline_line:
        dmin = df["deadline_ms"].min()
        dmax = df["deadline_ms"].max()
        ax.plot([dmin, dmax], [dmin, dmax],
                color="red", linestyle="--", linewidth=1.5,
                label="Deadline (y = x)")

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Deadline (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs. Deadline")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    profiling_data = get_profiling_data(500, N_DEVICES)

    t_local = all_local_time_ms(profiling_data)
    if DEADLINES_MS_OVERRIDE:
        DEADLINES_MS = [float(d) for d in DEADLINES_MS_OVERRIDE]
    else:
        DEADLINES_MS = [float(round(f * t_local)) for f in DEADLINE_FACTORS]

    print("\n" + "=" * 70)
    print(f"All-local completion time T_local = {t_local:.1f} ms")
    print(f"Deadlines (ms): {DEADLINES_MS}")
    print("=" * 70)

    common = dict(
        max_steps=MAX_STEPS,
        trace_csv_path=TRACE_CSV_PATH,
        deadline_levels=DEADLINES_MS,
    )

    # ---------- 1) TRAIN ----------
    if DO_TRAIN:
        for name, cfg in VARIANTS.items():
            print("\n" + "=" * 70)
            print(f"TRAINING: {name}  modes={cfg['modes']}")
            print("=" * 70)
            run_a2c_simulation(
                profiling_data,
                episodes=TRAIN_EPISODES,
                is_test=False,
                model_path=cfg["model_path"],
                seed=SEED,
                action_modes=cfg["modes"],
                deadline_ms=None,
                label=name,
                **common,
            )

    # ---------- 2) EVALUATE ----------
    rows = []
    for name, cfg in VARIANTS.items():
        for D in DEADLINES_MS:
            r = run_a2c_simulation(
                profiling_data,
                episodes=EVAL_EPISODES,
                is_test=True,
                model_path=cfg["model_path"],
                seed=SEED,
                action_modes=cfg["modes"],
                deadline_ms=D,
                label=f"{name} @ D={D:.0f}",
                **common,
            )
            n = max(1, r["n_episodes"])
            p_miss = r["deadline_miss_rate"]
            ac = r["action_counts"]
            tot = max(1, sum(ac.values()))
            rows.append({
                "variant": name,
                "deadline_ms": D,
                "deadline_factor": D / t_local,
                "energy": r["avg_energy"],
                "energy_ci": 1.96 * np.std(r["energies"]) / np.sqrt(n),
                "time_ms": r["avg_time_ms"],
                "time_ms_ci": 1.96 * np.std(r["times"]) / np.sqrt(n),
                "miss_pct": 100.0 * p_miss,
                "miss_pct_ci": 100.0 * 1.96 * np.sqrt(p_miss * (1 - p_miss) / n),
                "ack_losses_per_ep": r["avg_ack_losses_per_ep"],
                "edge_pct": 100.0 * ac[0] / tot,
                "opt_pct": 100.0 * ac[1] / tot,
                "cons_pct": 100.0 * ac[2] / tot,
                "unseen_pct": 100.0 * r["unseen_state_rate"],
                "reward": float(np.mean(r["rewards"])),
            })

    df = pd.DataFrame(rows)
    df.to_csv("deadline_sweep_results.csv", index=False)

    # ---------- 3) PLOTS ----------
    plot_vs_deadline(
        df, "miss_pct", "Deadline Miss Rate (%)",
        "cmp_miss_by_deadline.png", VARIANTS,
    )
    plot_vs_deadline(
        df, "time_ms", "Avg Completion Time (ms)",
        "cmp_time_by_deadline.png", VARIANTS, deadline_line=True,
    )
    plot_vs_deadline(
        df, "energy", "Avg Energy (J)",
        "cmp_energy_by_deadline.png", VARIANTS,
    )
    plot_vs_deadline(
        df, "ack_losses_per_ep", "Avg ACK Losses per Episode",
        "cmp_ack_loss_by_deadline.png", VARIANTS,
    )

    # ---------- 4) SUMMARY ----------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 100)
    print("SUMMARY (full table saved to deadline_sweep_results.csv)")
    print("=" * 100)
    for name in VARIANTS:
        sub = df[df["variant"] == name]
        print(f"\n{name}")
        print(f"{'D (ms)':<8} {'Energy (J)':<11} {'Time (ms)':<17} "
              f"{'DL-miss %':<15} {'ACK/ep':<7} {'E/O/C %':<16} "
              f"{'Unseen%':<8} {'Reward':<8}")
        for _, row in sub.iterrows():
            t_str = f"{row.time_ms:.1f}+/-{row.time_ms_ci:.1f}"
            m_str = f"{row.miss_pct:.1f}+/-{row.miss_pct_ci:.1f}"
            mix = f"{row.edge_pct:.0f}/{row.opt_pct:.0f}/{row.cons_pct:.0f}"
            print(f"{row.deadline_ms:<8.0f} {row.energy:<11.3f} "
                  f"{t_str:<17} {m_str:<15} {row.ack_losses_per_ep:<7.2f} "
                  f"{mix:<16} {row.unseen_pct:<8.1f} {row.reward:<8.1f}")

    piv = df.pivot_table(index="deadline_ms", columns="variant", values="miss_pct")
    baselines = [v for v in VARIANTS if v != "Ternary A2C"]
    piv["best_baseline"] = piv[baselines].min(axis=1)
    piv["ternary_minus_best"] = piv["Ternary A2C"] - piv["best_baseline"]
    print("\nTernary miss% minus best baseline miss% (negative = ternary better)")
    print(piv["ternary_minus_best"].round(1).to_string())
    print("=" * 100)