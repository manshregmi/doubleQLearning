import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from profiling.initialize_profiling import get_profiling_data
from simulator.a2c_simulator import run_a2c_simulation


# ---------- CONFIGURATION ----------
TRACE_CSV_PATH = ("C:\\Users\\SIU856613348\\OneDrive - Southern Illinois University\\"
                  "doubleQLearning\\simulator\\data\\bw_data_trace.csv")

N_DEVICES = 8

# Deadlines relative to the all-local completion time T_local:
#   < 1.0 -> infeasible without offloading (tight)
#   = 1.0 -> just feasible locally
#   > 1.0 -> loose
DEADLINE_FACTORS = [0.8, 1.0, 1.25, 1.5]
# Explicit deadlines in ms (overrides the factors). Set to None to use factors.
DEADLINES_MS_OVERRIDE = [400, 450, 500, 550, 600, 650, 700]

# Timeout as a fraction of the deadline: tau = ratio * D
TIMEOUT_RATIOS = [0.05, 0.10, 0.15, 0.20]

# Training settings
TRAIN_EPISODES = 3000000   # 28 (D, ratio) contexts for the ternary agent -> needs more data
EVAL_EPISODES = 2000       # per (variant, D, ratio) cell
MAX_STEPS = 10
PACKET_LOSS = 0.10
SEED = 42

# Set True to retrain; False reuses saved tables and only evaluates
DO_TRAIN = True

# ---------- AGENT VARIANTS ----------
# modes: 0 = user (edge), 1 = optimistic cloudlet, 2 = conservative cloudlet
VARIANTS = {
    "Ternary A2C": {
        "modes": (0, 1, 2),
        "threshold_aware": True,     # sees deadline AND timeout ratio
        "model_path": "a2c_ternary_dl.pkl",
        "color": "#1f77b4", "marker": "o",
    },
    "Binary A2C (Optimistic)": {
        "modes": (0, 1),
        "threshold_aware": False,    # sees deadline only
        "model_path": "a2c_binary_optimistic_dl.pkl",
        "color": "#ff7f0e", "marker": "s",
    },
    "Binary A2C (Conservative)": {
        "modes": (0, 2),
        "threshold_aware": False,    # sees deadline only
        "model_path": "a2c_binary_conservative_dl.pkl",
        "color": "#2ca02c", "marker": "^",
    },
}

# ---------- PLOTTING STYLE ----------
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
    """Deterministic all-local completion time, using the simulator's
    parallelism model (layers 3 and 5 run their nodes in parallel)."""
    total = 0.0
    for l in range(len(p.layers)):
        t = [p.get_node_edge_time(l, i) for i in range(p.get_num_nodes(l))]
        total += max(t) if l in (3, 5) else sum(t)
    return total


def plot_grid(df, metric, ylabel, filename, deadlines, deadline_line=False):
    """One panel per deadline; x = timeout ratio; one line per variant."""
    ncols = min(4, len(deadlines))
    nrows = int(np.ceil(len(deadlines) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows),
                             sharey=(metric != "time_ms"), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(deadlines):]:
        ax.set_visible(False)
    for ax, D in zip(axes, deadlines):
        sub = df[df["deadline_ms"] == D]
        for name, cfg in VARIANTS.items():
            s = sub[sub["variant"] == name].sort_values("ratio")
            ci_col = metric + "_ci"
            ax.errorbar(s["ratio"] * 100, s[metric],
                        yerr=s[ci_col] if ci_col in s else None,
                        marker=cfg["marker"], color=cfg["color"], label=name, capsize=3)
        if deadline_line:
            ax.axhline(D, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"D = {D:.0f} ms")
        ax.set_xlabel("Timeout (% of deadline)")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    profiling_data = get_profiling_data(500, N_DEVICES)  # deadline is overridden per episode

    t_local = all_local_time_ms(profiling_data)
    if DEADLINES_MS_OVERRIDE:
        DEADLINES_MS = [float(d) for d in DEADLINES_MS_OVERRIDE]
    else:
        DEADLINES_MS = [float(round(f * t_local)) for f in DEADLINE_FACTORS]

    print("\n" + "=" * 70)
    print(f"All-local completion time T_local = {t_local:.1f} ms")
    print(f"Deadlines (ms): {DEADLINES_MS}")
    print(f"Timeout ratios: {TIMEOUT_RATIOS}")
    print("Timeout grid (ms):")
    for D in DEADLINES_MS:
        print(f"  D={D:>6.0f}: " + ", ".join(f"{r*D:.0f}" for r in TIMEOUT_RATIOS))
    print("=" * 70)

    common = dict(
        max_steps=MAX_STEPS,
        trace_csv_path=TRACE_CSV_PATH,
        packet_loss_prob=PACKET_LOSS,
        deadline_levels=DEADLINES_MS,
        threshold_ratios=TIMEOUT_RATIOS,
    )

    # ---------- 1) TRAIN each variant over all (deadline, ratio) contexts ----------
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
                threshold_aware=cfg["threshold_aware"],
                deadline_ms=None,       # sampled per episode
                threshold_ratio=None,   # sampled per episode
                label=name,
                **common,
            )

    # ---------- 2) EVALUATE on the full (deadline x ratio) grid ----------
    rows = []
    for name, cfg in VARIANTS.items():
        for D in DEADLINES_MS:
            for rho in TIMEOUT_RATIOS:
                r = run_a2c_simulation(
                    profiling_data,
                    episodes=EVAL_EPISODES,
                    is_test=True,          # greedy policy, no updates, no save
                    model_path=cfg["model_path"],
                    seed=SEED,             # paired evaluation across variants
                    action_modes=cfg["modes"],
                    threshold_aware=cfg["threshold_aware"],
                    deadline_ms=D,
                    threshold_ratio=rho,
                    label=f"{name} @ D={D:.0f} tau={rho*D:.0f}",
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
                    "ratio": rho,
                    "timeout_ms": rho * D,
                    "energy": r["avg_energy"],
                    "energy_ci": 1.96 * np.std(r["energies"]) / np.sqrt(n),
                    "time_ms": r["avg_time_ms"],
                    "time_ms_ci": 1.96 * np.std(r["times"]) / np.sqrt(n),
                    "miss_pct": 100.0 * p_miss,
                    "miss_pct_ci": 100.0 * 1.96 * np.sqrt(p_miss * (1 - p_miss) / n),
                    "timeouts_per_ep": r["avg_timeouts_per_ep"],
                    "edge_pct": 100.0 * ac[0] / tot,
                    "opt_pct": 100.0 * ac[1] / tot,
                    "cons_pct": 100.0 * ac[2] / tot,
                    "unseen_pct": 100.0 * r["unseen_state_rate"],
                    "reward": float(np.mean(r["rewards"])),
                })

    df = pd.DataFrame(rows)
    df.to_csv("deadline_sweep_results.csv", index=False)

    # ---------- 3) PLOTS ----------
    plot_grid(df, "miss_pct", "Deadline Miss Rate (%)", "cmp_miss_by_deadline.png", DEADLINES_MS)
    plot_grid(df, "time_ms", "Avg Completion Time (ms)", "cmp_time_by_deadline.png",
              DEADLINES_MS, deadline_line=True)
    plot_grid(df, "energy", "Avg Energy (J)", "cmp_energy_by_deadline.png", DEADLINES_MS)

    # ---------- 4) SUMMARY ----------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 100)
    print("SUMMARY (full table saved to deadline_sweep_results.csv)")
    print("=" * 100)
    for name in VARIANTS:
        sub = df[df["variant"] == name]
        print(f"\n{name}")
        print(f"{'D (ms)':<8} {'tau (ms)':<9} {'Energy (J)':<11} {'Time (ms)':<17} "
              f"{'DL-miss %':<15} {'TO/ep':<7} {'E/O/C %':<12} {'Unseen%':<8} {'Reward':<8}")
        for _, row in sub.iterrows():
            t_str = f"{row.time_ms:.1f}+/-{row.time_ms_ci:.1f}"
            m_str = f"{row.miss_pct:.1f}+/-{row.miss_pct_ci:.1f}"
            mix = f"{row.edge_pct:.0f}/{row.opt_pct:.0f}/{row.cons_pct:.0f}"
            print(f"{row.deadline_ms:<8.0f} {row.timeout_ms:<9.0f} {row.energy:<11.3f} "
                  f"{t_str:<17} {m_str:<15} {row.timeouts_per_ep:<7.2f} {mix:<12} "
                  f"{row.unseen_pct:<8.1f} {row.reward:<8.1f}")

    # Ternary advantage: miss% (ternary) - min(miss% of baselines). Negative = ternary better.
    piv = df.pivot_table(index=["deadline_ms", "ratio"], columns="variant", values="miss_pct")
    baselines = [v for v in VARIANTS if v != "Ternary A2C"]
    piv["best_baseline"] = piv[baselines].min(axis=1)
    piv["ternary_minus_best"] = piv["Ternary A2C"] - piv["best_baseline"]
    adv = piv["ternary_minus_best"].unstack("ratio")
    adv.columns = [f"{c*100:.0f}%" for c in adv.columns]
    print("\nTernary miss% minus best baseline miss% (negative = ternary better)")
    print("rows: deadline (ms), columns: timeout as % of deadline")
    print(adv.round(1).to_string())
    print("=" * 100)