from ppo.ppo_agent import PPOAgent, _nodes_in_layer
from profiling.profile import ProfilingData
import numpy as np
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd


def run_ppo_simulation(
    profiling_data,
    episodes=1000,
    max_steps=10,
    is_test=False,
    trace_csv_path=None,
    model_path="ppo_ternary.pt",
    seed=None,
    action_modes=(0, 1, 2),
    deadline_levels=(500.0,),
    deadline_ms=None,
    label="PPO",
    verbose=False,
):
    """PPO trainer/evaluator. Episode-level rollouts, retroactive Eq. 11."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    deadline_levels = tuple(float(d) for d in deadline_levels)

    agent = PPOAgent(deadline_levels=deadline_levels, is_test=is_test)
    agent.load(model_path)

    edge_energy, completion_time, rewards, episode_modified_rewards = [], [], [], []
    per_episode_ack_losses = []
    action_counts = {0: 0, 1: 0, 2: 0}
    deadline_missed_count = 0
    deadline_met_count = 0

    start_time = time.time()
    d_str = f"{deadline_ms:.0f}ms" if deadline_ms is not None else f"sampled {deadline_levels}"
    print(f"Running {label} (is_test={is_test}, modes={tuple(action_modes)})")
    print(f"  Episodes: {episodes} | deadline: {d_str}")

    PHI = 1000.0

    for ep in range(episodes):
        D = float(deadline_ms) if deadline_ms is not None else random.choice(deadline_levels)
        profiling_data.deadline = D
        d_idx = int(np.argmin(np.abs(np.asarray(deadline_levels) - D)))

        agent.simulator = getattr(agent, "simulator", None)
        # Agents don't own the simulator; the runner drives it.
        # Use a fresh simulator instance shared with A2C.
        from simulator.simulator import CloudEdgeSimulator
        sim = CloudEdgeSimulator(profiling_data, trace_csv_path=trace_csv_path)
        sim.verbose = verbose
        sim.reset_episode_time()

        bw = sim.current_bandwidth
        rtt = sim.current_rtt_ms
        current_state = (bw, rtt, 0.0, 0, None, 0.0, 0, 0, d_idx, 0)

        total_energy = 0.0
        total_time = 0.0
        trajectory = []
        episode_ack_losses = 0
        episode_action_counts = {0: 0, 1: 0, 2: 0}
        prev_neg = 0

        for step in range(max_steps):
            layer = int(current_state[3])
            n_real = _nodes_in_layer(layer)
            modes, logp, value = agent.choose_action(current_state, n_real)

            # Build the action matrix the simulator expects.
            action = np.zeros((n_real, 2), dtype=int)
            action[:, 0] = layer
            action[:, 1] = modes

            next_cloud = sim.get_next_state_cloud_waiting_time(
                current_layer=layer, current_action=action, isAllCloud=False)
            energy, completion_time_s = sim.compute_energy_and_time(
                current_state=current_state, current_action=action,
                cloud_pending_ms=next_cloud)
            reward, surplus, neg_count, fractional_deadline = sim.calculate_reward(
                layer, energy, completion_time_s,
                current_state[5], current_state[6], isA2C=True)
            next_state, terminal = sim.get_next_state(
                current_state, action, new_cloud_pending=next_cloud,
                surplus=surplus, neg_count=neg_count)

            for node_mode in action[:, 1]:
                m = int(node_mode)
                episode_action_counts[m] += 1
                action_counts[m] += 1
            if sim.last_ack_loss_occurred:
                episode_ack_losses += 1

            neg_delta = int(neg_count) - int(prev_neg)
            prev_neg = int(neg_count)

            trajectory.append({
                "state": current_state,
                "modes": np.asarray(modes, dtype=np.int64),
                "logp": logp,
                "value": value,
                "reward": reward,
                "neg_delta": neg_delta,
                "done": terminal,
            })

            total_energy += energy
            total_time += completion_time_s * 1000.0
            current_state = next_state
            if terminal:
                break

        per_episode_ack_losses.append(episode_ack_losses)

        # Retroactive Eq. 11 terminal penalty.
        deadline_violated = total_time > D
        if deadline_violated:
            deadline_missed_count += 1
            penalty = PHI * max(0.0, total_time - D) / D
            violating = [i for i, s in enumerate(trajectory) if s["neg_delta"] > 0]
            if violating:
                for i in violating:
                    trajectory[i]["reward"] -= penalty
            else:
                trajectory[-1]["reward"] -= penalty
        else:
            deadline_met_count += 1

        # Scale rewards by 1/100 before buffering.
        for s in trajectory:
            s["reward_scaled"] = float(s["reward"]) * 0.01

        if not is_test:
            agent.store_episode(trajectory)
            if agent.should_update():
                agent.update()

        modified_reward = sum(s["reward"] for s in trajectory)
        edge_energy.append(total_energy)
        completion_time.append(total_time)
        rewards.append(sum(s["reward"] for s in trajectory))
        episode_modified_rewards.append(modified_reward)

        if (ep + 1) % max(1, episodes // 10) == 0 or ep == 0:
            ec, oc, cc = (episode_action_counts[0], episode_action_counts[1], episode_action_counts[2])
            print(f"[{label}] Ep {ep+1}/{episodes}: D={D:.0f}ms "
                  f"E={total_energy:.2f}J, T={total_time:.1f}ms, R={modified_reward:.1f}, "
                  f"ACK_loss={episode_ack_losses}, "
                  f"[E:{ec} O:{oc} C:{cc} (n={ec+oc+cc})]")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"{label} COMPLETE: {episodes} episodes in {elapsed:.1f}s")
    print(f"  Avg Energy: {np.mean(edge_energy):.3f} J")
    print(f"  Avg Time:   {np.mean(completion_time):.1f} ms")
    print(f"  DL-Met:     {deadline_met_count}/{episodes} "
          f"({100.0*deadline_met_count/episodes:.1f}%)")
    print(f"  Avg ACK losses/ep: {np.mean(per_episode_ack_losses):.2f}")
    print(f"  Per-node actions: E={action_counts[0]}, O={action_counts[1]}, C={action_counts[2]}")
    print(f"{'='*60}")

    if not is_test:
        agent.save(model_path)

    return {
        "avg_energy": float(np.mean(edge_energy)),
        "avg_time_ms": float(np.mean(completion_time)),
        "deadline_miss_rate": deadline_missed_count / episodes,
        "avg_ack_losses_per_ep": float(np.mean(per_episode_ack_losses)),
        "action_counts": dict(action_counts),
        "unseen_state_rate": 0.0,
        "n_episodes": episodes,
        "energies": edge_energy,
        "times": completion_time,
        "rewards": episode_modified_rewards,
    }