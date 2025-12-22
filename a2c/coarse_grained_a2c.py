import numpy as np
import random
import pickle
import os
from collections import defaultdict
from typing import Tuple, Any
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator

# -------------------------
# Episode-Level Tabular Actor-Critic
# -------------------------
class EpisodeActorCriticAgent:
    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test=False,
        alpha=0.05,
        gamma=0.95,
        temperature=1.0,
    ):
        self.profiling = profiling_data
        self.is_test = is_test
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature  # acts like epsilon

        # Tabular policy and value tables
        self.policy_table = {}  # key: joint_action -> probability
        self.value_table = {}   # key: state_key -> value

        # Exploration & visit count
        self.visit_counts = {}
        self.beta = 0.5  # exploration bonus

        # Temperature boost logic
        self.temperature_min = 0.05
        self.temperature_decay = 0.9995
        self.temperature_boost = 0.3
        self.stagnant_limit = 5000
        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0

        self.simulator = CloudEdgeSimulator(profiling_data)

        # For discretization of state (same as double-Q)
        self.bandwidth_bins = np.linspace(1, 15, 15)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        self.surplus_bins = np.linspace(-25, 25, 25)

        self.optimistic_init_value = 1.0

    # -------------------------
    # State / Action keys
    # -------------------------
    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        idx = max(0, min(idx, len(bins) - 1))
        return float(bins[idx])

    def _state_to_key(self, state):
        bw, ctime, layer, prev_action, surplus, neg_count = state
        bw_disc = self._discretize(float(bw), self.bandwidth_bins)
        ctime_disc = self._discretize(float(ctime), self.cloudtime_bins)
        surplus_disc = self._discretize(float(surplus), self.surplus_bins)
        return (bw_disc, ctime_disc, surplus_disc, int(neg_count))

    def _action_to_key(self, joint_action):
        # Joint action is a list of layer-level arrays
        return tuple(tuple(int(x) for x in layer[:, 1]) for layer in joint_action)

    # -------------------------
    # Action Space
    # -------------------------
    def _get_possible_actions_per_layer(self, layer_idx, max_patterns=8):
        nodes = self.profiling.get_num_nodes(layer_idx)
        all_patterns = list(range(2 ** nodes))
        if len(all_patterns) > max_patterns:
            patterns = random.sample(all_patterns, max_patterns)
        else:
            patterns = all_patterns
        actions = []
        for pattern in patterns:
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            for i in range(nodes):
                a[i, 1] = (pattern >> i) & 1
            actions.append(a)
        return actions

    def build_joint_action_space(self, max_per_layer=4):
        joint_actions = [[]]
        for layer in range(len(self.profiling.layers)):
            actions = self._get_possible_actions_per_layer(layer, max_per_layer)
            joint_actions = [prev + [a] for prev in joint_actions for a in actions]
        return joint_actions

    # -------------------------
    # Policy selection
    # -------------------------
    def choose_joint_action(self, state_key, joint_actions):
        if self.is_test or random.random() > self.temperature:
            # deterministic: pick best value
            best_val = -np.inf
            best_action = joint_actions[0]
            for a in joint_actions:
                key = self._action_to_key(a)
                val = self.policy_table.get(key, self.optimistic_init_value)
                if val > best_val:
                    best_val = val
                    best_action = a
            return best_action

        # exploration: weighted sampling with temperature
        vals = []
        for a in joint_actions:
            key = self._action_to_key(a)
            vals.append(self.policy_table.get(key, self.optimistic_init_value))
        vals = np.array(vals)
        probs = np.exp(vals / max(self.temperature, 1e-5))
        probs /= probs.sum()
        chosen_idx = np.random.choice(len(joint_actions), p=probs)
        return joint_actions[chosen_idx]

    # -------------------------
    # Episode rollout
    # -------------------------
    def rollout_episode(self, joint_action):
        cloud_time = 0.0
        state = (self.profiling.bandwidth, cloud_time, 0, None, 0, 0)
        total_energy = 0.0
        total_time = 0.0
        surplus = 0.0
        negative_surplus_count = 0

        trajectory = []

        for layer_idx, action in enumerate(joint_action):
            next_cloud = self.simulator.get_next_state_cloud_waiting_time(
                next_layer=layer_idx + 1,
                current_action=action,
                isAllCloud=False,
            )
            energy, completion_time_s = self.simulator.compute_energy_and_time(
                state, action, next_cloud
            )

            r, surplus, negative_surplus_count, _ = self.simulator.calculate_reward(
                layer_idx,
                energy,
                completion_time_s,
                surplus,
                negative_surplus_count,
                isA2C=True,
            )

            state, _, _ = self.simulator.get_next_state(
                state, action, surplus, negative_surplus_count, next_cloud
            )

            trajectory.append({
                "state_key": self._state_to_key(state),
                "action_key": self._action_to_key([action]),
                "reward": r,
                "energy": energy,
                "completion_time": completion_time_s * 1000.0,
            })

            total_energy += energy
            total_time += completion_time_s * 1000.0

        return trajectory, total_energy, total_time

    # -------------------------
    # Monte Carlo Actor-Critic update
    # -------------------------
    def update_from_trajectory(self, trajectory):
        G = 0.0
        for step in reversed(trajectory):
            G = step["reward"] + self.gamma * G
            state_key = step["state_key"]
            action_key = step["action_key"]

            V = self.value_table.get(state_key, 0.0)
            advantage = G - V
            self.value_table[state_key] = V + self.alpha * advantage

            # Policy update (simple tabular gradient)
            old_val = self.policy_table.get(action_key, self.optimistic_init_value)
            self.policy_table[action_key] = old_val + self.alpha * advantage

            # update visit counts
            self.visit_counts[action_key] = self.visit_counts.get(action_key, 0) + 1

    # -------------------------
    # Temperature boost / decay
    # -------------------------
    def notify_episode_end(self, total_reward):
        if total_reward > self.best_episode_reward + 1e-9:
            self.best_episode_reward = total_reward
            self.episodes_since_improvement = 0
            self.temperature = max(self.temperature_min, self.temperature * 0.995)
        else:
            self.episodes_since_improvement += 1
            if self.episodes_since_improvement >= self.stagnant_limit // 2:
                self.temperature = min(0.8, self.temperature + self.temperature_boost)
                self.episodes_since_improvement = 0
                print(f"🔥 Temperature boosted to {self.temperature:.2f}")
            else:
                self.temperature = max(self.temperature_min, self.temperature * 0.999)

    # -------------------------
    # Persistence
    # -------------------------
    def save_tables(self, filename="a2c_tables_cg.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump((self.policy_table, self.value_table, self.visit_counts), f)
        except Exception as e:
            print(f"Error saving tables: {e}")

    def load_tables(self, filename="a2c_tables_cg.pkl"):
        try:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, tuple) and len(data) == 3:
                        self.policy_table, self.value_table, self.visit_counts = data
                print(f"A2C tables loaded from {filename}")
        except Exception as e:
            print(f"Error loading tables: {e}")

# -------------------------
# Runner Function
# -------------------------
def run_a2c_episode_level(profiling_data: ProfilingData, episodes=1000, max_per_layer=4, is_test=False):
    agent = EpisodeActorCriticAgent(profiling_data, is_test=is_test)
    agent.load_tables()

    edge_energy = []
    completion_time = []
    rewards = []
    cumulative_rewards = []
    deadline_missed = 0

    for ep in range(episodes):
        joint_actions = agent.build_joint_action_space(max_per_layer)
        state_key = agent._state_to_key((profiling_data.bandwidth, 0, 0, None, 0, 0))
        action = agent.choose_joint_action(state_key, joint_actions)
        trajectory, total_energy, total_time = agent.rollout_episode(action)

        # Apply same reward shaping as double Q
        modified_total_reward = sum(step["reward"] for step in trajectory)

        if not is_test:
            agent.update_from_trajectory(trajectory)
            agent.notify_episode_end(modified_total_reward)

        edge_energy.append(total_energy)
        completion_time.append(total_time)
        if total_time > profiling_data.deadline:
            deadline_missed += 1 
        rewards.append(sum(step["reward"] for step in trajectory))
        cumulative_rewards.append(modified_total_reward)

        if ep < 5 or (ep + 1) % 50 == 0:
            status = "MISS" if total_time > profiling_data.deadline else "MET "
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Time={total_time:.1f}ms ({status}) | "
                f"Energy={total_energy:.2f}J | "
                f"Reward={modified_total_reward:.2f} | "
                f"Temp={agent.temperature:.3f}"
            )

    agent.save_tables()

    print("\nSimulation summary:")
    print(f"Average Energy: {np.mean(edge_energy):.2f} J")
    print(f"Average Completion Time: {np.mean(completion_time):.1f} ms")
    print(f"Average Reward: {np.mean(cumulative_rewards):.2f}")
    

    return np.mean(edge_energy), np.mean(completion_time), deadline_missed
