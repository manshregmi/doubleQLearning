import numpy as np
import random
import pickle
import os
from typing import Tuple, Any
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


class DoubleQLearningAgent:
    """
    Tabular Double Q-learning agent for layer-by-layer offloading.

    Key improvements over the original:
    - optimistic initialization
    - count-based intrinsic exploration bonus: bonus = beta / sqrt(1 + N(s,a))
    - visit counts tracking
    - slow epsilon decay and adaptive epsilon boost when training stagnates
    - `notify_episode_end(episode_reward)` to update epsilon based on whole-episode reward
      (preferred) — fallback per-step update is also provided to avoid silent stagnation
    """

    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test: bool,
        alpha: float = 0.05,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ):
        """
        Args:
            profiling_data: ProfilingData instance (your environment metadata).
            is_test: If True, do not apply stochastic decisions nor update Q-tables.
            alpha: learning rate
            gamma: discount factor
            epsilon: initial epsilon for epsilon-greedy
        """
        self.profiling = profiling_data
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.is_test = is_test

        # bookkeeping for diagnostics you had previously
        self.optimum_action_layer_count = [0] * 7
        self.last_layer_not_optimum = 0

        # Q-tables (tabular double Q)
        self.Q1 = {}
        self.Q2 = {}

        # Simulator
        self.simulator = CloudEdgeSimulator(profiling_data)

        # --- Discretization bins (same defaults you had) ---
        self.bandwidth_bins = np.linspace(1, 15, 60)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        self.surplus_bins = np.linspace(-25, 25, 25)

        # --- Exploration & visit-counts ---
        self.visit_counts = {}  # key: (s_key, a_key) -> int
        self.beta = 0.5  # exploration bonus coefficient (tuneable)

        # --- Adaptive epsilon scheduling ---
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # slow decay
        self.epsilon_boost = 0.1  # boost when stuck
        self.stagnant_limit = 5000  # episodes without improvement to consider "stuck"

        # Episode-level tracking (use notify_episode_end for correct behavior)
        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0

        # Optimistic init value for missing Q entries
        self.optimistic_init_value = 1.0

    # ---------------------------
    # Utility & discretization
    # ---------------------------
    def _discretize(self, value: float, bins: np.ndarray) -> float:
        idx = np.digitize([value], bins, right=True)[0] - 1
        idx = max(0, min(idx, len(bins) - 1))
        return float(bins[idx])

    def _action_to_key(self, action: np.ndarray) -> Tuple:
        """Convert action array to a unique, hashable key (tuple of ints)."""
        return tuple(int(x) for x in action[:, 1].tolist())

    def _state_to_key(self, state: Tuple[Any, ...]) -> Tuple:
        """
        Discretize continuous state components and return a hashable key.

        state is expected: (bandwidth, cloud_pending_ms, layer, prev_action, surplus, negative_surplus_count)
        """
        bw, ctime, layer, prev_action, surplus, negative_surplus_count = state
        bw_disc = self._discretize(float(bw), self.bandwidth_bins)
        ctime_disc = self._discretize(float(ctime), self.cloudtime_bins)
        surplus_disc = self._discretize(float(surplus), self.surplus_bins)
        layer_i = int(layer)
        neg_count_i = int(negative_surplus_count)
        prev_key = None if prev_action is None else self._action_to_key(prev_action)
        return (bw_disc, ctime_disc, layer_i, surplus_disc, neg_count_i, prev_key)

    # ---------------------------
    # Action generation helpers
    # ---------------------------
    def _get_possible_actions(self, layer_idx: int):
        """Generate feasible offloading action patterns for a given layer."""
        nodes = self.profiling.get_num_nodes(layer_idx)

        # Terminal layer → fixed local execution
        if layer_idx == (len(self.profiling.layers) - 1):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            a[:, 1] = 0
            return [a]

        # Limit pattern explosion for large node counts
        max_patterns = 64
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

    def _argmax_with_tiebreak(self, values):
        """Return index of max with random tie-breaking."""
        if len(values) == 0:
            return 0
        # If all equal, random pick among them (only in training)
        if np.allclose(values, values[0]):
            return random.randint(0, len(values) - 1)
        max_val = max(values)
        candidates = [i for i, v in enumerate(values) if np.isclose(v, max_val)]
        return random.choice(candidates)

    # ---------------------------
    # Visit counts & bonus
    # ---------------------------
    def _increment_visit(self, s_key, a_key):
        key = (s_key, a_key)
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

    def _exploration_bonus(self, s_key, a_key) -> float:
        count = self.visit_counts.get((s_key, a_key), 0)
        return self.beta / np.sqrt(1.0 + count)

    # ---------------------------
    # Action selection
    # ---------------------------
    def _ensure_q_entry(self, full_key):
        """Ensure optimistic initialization for a state-action key in both Q-tables."""
        if full_key not in self.Q1:
            self.Q1[full_key] = self.optimistic_init_value
        if full_key not in self.Q2:
            self.Q2[full_key] = self.optimistic_init_value

    def choose_action(self, state: Tuple[Any, ...]):
        """
        Epsilon-greedy selection based on Q1 + Q2 with count-based exploration bonus.

        When is_test=True: deterministic choice based purely on argmax(Q1+Q2) with NO epsilon and NO random tie-break.
        """
        layer = int(state[2])
        actions = self._get_possible_actions(layer)
        s_key = self._state_to_key(state)

        # Exploration allowed for all non-terminal layers only when training
        if (not self.is_test) and (random.random() < self.epsilon) and (layer < len(self.profiling.layers) - 1):
            return random.choice(actions)

        # Compute Q-values from both tables and add bonus
        q_vals = []
        for a in actions:
            a_key = self._action_to_key(a)
            full_key = (s_key, a_key)

            # Ensure optimistic initialization
            self._ensure_q_entry(full_key)

            q_val = self.Q1[full_key] + self.Q2[full_key]
            bonus = 0.0
            # Only add bonus during training (not during test)
            if not self.is_test:
                bonus = self._exploration_bonus(s_key, a_key)
            q_vals.append(q_val + bonus)

        # Deterministic when testing: pure argmax of Q1+Q2 (no bonus, no tie-break randomness)
        if self.is_test:
            # build list of pure q sums (no bonus) to avoid tie randomness
            pure_qs = []
            for a in actions:
                a_key = self._action_to_key(a)
                full_key = (s_key, a_key)
                self._ensure_q_entry(full_key)
                pure_qs.append(self.Q1[full_key] + self.Q2[full_key])
            chosen_idx = int(np.argmax(pure_qs))
            return actions[chosen_idx]

        # Training mode: use argmax with tie-break randomness on q_vals (which include bonus)
        chosen_idx = self._argmax_with_tiebreak(q_vals)
        return actions[chosen_idx]

    # ---------------------------
    # Optimum action counting (kept for backward compatibility)
    # ---------------------------
    def is_optimum_action(self, chosen_action: np.ndarray, layer_idx: int):
        optimum_actions = [
            np.array([[0, 0]]),
            np.array([[1, 1]]),
            np.array([[2, 1]]),
            np.array([[3, 1], [3, 1], [3, 1]]),
            np.array([[4, 1], [4, 1], [4, 1]]),
            np.array([[5, 1], [5, 1], [5, 1]]),
            np.array([[6, 0]]),
        ]

        if layer_idx < 0 or layer_idx >= len(optimum_actions):
            print(f"⚠️ Layer {layer_idx} not in optimum action list.")
            return False

        optimum = optimum_actions[layer_idx]
        same_shape = chosen_action.shape == optimum.shape
        same_values = np.array_equal(chosen_action, optimum)

        if same_shape and same_values:
            self.optimum_action_layer_count[layer_idx] += 1
            return True
        elif layer_idx == len(self.profiling.layers) - 1:
            self.last_layer_not_optimum += 1
            return False
        return False

    # ---------------------------
    # Training step
    # ---------------------------
    def train(self, current_state: Tuple[Any, ...]):
        """
        Perform one environment step.

        If is_test=True: do NOT update Q-tables, do NOT apply discounting (returns purely observational).
        Returns:
            (action, reward, next_state, terminal, energy, completion_time, next_bandwidth, surplus, fractional_deadline)
        """
        # Choose action (deterministic if test)
        action = self.choose_action(current_state)
        self.is_optimum_action(action, int(current_state[2]))

        # Count visit for exploration bonus (training only)
        if not self.is_test:
            self._increment_visit(self._state_to_key(current_state), self._action_to_key(action))

        next_state_cloud_processing = self.simulator.get_next_state_cloud_waiting_time(
            next_layer=(int(current_state[2]))
            if ((int(current_state[2]) + 1) < len(self.profiling.layers))
            else int(current_state[2]),
            current_action=action,
            isAllCloud=False,
        )

        # Simulator step(s)
        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state,
            current_action=action,
            cloud_pending_ms=next_state_cloud_processing,
        )

        # Reward computation
        reward, surplus, negative_surplus_count, fractional_deadline = self.simulator.calculate_reward(
            int(current_state[2]),
            energy,
            completion_time_s,
            current_state[4],
            current_state[5],
            isA2C=False,
        )

        # Next state
        next_state, terminal, _ = self.simulator.get_next_state(
            current_state,
            action,
            surplus,
            negative_surplus_count,
            new_cloud_pending=next_state_cloud_processing,
        )

        # No Q-updates during testing
        if self.is_test:
            return (
                action,
                reward,
                next_state,
                terminal,
                energy,
                completion_time_s,
                next_state[0],
                surplus,
                fractional_deadline,
                negative_surplus_count
            )

        # -------------------
        # Double Q-learning update (training mode)
        # -------------------
        cur_key = (self._state_to_key(current_state), self._action_to_key(action))

        # Randomly pick which table to update
        if random.random() < 0.5:
            q_table, q_eval = self.Q1, self.Q2
        else:
            q_table, q_eval = self.Q2, self.Q1

        # Ensure optimistic initialization for current key
        if cur_key not in q_table:
            q_table[cur_key] = self.optimistic_init_value
        if cur_key not in q_eval:
            q_eval[cur_key] = self.optimistic_init_value

        if terminal:
            target = reward
        else:
            next_actions = self._get_possible_actions(int(next_state[2]))
            # choose best next action according to the q_table being updated
            best_next_action = max(
                next_actions,
                key=lambda a: q_table.get((self._state_to_key(next_state), self._action_to_key(a)), self.optimistic_init_value),
            )
            eval_value = q_eval.get(
                (self._state_to_key(next_state), self._action_to_key(best_next_action)),
                self.optimistic_init_value,
            )
            target = reward + (self.gamma * eval_value)

        old_value = q_table.get(cur_key, self.optimistic_init_value)
        q_table[cur_key] = old_value + self.alpha * (target - old_value)

        # ---- NOTE: episode-level epsilon adjustments should be done via notify_episode_end()
        # Fallback: apply a tiny per-step decay so epsilon doesn't remain stuck if user never calls notify
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Return same tuple as before
        return (
            action,
            reward,
            next_state,
            terminal,
            energy,
            completion_time_s,
            next_state[0],
            surplus,
            fractional_deadline,
            negative_surplus_count,
        )

    # ---------------------------
    # Episode-level API for adaptive epsilon
    # ---------------------------
    def notify_episode_end(self, episode_reward: float):
        """
        More aggressive epsilon adjustment for faster learning.
        """
        if episode_reward > self.best_episode_reward + 1e-9:
            self.best_episode_reward = episode_reward
            self.episodes_since_improvement = 0
            # Good reward: reduce epsilon slowly
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)
        else:
            self.episodes_since_improvement += 1
            
            # If stuck for too long, boost exploration more aggressively
            if self.episodes_since_improvement >= self.stagnant_limit // 2:  # More frequent boost
                self.epsilon = min(0.8, self.epsilon + 0.3)  # Boost more
                self.episodes_since_improvement = 0
                print(f"  Epsilon boosted to {self.epsilon:.3f} due to stagnation")
            else:
                # Normal decay
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.999)

    def _update_trajectory_q_values(self, trajectory):
        """
        Update Q-values using Monte Carlo returns from trajectory.
        """
        if not trajectory:
            return
        
        # Calculate Monte Carlo returns
        returns = 0.0
        
        for i in reversed(range(len(trajectory))):
            step = trajectory[i]
            reward = step['reward']
            
            # Calculate return
            returns = reward + self.gamma * returns
            
            # Clip returns to prevent explosion
            returns = np.clip(returns, -1000.0, 1000.0)
            
            state_key = step['state_key']
            action_key = step['action_key']
            full_key = (state_key, action_key)
            
            # Double Q-learning update
            if random.random() < 0.5:
                q_table, q_eval = self.Q1, self.Q2
            else:
                q_table, q_eval = self.Q2, self.Q1
            
            # Ensure optimistic initialization
            if full_key not in q_table:
                q_table[full_key] = self.optimistic_init_value
            if full_key not in q_eval:
                q_eval[full_key] = self.optimistic_init_value
            
            # Get current Q-value
            current_q = q_table[full_key]
            
            # Monte Carlo update
            new_q = current_q + self.alpha * (returns - current_q)
            
            # CRITICAL: Clip Q-values to prevent explosion
            q_table[full_key] = np.clip(new_q, -100.0, 100.0)
            
            # Update visit counts
            self._increment_visit(state_key, action_key)

    # ---------------------------
    # Persistence
    # ---------------------------
    def save_qtables(self, filename: str = "q_tables.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump((self.Q1, self.Q2, self.visit_counts), f)
        except Exception as e:
            print(f"Error saving Q-tables: {e}")

    def load_qtables(self, filename: str = "q_tables.pkl"):
        try:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, tuple) and len(data) == 3:
                        self.Q1, self.Q2, self.visit_counts = data
                    else:
                        # backwards compatibility: older file with only Q1,Q2
                        self.Q1, self.Q2 = data
                        self.visit_counts = {}
                print(
                    f"Q-tables found and loaded from {filename}, q1 size: {len(self.Q1)}, q2 size: {len(self.Q2)}"
                )
                return True
            else:
                print(f"Q-table file not found at {filename}. Starting fresh.")
                return False
        except Exception as e:
            print(f"Error loading Q-tables: {e}")
            return False
