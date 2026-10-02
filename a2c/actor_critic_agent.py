import numpy as np
import random
import pickle
import os
import itertools

from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator


# Predefined action spaces (per-node modes)
#   0 = local (user/edge device)
#   1 = optimistic cloudlet (result download deferred)
#   2 = conservative cloudlet (result download immediate)
ACTION_SPACES = {
    "ternary": (0, 1, 2),
    "optimistic": (0, 1),
    "conservative": (0, 2),
}

DEFAULT_THRESHOLD_LEVELS = (25, 50, 75, 100, 150, 200, 300)


class TabularActorCriticAgent:
    def __init__(
        self,
        profiling_data,
        is_test=False,
        alpha_actor=0.02,
        alpha_critic=0.05,
        gamma=0.95,
        trace_csv_path=None,
        timeout_threshold_ms=150.0,
        packet_loss_prob=0.10,
        action_modes=(0, 1, 2),
        threshold_aware=True,
        threshold_levels=DEFAULT_THRESHOLD_LEVELS,
        threshold_sharing=True,
        verbose=False,
    ):
        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        # Threshold-aware state.
        self.threshold_aware = bool(threshold_aware)
        self.threshold_levels = np.asarray(
            sorted(threshold_levels), dtype=float
        )

        # Allowed per-node modes.
        self.action_modes = tuple(int(m) for m in action_modes)
        if 0 not in self.action_modes:
            raise ValueError("action_modes must include 0 (local execution).")

        self._action_cache = {}

        # Threshold-specific tables.
        self.policy_table = {}
        self.value_table = {}

        # Optional shared tables.
        self.threshold_sharing = (
            bool(threshold_sharing) and self.threshold_aware
        )
        self.shared_policy_table = {}
        self.shared_value_table = {}

        # Evaluation diagnostics.
        self.eval_decisions = 0
        self.eval_unseen_states = 0

        self.simulator = CloudEdgeSimulator(
            profiling_data,
            trace_csv_path=trace_csv_path,
            timeout_threshold_ms=timeout_threshold_ms,
            packet_loss_prob=packet_loss_prob,
        )
        self.simulator.verbose = verbose

        # State discretization.
        self.bandwidth_bins = np.linspace(0.5, 30, 40)
        self.rtt_bins = np.linspace(0, 100, 20)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        self.surplus_bins = np.linspace(-100, 100, 20)
        self.chain_bins = np.linspace(0, 7, 8)

        # Exploration.
        # Slower decay is intentional: with 1M episodes, 0.9995
        # reaches the minimum temperature far too early.
        self.temperature = 1.5
        self.temperature_min = 0.40
        self.temperature_max = 1.5
        self.temperature_decay = 0.99998
        self.temperature_boost = 1.25

        # Small random-action floor.
        self.epsilon_min = 0.05

        self.best_episode_reward = -1e9
        self.episodes_since_improvement = 0
        self.stagnant_limit = 5000

        self.total_episodes = 0
        self.edge_execution_counts = {}
        self.optimistic_execution_counts = {}
        self.conservative_execution_counts = {}

    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])

    def _timeout_bin(self):
        """Index of the nearest configured threshold level."""
        th = float(self.simulator.timeout_threshold_ms)
        return int(np.argmin(np.abs(self.threshold_levels - th)))

    def _state_to_key(self, state):
        """
        State:
          (bw, rtt, ctime, layer, prev_action, surplus,
           neg_count, chain_len, timeout_bin)

        FIX:
        Do NOT hash prev_action modulo 1000. The previous ternary action
        pattern is retained exactly, eliminating state collisions.
        """
        bw, rtt, ctime, layer, prev_action, surplus, neg_count, chain_len, t_bin = state

        if prev_action is None:
            prev_key = (-1,)
        else:
            prev_key = tuple(int(x) for x in prev_action)

        key = (
            self._discretize(float(bw), self.bandwidth_bins),
            self._discretize(float(rtt), self.rtt_bins),
            self._discretize(float(ctime), self.cloudtime_bins),
            int(layer),
            prev_key,
            self._discretize(float(surplus), self.surplus_bins),
            int(neg_count),
            int(chain_len),
        )

        if self.threshold_aware:
            key = key + (int(t_bin),)

        return key

    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1])

    # ------------------------------------------------------------------
    # Shared + threshold-specific decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_key(s_key):
        # If threshold is present, remove it.
        return s_key[:-1]

    def _pref(self, s_key, a_key):
        p = self.policy_table.get((s_key, a_key), 0.0)

        if self.threshold_sharing:
            p += self.shared_policy_table.get(
                (self._shared_key(s_key), a_key), 0.0
            )

        return p

    def _value(self, s_key):
        v = self.value_table.get(s_key, 0.0)

        if self.threshold_sharing:
            v += self.shared_value_table.get(
                self._shared_key(s_key), 0.0
            )

        return v

    def _is_trained(self, s_key, a_key):
        if (s_key, a_key) in self.policy_table:
            return True

        return (
            self.threshold_sharing
            and (self._shared_key(s_key), a_key)
            in self.shared_policy_table
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _get_possible_actions(self, layer_idx):
        """All per-node assignments using only this agent's allowed modes."""
        if layer_idx in self._action_cache:
            return self._action_cache[layer_idx]

        nodes = self.profiling.get_num_nodes(layer_idx)
        actions = []

        for pattern in itertools.product(self.action_modes, repeat=nodes):
            a = np.zeros((nodes, 2), dtype=int)
            a[:, 0] = layer_idx
            a[:, 1] = pattern
            actions.append(a)

        self._action_cache[layer_idx] = actions
        return actions

    def _policy_probs(self, state):
        """
        Return possible actions, action keys, and softmax probabilities.

        The same function is used for action selection and the actor update,
        so the gradient is consistent with the actual policy.
        """
        layer = int(state[3])
        actions = self._get_possible_actions(layer)
        s_key = self._state_to_key(state)

        action_keys = [self._action_to_key(a) for a in actions]

        prefs = np.asarray(
            [self._pref(s_key, ak) for ak in action_keys],
            dtype=np.float64,
        )

        temperature = max(float(self.temperature), 1e-6)

        logits = prefs / temperature
        logits -= np.max(logits)

        probs = np.exp(logits)
        denom = np.sum(probs)

        if not np.isfinite(denom) or denom <= 0:
            probs = np.ones(len(actions), dtype=np.float64) / len(actions)
        else:
            probs /= denom

        return actions, action_keys, probs, s_key

    def choose_action(self, state):
        actions, action_keys, probs, s_key = self._policy_probs(state)

        if not self.is_test and random.random() < self.epsilon_min:
            return random.choice(actions)

        if self.is_test:
            self.eval_decisions += 1

            if not any(self._is_trained(s_key, ak) for ak in action_keys):
                self.eval_unseen_states += 1

            return actions[int(np.argmax(probs))]

        return actions[np.random.choice(len(actions), p=probs)]

    # ------------------------------------------------------------------
    # Environment interaction
    # ------------------------------------------------------------------

    def train(self, current_state):
        action = self.choose_action(current_state)
        current_layer = int(current_state[3])

        next_cloud = self.simulator.get_next_state_cloud_waiting_time(
            current_layer=current_layer,
            current_action=action,
            isAllCloud=False,
        )

        # Must run BEFORE get_next_state because it sets
        # last_timeout_occurred.
        energy, completion_time_s = self.simulator.compute_energy_and_time(
            current_state=current_state,
            current_action=action,
            cloud_pending_ms=next_cloud,
        )

        reward, surplus, neg_count, fractional_deadline = (
            self.simulator.calculate_reward(
                current_layer,
                energy,
                completion_time_s,
                current_state[5],
                current_state[6],
                isA2C=True,
            )
        )

        next_state, terminal = self.simulator.get_next_state(
            current_state,
            action,
            new_cloud_pending=next_cloud,
            surplus=surplus,
            neg_count=neg_count,
        )

        self.track_action_execution(action, current_layer)

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
            neg_count,
        )

    # ------------------------------------------------------------------
    # CORRECT TABULAR ACTOR-CRITIC UPDATE
    # ------------------------------------------------------------------

    def _update_actor(self, s_key, a_key, advantage):
        """
        Policy-gradient update for a softmax policy.

        For logits theta:
            d log pi(a_t|s) / d theta_a
                = 1[a=a_t] - pi(a|s)

        Therefore every action at the state is updated:
            theta_a <- theta_a
                      + alpha * A * (1[a=a_t] - pi(a|s))

        This is different from simply adding alpha*A to the selected
        action preference.
        """
        # Reconstruct a minimal state-independent action distribution by
        # using the state key directly.
        #
        # We need the actual action keys available at this state. They are
        # reconstructed from the layer encoded in s_key.
        layer = int(s_key[3])
        actions = self._get_possible_actions(layer)
        action_keys = [self._action_to_key(a) for a in actions]

        prefs = np.asarray(
            [self._pref(s_key, ak) for ak in action_keys],
            dtype=np.float64,
        )

        temperature = max(float(self.temperature), 1e-6)
        logits = prefs / temperature
        logits -= np.max(logits)

        probs = np.exp(logits)
        probs /= max(np.sum(probs), 1e-12)

        # Small advantage clipping prevents a single highly penalized
        # episode from saturating the actor.
        A = float(np.clip(advantage, -100.0, 100.0))

        for ak, p in zip(action_keys, probs):
            indicator = 1.0 if ak == a_key else 0.0

            # Temperature is included because logits = preference / T.
            grad = A * (indicator - p) / temperature

            if self.threshold_sharing:
                ss = self._shared_key(s_key)

                # Split the total update between shared and
                # threshold-specific components.
                w = 0.5

                shared_key = (ss, ak)
                self.shared_policy_table[shared_key] = float(
                    np.clip(
                        self.shared_policy_table.get(shared_key, 0.0)
                        + w * self.alpha_actor * grad,
                        -50.0,
                        50.0,
                    )
                )

                specific_key = (s_key, ak)
                self.policy_table[specific_key] = float(
                    np.clip(
                        self.policy_table.get(specific_key, 0.0)
                        + w * self.alpha_actor * grad,
                        -50.0,
                        50.0,
                    )
                )

            else:
                key = (s_key, ak)
                self.policy_table[key] = float(
                    np.clip(
                        self.policy_table.get(key, 0.0)
                        + self.alpha_actor * grad,
                        -50.0,
                        50.0,
                    )
                )

    def update_trajectory(self, trajectory):
        """
        Monte-Carlo actor-critic update.

        Critic:
            V(s) <- V(s) + alpha_c * A

        Actor:
            softmax policy-gradient update using the advantage.
        """
        G = 0.0

        for step in reversed(trajectory):
            G = step["reward"] + self.gamma * G
            G = float(np.clip(G, -1000.0, 1000.0))

            s = step["state_key"]
            a = step["action_key"]

            value = self._value(s)
            advantage = G - value

            # Critic update.
            if self.threshold_sharing:
                ss = self._shared_key(s)
                w = 0.5

                self.shared_value_table[ss] = (
                    self.shared_value_table.get(ss, 0.0)
                    + w * self.alpha_critic * advantage
                )

                self.value_table[s] = (
                    self.value_table.get(s, 0.0)
                    + w * self.alpha_critic * advantage
                )
            else:
                self.value_table[s] = (
                    self.value_table.get(s, 0.0)
                    + self.alpha_critic * advantage
                )

            # Correct actor update.
            self._update_actor(s, a, advantage)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def track_action_execution(self, action, layer):
        for node_idx, (_, location) in enumerate(action):
            key = (layer, node_idx)

            if location == 0:
                self.edge_execution_counts[key] = (
                    self.edge_execution_counts.get(key, 0) + 1
                )
            elif location == 1:
                self.optimistic_execution_counts[key] = (
                    self.optimistic_execution_counts.get(key, 0) + 1
                )
            else:
                self.conservative_execution_counts[key] = (
                    self.conservative_execution_counts.get(key, 0) + 1
                )

    def get_execution_stats(self):
        return {
            "edge_counts": self.edge_execution_counts,
            "optimistic_counts": self.optimistic_execution_counts,
            "conservative_counts": self.conservative_execution_counts,
            "total_episodes": self.total_episodes,
        }

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------

    def notify_episode_end(self, episode_reward):
        self.total_episodes += 1

        if episode_reward > self.best_episode_reward + 1e-6:
            self.best_episode_reward = episode_reward
            self.episodes_since_improvement = 0
        else:
            self.episodes_since_improvement += 1

        # Episode-based temperature schedule.
        #
        # This avoids reaching the minimum temperature after only a small
        # fraction of a 1M-episode training run.
        progress = min(1.0, self.total_episodes / 300000.0)

        self.temperature = (
            self.temperature_max * (1.0 - progress)
            + self.temperature_min * progress
        )

        # Optional temporary exploration boost if training is stagnant.
        if self.episodes_since_improvement >= self.stagnant_limit:
            self.temperature = min(
                self.temperature_max,
                self.temperature * self.temperature_boost,
            )
            self.episodes_since_improvement = 0

            print(
                f"  [Boost] Temperature -> {self.temperature:.3f}"
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file="a2c_tables.pkl"):
        with open(file, "wb") as f:
            pickle.dump(
                {
                    "policy": self.policy_table,
                    "value": self.value_table,
                    "shared_policy": self.shared_policy_table,
                    "shared_value": self.shared_value_table,
                },
                f,
            )

    def load(self, file="a2c_tables.pkl"):
        if os.path.exists(file):
            with open(file, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                self.policy_table = data.get("policy", {})
                self.value_table = data.get("value", {})
                self.shared_policy_table = data.get(
                    "shared_policy", {}
                )
                self.shared_value_table = data.get(
                    "shared_value", {}
                )
            else:
                # Legacy (policy, value) tuple.
                self.policy_table, self.value_table = data
