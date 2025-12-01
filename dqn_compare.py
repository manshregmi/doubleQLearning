# dqn_vs_doubleq.py
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
from profiling.profile import ProfilingData
from simulator.simulator import CloudEdgeSimulator
from profiling.initialize_profiling import get_profiling_data



# -------------------------
# Q-network
# -------------------------
class QNet(nn.Module):
    def __init__(self, state_dim: int = 2, hidden: int = 64, action_count: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_count)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Replay buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------
# DQN Agent
# -------------------------
class DQNAgent:
    def __init__(
        self,
        state_dim: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 50000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999995,
        target_update_steps: int = 500,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_steps = target_update_steps

        self.q = QNet(state_dim=state_dim, action_count=2).to(self.device)
        self.q_target = QNet(state_dim=state_dim, action_count=2).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.learn_step = 0

    def act(self, state: np.ndarray, action_count: int):
        # state: 1D numpy vector length 2
        if random.random() < self.epsilon:
            return random.randrange(action_count)
        s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(s_t)  # shape (1, action_count)
        return int(torch.argmax(qvals, dim=1).item())

    def push(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        s_b, a_b, r_b, s2_b, d_b = self.buffer.sample(self.batch_size)

        s_b = torch.tensor(s_b, dtype=torch.float32, device=self.device)
        a_b = torch.tensor(a_b, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_b = torch.tensor(r_b, dtype=torch.float32, device=self.device)
        s2_b = torch.tensor(s2_b, dtype=torch.float32, device=self.device)
        d_b = torch.tensor(d_b, dtype=torch.float32, device=self.device)

        q_vals = self.q(s_b).gather(1, a_b).squeeze(1)  # Q(s,a)
        with torch.no_grad():
            next_q = self.q_target(s2_b).max(dim=1)[0]
            target = r_b + self.gamma * next_q * (1.0 - d_b)

        loss = nn.functional.mse_loss(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        # epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -------------------------
# Utility functions to map action index -> action matrix
# -------------------------
def choose_action_matrix_from_index(actions: List[np.ndarray], idx: int) -> np.ndarray:
    return actions[idx].copy()


# -------------------------
# DQN training loop using CloudEdgeSimulator
# -------------------------
def train_dqn_with_simulator(
    sim,                      # instance of CloudEdgeSimulator
    profiling,                # instance of ProfilingData (sim.profiling is same)
    episodes: int = 400,
    replay_warmup: int = 1000,
    agent_kwargs: dict = None,
    seed: int = 0,
):
    """
    Trains DQN with the simulator. Returns (agent, history)
    history => dict with 'episode_reward' list and 'episode_info' list
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if agent_kwargs is None:
        agent_kwargs = {}

    agent = DQNAgent(state_dim=2, **agent_kwargs)
    history = {"episode_reward": [], "episode_info": []}

    num_layers = len(profiling.layers)

    for ep in range(episodes):
        # Episode initialization
        # current_state format required by simulator methods:
        # (bandwidth, cloud_pending, layer, prev_action, surplus, negative_surplus_count)
        initial_bandwidth = profiling.bandwidth if hasattr(profiling, "bandwidth") else 8
        current_state = (initial_bandwidth, 0.0, 0, None, 0.0, 0)
        prev_action = None
        surplus = 0.0
        negative_surplus = 0

        ep_reward = 0.0
        ep_info = {"per_layer": []}

        for layer in range(num_layers):
            # generate possible actions for this layer using your simulator
            possible_actions = sim.get_possible_actions(layer)
            action_count = len(possible_actions)
            if action_count == 0:
                # nothing to do (shouldn't happen)
                continue



            # ---------- construct current observable state (per your Q1) ----------
            # For the initial state of this layer we need a "layer-cost" estimate.
            # We'll set cost_to_use to 0 for the first instant (before action).
            # Many RL setups use the last step's resulting cost as current state's cost.
            # Here we'll provide state = (0.0, available_capacity) before action selection.
            # That is acceptable because action is chosen based on capacity and previous cost is the layer cost.

            available_capacity = np.random.uniform(0.5,1)
            state = np.array([0.0, available_capacity], dtype=np.float32)

            # ---------- choose action index via DQN ----------
            action_idx = agent.act(state, action_count)

            # map action index to actual action matrix
            current_action = choose_action_matrix_from_index(possible_actions, action_idx)

            new_cloud_pending, available_capacity = sim.get_next_state_cloud_waiting_time(next_layer=layer, current_action=current_action, isAllCloud=False)


            # ---------- compute threshold: cost_if_all_edge ----------
            # Create an all-edge action for this layer (all nodes loc=0)
            all_edge_action = current_action.copy()
            all_edge_action[:, 1] = 0
            # Use your compute_energy_and_time() to compute the all-edge cost
            energy_edge, time_edge_s = sim.compute_energy_and_time(
                current_state=current_state,
                current_action=all_edge_action,
                cloud_pending_ms=0.0
            )
            completion_time_edge_ms = time_edge_s * 1000.0
            threshold_cost = 0.5 * completion_time_edge_ms + 0.5 * energy_edge  # per your instruction Q5

            # ---------- Use simulator to compute new cloud pending (we also keep the sampled congestion) ----------
            # NOTE: your simulator's method does an internal random too; we keep `congestion_sample` for available_capacity

            # ---------- compute energy & time for chosen action using simulator -->
            energy, time_s = sim.compute_energy_and_time(
                current_state=current_state,
                current_action=current_action,
                cloud_pending_ms=new_cloud_pending
            )
            completion_time_ms = time_s * 1000.0

            # ---------- compute cost for THIS layer (per your Q1) ----------
            cost = 0.5 * completion_time_ms + 0.5 * energy

            # ---------- compute reward using interpretation 2 (Q4) ----------
            # reward = (threshold - cost) / threshold  (if threshold == 0 avoid divide-by-zero)
            if threshold_cost == 0:
                reward = 0.0
            else:
                reward = (threshold_cost - cost) / threshold_cost

            # ---------- determine next_state using simulator get_next_state ----------
            next_state_tuple, terminal_flag, next_cloud_pending = sim.get_next_state(
                current_state=current_state,
                action=current_action,
                surplus=surplus,
                negative_surplus_count=negative_surplus,
                new_cloud_pending=new_cloud_pending
            )
            # next_state_tuple = (bandwidth, cloud_pending, next_layer, action.copy(), surplus, negative_surplus_count)
            # For our agent state representation we use cost of next layer (we will set it to 0 until agent takes action there)
            # and the available_capacity derived from a fresh sample of congestion for the next decision.
            # But to be consistent with your requirement we use the same sampled congestion_sample for this step's "available_capacity".
            next_bandwidth, _, next_layer, prev_action_for_next, _, negative_surplus = next_state_tuple

            # For the next state's available_capacity we sample a new congestion as the next layer's capacity.
            # (This mirrors how get_next_state_cloud_waiting_time uses randomness)
            next_state_agent = np.array([cost, available_capacity], dtype=np.float32)

            # ---------- push transition and train ----------
            agent.push(state, action_idx, reward, next_state_agent, 1 if terminal_flag else 0)
            agent.train_step()

            # ---------- bookkeeping ----------
            ep_reward += reward
            ep_info["per_layer"].append({
                "layer": layer,
                "action_idx": int(action_idx),
                "cost": float(cost),
                "threshold": float(threshold_cost),
                "reward": float(reward),
                "energy": float(energy),
                "time_ms": float(completion_time_ms),
                "available_capacity": float(available_capacity)
            })

            # update current_state for next layer: use the next_state_tuple returned by simulator
            current_state = (
                next_bandwidth,
                next_cloud_pending,
                next_layer,
                prev_action_for_next,
                surplus,                
                int(negative_surplus),
            )

            # update prev_action for next layer
            prev_action = current_action.copy()

            # if terminal_flag -> episode ends (should only happen after last layer)
            if terminal_flag:
                break

        # end-of-episode bookkeeping
        history["episode_reward"].append(ep_reward)
        history["episode_info"].append(ep_info)

        # print progress
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_last10 = np.mean(history["episode_reward"][-10:]) if len(history["episode_reward"]) >= 1 else ep_reward
            print(f"Episode {ep+1}/{episodes}  reward={ep_reward:.4f}  eps={agent.epsilon:.3f}  avg10={avg_last10:.4f}")

    return agent, history

# ---------------------------------------------------------
# Evaluate one episode with epsilon = 0 (pure exploitation)
# ---------------------------------------------------------
def evaluate_one_episode(sim, profiling, agent):
    """
    Run ONE full episode with greedy actions (epsilon = 0)
    and print final energy + time.
    """
    print("\n=== Running Greedy Evaluation Episode (ε = 0) ===")

    # backup original epsilon
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    num_layers = len(profiling.layers)

    initial_bandwidth = profiling.bandwidth if hasattr(profiling, "bandwidth") else 5.0
    current_state = (initial_bandwidth, 0.0, 0, None, 0.0, 0)

    prev_action = None
    surplus = 0.0
    negative_surplus = 0

    total_energy_all_layers = 0.0
    total_time_all_layers = 0.0  # milliseconds

    for layer in range(num_layers):

        possible_actions = sim.get_possible_actions(layer)
        action_count = len(possible_actions)

        available_capacity = np.random.uniform(0.5, 1.0)
        state = np.array([0.0, available_capacity], dtype=np.float32)

        # always pick greedy action
        action_idx = agent.act(state, action_count)
        current_action = possible_actions[action_idx].copy()

        # get cloud waiting
        new_cloud_pending, available_capacity = sim.get_next_state_cloud_waiting_time(
            next_layer=layer,
            current_action=current_action,
            isAllCloud=False
        )

        # threshold = cost if executed on edge
        all_edge_action = current_action.copy()
        all_edge_action[:, 1] = 0
        e_edge, t_edge_s = sim.compute_energy_and_time(
            current_state=current_state,
            current_action=all_edge_action,
            cloud_pending_ms=0.0
        )

        threshold_cost = 0.5 * (t_edge_s * 1000.0) + 0.5 * e_edge

        # compute chosen action metrics
        energy, time_s = sim.compute_energy_and_time(
            current_state=current_state,
            current_action=current_action,
            cloud_pending_ms=new_cloud_pending
        )
        time_ms = time_s * 1000.0
        cost = 0.5 * time_ms + 0.5 * energy

        # accumulate totals
        total_energy_all_layers += energy
        total_time_all_layers += time_ms

        # move to next layer
        next_state_tuple, terminal_flag, next_cloud_pending = sim.get_next_state(
            current_state=current_state,
            action=current_action,
            surplus=surplus,
            negative_surplus_count=negative_surplus,
            new_cloud_pending=new_cloud_pending
        )
        next_bandwidth, _, next_layer, prev_action_for_next, _, negative_surplus = next_state_tuple

        current_state = (
            next_bandwidth,
            next_cloud_pending,
            next_layer,
            prev_action_for_next,
            surplus,
            int(negative_surplus),
        )
        prev_action = current_action.copy()

        if terminal_flag:
            break

    # restore epsilon
    agent.epsilon = old_epsilon

    # print("=== Evaluation Completed ===")
    # print(f"Total Energy  : {total_energy_all_layers:.4f} J")
    # print(f"Total Time    : {total_time_all_layers:.2f} ms")
    # print("================================\n")

    return total_energy_all_layers, total_time_all_layers


# -------------------------
# Example: how to run (you must adapt imports and objects)
# -------------------------
if __name__ == "__main__":

    p = get_profiling_data(500)
    sim = CloudEdgeSimulator(p)
    agent, hist = train_dqn_with_simulator(sim, p, episodes=80000)
    E, T = [], []
    for _ in range(100):
        e,t = evaluate_one_episode(sim=sim, profiling=p, agent=agent)
        E.append(e)
        T.append(t)
    print(f"Avg Energy over 100 eval episodes: {np.mean(E):.4f} J")
    print(f"Avg Time over 100 eval episodes: {np.mean(T):.2f} ms")
    pass
