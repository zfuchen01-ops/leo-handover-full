#!/usr/bin/env python3
"""单 CenterAgent 顺序决策 200 用户 —— 消除 multi-agent 羊群效应.

用法:
    LEO_ISL_FAIL_MODE=random python3 -u train_single.py --tag v1 --slots 200
"""

import argparse, os, sys, collections, copy
from math import pi as _pi
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

import numpy as np; np.random.seed(0)
import random as _random; _random.seed(42)
import torch
import torch.nn as nn
import torch.optim as optim

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from Handover import Handover, RATE_UPPER


# ──── 环境构建 (复用 train_drqn.py) ────
def reset_ids():
    Topology.index_con = 0
    User.uid = 0

def make_constellation(topo, constellation="C"):
    if constellation == "C":
        orbit_num, sat_per_orbit = 16, 16
    else:
        orbit_num, sat_per_orbit = 8, 9 if constellation == "A" else 12
    phase = 1
    first_phi = 2.0 * phase * _pi / (orbit_num * sat_per_orbit)
    lean = 54.0 / 180.0 * _pi
    theta = 2.0 * _pi / orbit_num
    topo.Add_Constellation(orbit_num, sat_per_orbit, SAT_HEIGHT, first_phi, lean, theta, TYPE_2PI)
    topo.Each_Satellite()

def make_user_locations(count):
    users = []
    for _ in range(count):
        lat = _random.uniform(-60, 60) * _pi / 180.0
        lon = _random.uniform(-120, 120) * _pi / 180.0
        users.append((lon, lat))
    return users

def build_env(user_count=200, constellation="C"):
    reset_ids()
    topo = Topology()
    make_constellation(topo, constellation)
    gw_coords = [(0, 0, "GW1"), (60, 0, "GW2"), (120, 0, "GW3"),
                 (180, 0, "GW4"), (-120, 0, "GW5"), (-60, 0, "GW6")]
    for lon_deg, lat_deg, name in gw_coords:
        topo.Add_Gateway_Loc(lon_deg / 180.0 * _pi, lat_deg / 180.0 * _pi,
                             antenna_Num=5, name_str=name)
    user_locs = make_user_locations(user_count)
    topo.Add_User_From_Input(user_locs)
    for user in topo.user:
        user.assigned_gateway = topo.gateway[_random.randrange(len(topo.gateway))]
    env = PaperRebuildHandover(net=Network(topo))
    return env


# ──── Q 网络 (无 LSTM, 无 multi-agent, 纯 per-sat scoring) ────
class QNet(nn.Module):
    """Per-sat scoring: 5维特征 → 16维嵌入, +全局上下文(含位置) → Q标量"""
    def __init__(self, n_sats, hidden=128):
        super().__init__()
        self.N = n_sats
        # Per-sat MLP: 5 → 32 → 16
        self.sat_net = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        # 全局上下文: 5N + 1(位置) → 128 → 16
        self.ctx_fc = nn.Sequential(
            nn.Linear(5 * n_sats + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 16), nn.ReLU(),
        )
        # 合并 → Q
        self.q_out = nn.Linear(32, 1)

    def forward(self, state):
        # state: [B, 5N+1] (5N卫星特征 + 1位置)
        B, D = state.shape
        N = self.N
        # Per-sat features: 仅前5N → [B, N, 5] → [B*N, 5] → [B*N, 16] → [B, N, 16]
        per_sat = state[:, :5*N].reshape(B, N, 5)
        sat_feat = self.sat_net(per_sat.reshape(B * N, 5)).reshape(B, N, 16)
        # Global context: [B, 16] → [B, N, 16]
        ctx = self.ctx_fc(state).unsqueeze(1).expand(-1, N, -1)
        # Q per sat: [B, N, 32] → [B, N]
        combined = torch.cat([sat_feat, ctx], dim=-1)
        q = self.q_out(combined).squeeze(-1)
        return q


# ──── Replay Buffer ────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def put(self, state, action, reward, next_state):
        self.buffer.append([state, action, reward, next_state])

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = [], [], [], []
        for i in indices:
            s, a, r, ns = self.buffer[i]
            states.append(s); actions.append(a); rewards.append(r); next_states.append(ns)
        return (torch.tensor(states, dtype=torch.float),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float),
                torch.tensor(next_states, dtype=torch.float))

    def __len__(self):
        return len(self.buffer)


# ──── 训练 ────
def train(env, q_net, target_net, buffer, args):
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    device = next(q_net.parameters()).device
    N = env.topo.total_sat
    users = env.topo.user
    n_users = len(users)
    ISL_FAIL_MODE = os.environ.get('LEO_ISL_FAIL_MODE', '0')

    csv = open(f'./log/RL/DRQN_{args.tag}_per_ep.csv', 'w')
    csv.write('episode,reward,rate_avg,hops_avg,ho_avg,q_spread,uniq_sat,beam_avg,rew_rate,rew_ho,loss\n')

    epsilon_start, epsilon_end = 1.0, 0.05
    total_eps = args.slots

    reward_history = []
    env.reset(0, 'NETWORK_LOAD')

    for ep in range(total_eps):
        # ISL failures
        if ISL_FAIL_MODE == 'random':
            if not hasattr(env.net, '_active_lsa_cache'):
                env.net._active_lsa_cache = []
                for ci in range(len(env.net.LSDB)):
                    for si in range(len(env.net.LSDB[ci])):
                        for lsa in env.net.LSDB[ci][si]:
                            if lsa.isEstablished and lsa.total_band > 0:
                                env.net._active_lsa_cache.append((ci, si, lsa))
                env.net._failed_lsa = []
            active = env.net._active_lsa_cache
            failed = env.net._failed_lsa
            _random.shuffle(active)
            n_total = len(active) + len(failed)
            n_change = max(1, int(n_total * 0.05))
            _random.shuffle(failed)
            n_rec = min(n_change, len(failed))
            for ci, si, lsa in failed[:n_rec]:
                lsa.total_band = lsa._orig; lsa.isEstablished = True
                active.append((ci, si, lsa))
            env.net._failed_lsa = failed[n_rec:]
            n_fail = min(n_change, len(active))
            for ci, si, lsa in active[:n_fail]:
                if lsa.total_band > 0:
                    if not hasattr(lsa, '_orig'): lsa._orig = lsa.total_band
                    lsa.total_band = 0; lsa.isEstablished = False
                    env.net._failed_lsa.append((ci, si, lsa))
            env.net._active_lsa_cache = active[n_fail:]
            env.net.Update_N2N_Load_By_LSDB_All()

        # ε 衰减
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * ep / max(1, total_eps - 1))

        ep_reward = 0.0
        ep_rate, ep_hops, ep_ho, n_conn = 0.0, 0.0, 0.0, 0
        q_spreads = []

        # ── 核心: 顺序决策 200 用户 ──
        for user_idx, user in enumerate(users):
            state = env.Observe(user, 'NETWORK_LOAD')
            # 加位置信息: Q网络需要知道这是第几个用户 (1st vs 200th策略完全不同)
            position = user_idx / max(1, n_users - 1)
            state = np.append(state, position)

            # ε-greedy
            if _random.random() < epsilon:
                visible = [s.ID - 1 for s in user.sat_covered]
                action = _random.choice(visible) if visible else 0
            else:
                state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals = q_net(state_t).squeeze(0)
                # Mask invisible
                for i in range(N):
                    if state[N + i] <= 0.0:  # Cavai=0 → invisible
                        q_vals[i] = -float('inf')
                # Q spread for diagnostics
                visible_q = q_vals[q_vals > -float('inf')]
                q_spreads.append((visible_q.max() - visible_q.min()).item() if len(visible_q) > 1 else 0.0)
                action = q_vals.argmax().item()

            # Apply action
            env.step({user: action + 1}, 'NETWORK')

            # Reward
            reward = env.Get_Reward(user)

            # Observe next (for the NEXT user, not the current one)
            # We'll store the transition when processing the NEXT user
            # For the first user, we had no previous state
            # We buffer: (prev_state, prev_action, reward, current_state)
            if hasattr(env, '_last_state'):
                buffer.put(env._last_state, env._last_action, reward, state)
            env._last_state = state
            env._last_action = action

            ep_reward += reward

            # Stats
            u = user
            if u.sat_connected is not None and u.sat_connected in env.ho[u]:
                access = min(env.ho[u][u.sat_connected].c_quality, RATE_UPPER)
                fd = env._get_feeder_sat(u, u.sat_connected)
                isl_b = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected != fd else RATE_UPPER
                r = min(access, isl_b, RATE_UPPER)
                h = 1.0
                if fd and u.sat_connected != fd:
                    h = max(1.0, env.Calc_Path_Hops(u.sat_connected, fd))
                ep_rate += r; ep_hops += h
                ep_ho += 1.0 if u.sat_connected != u.last_connected else 0.0
                n_conn += 1

        # Clear the last state (it was for the last user; we don't need it for next ep)
        del env._last_state, env._last_action

        # Learn
        loss_val = 0.0
        if len(buffer) >= args.batch:
            batch = buffer.sample(args.batch)
            if batch is not None:
                states_b, actions_b, rewards_b, next_states_b = [t.to(device) for t in batch]

                with torch.no_grad():
                    next_q = target_net(next_states_b)
                    # Mask invisible in next_state
                    for b in range(next_q.shape[0]):
                        for i in range(N):
                            if next_states_b[b, N + i] <= 0.0:
                                next_q[b, i] = -float('inf')
                    next_max, _ = next_q.max(dim=-1)
                    target = rewards_b + args.gamma * next_max

                pred_q = q_net(states_b)
                pred = pred_q.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                loss = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                loss_val = loss.item()

        # Sync target net
        if ep % 20 == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Log
        if n_conn > 0:
            ep_rate /= n_conn; ep_hops /= n_conn; ep_ho /= n_conn
        ep_q_spread = sum(q_spreads) / max(1, len(q_spreads))
        uniq_sats = len(set(u.sat_connected.ID for u in users if u.sat_connected is not None))
        beam_avg = n_conn / max(1, uniq_sats * 64.0)
        rew_rate = ep_rate / RATE_UPPER
        rew_ho = -1.5 * (1.0 - ep_ho) if n_conn > 0 else 0.0

        csv.write(f'{ep},{ep_reward:.4f},{ep_rate:.0f},{ep_hops:.1f},{ep_ho:.3f},'
                  f'{ep_q_spread:.4f},{uniq_sats},{beam_avg:.3f},'
                  f'{rew_rate:.2f},{rew_ho:.2f},{loss_val:.4f}\n')
        csv.flush()

        if ep < 20 or ep % 50 == 0:
            print(f'[ep {ep:4d}] rate={ep_rate:5.0f}  HO={ep_ho:.3f}  eps={epsilon:.2f}  Q={ep_q_spread:.4f}  loss={loss_val:.4f}',
                  flush=True)

        reward_history.append(ep_reward)
        env.Update_Env(ep * 50, 'NETWORK_LOAD')

    csv.close()
    return reward_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constellation", default="C")
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--slots", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--buffer", type=int, default=100000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--tag", default="v1_single")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    print(f"=== 单 Agent 顺序决策训练 ===")
    print(f"  用户: {args.users}  时隙: {args.slots}  lr={args.lr}  gamma={args.gamma}  batch={args.batch}")
    print(f"  设备: {device_str}")

    env = build_env(args.users, args.constellation)
    N = env.topo.total_sat
    print(f"  卫星: {N}  用户: {len(env.topo.user)}")

    q_net = QNet(N, args.hidden).to(device)
    target_net = QNet(N, args.hidden).to(device)
    target_net.load_state_dict(q_net.state_dict())
    buffer = ReplayBuffer(args.buffer)

    print(f"  Q_net参数: {sum(p.numel() for p in q_net.parameters()):,}")
    print(f"  架构: 5N→per-sat(5→32→16) + ctx(5N→128→16) → q_per_sat")
    print("-" * 60)

    train(env, q_net, target_net, buffer, args)

    print("-" * 60)
    print("训练完成!")


if __name__ == "__main__":
    main()
