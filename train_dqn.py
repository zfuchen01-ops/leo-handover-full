#!/usr/bin/env python3
"""DQN 训练脚本 (无LSTM, 纯全连接)"""
import argparse, os, sys, copy, time, collections, csv
from math import pi as _pi, isnan, isinf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
np.random.seed(0)

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI, ISL_CAPACITY

def reset_ids():
    Topology.index_con = 0; User.uid = 0

def make_constellation(topo, c):
    if c == 'A': o, s = 8, 9
    elif c == 'B': o, s = 12, 12
    elif c == 'C': o, s = 16, 16
    else: raise ValueError(c)
    phase = 1
    fp = 2.0*_pi/(o*s); lean = 54.0/180.0*_pi; theta = 2.0*_pi/o
    topo.Add_Constellation(o, s, SAT_HEIGHT, fp, lean, theta, TYPE_2PI)
    topo.Each_Satellite()

def build_env(user_count, constellation):
    reset_ids(); topo = Topology()
    make_constellation(topo, constellation)
    gw_coords = [(0,0,"GW1"),(60,0,"GW2"),(120,0,"GW3"),
                 (180,0,"GW4"),(-120,0,"GW5"),(-60,0,"GW6")]
    for lon, lat, name in gw_coords:
        topo.Add_Gateway_Loc(lon/180*_pi, lat/180*_pi, antenna_Num=5, name_str=name)  # 论文5天线
    import random; random.seed(42)
    random.seed(42)
    users = [(random.uniform(-120,120)*_pi/180, random.uniform(-60,60)*_pi/180) for _ in range(user_count)]
    topo.Add_User_From_Input(users)
    rnd = __import__('random'); rnd.seed(42)
    for u in topo.user:
        u.assigned_gateway = topo.gateway[rnd.randrange(len(topo.gateway))]
    # 论文: 无用户配对, 每用户直接与gateway通信
    return PaperRebuildHandover(net=Network(topo))

class QNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))

    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def put(self, s, a, r, ns):
        self.buf.append((s,a,r,ns))

    def sample(self, n):
        idx = np.random.choice(len(self.buf), n)
        batch = [self.buf[i] for i in idx]
        s = torch.FloatTensor([b[0] for b in batch])
        a = torch.LongTensor([b[1] for b in batch])
        r = torch.FloatTensor([b[2] for b in batch])
        ns = torch.FloatTensor([b[3] for b in batch])
        return s,a,r,ns

    def __len__(self): return len(self.buf)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--constellation", default="C")
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--slots", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.002)  # v1.1值
    p.add_argument("--hidden", type=int, default=128)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    env = build_env(args.users, args.constellation)
    N = env.topo.total_sat
    S = 5*N+1  # prev_conn + cc + hops + fb + rvt + last_rate
    qnet = QNet(S, N, args.hidden).to(device)
    target = QNet(S, N, args.hidden).to(device)
    target.load_state_dict(qnet.state_dict())
    opt = optim.Adam(qnet.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()  # 与v1.1/原始一致
    buf = ReplayBuffer(10000)

    env.reset(0, 'NETWORK_LOAD')
    actions = {}
    all_agents = list(env.topo.user)  # 200 users

    csv_f = open('./log/RL/DQN_per_ep.csv', 'w')
    csv_f.write('episode,reward,rate_avg,hops_avg,ho_avg\n')
    loss_f = open('./log/RL/DQN_loss.csv', 'w', buffering=1)
    loss_f.write('episode,loss\n')
    reward_hist = []

    epsilon_start, epsilon_end = 0.5, 0.02
    epsilon_decay = 500  # 前500ep从0.5衰减到0.02
    GAMMA = 0.9
    BATCH = 128
    NET_STEP = 20
    GLOBAL_BETA = 0.1

    print(f'DQN: {N}sats hidden={args.hidden} input={S} output={N}', flush=True)
    t0 = time.time()

    cur_loss = 0.0
    for ep in range(args.slots):
        # epsilon decay
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * ep / epsilon_decay)
        ep_r = 0.0

        # ── 补全上集pending: 用本集开头状态 = 上集的 s_{t+1} ──
        for user in all_agents:
            pending = getattr(user, '_pending', None)
            if pending is not None:
                s_t, a_t, r_t = pending
                s_next = env.Observe(user, 'NETWORK_LOAD')  # 本集开头状态 = 上集Update_Env后的s_{t+1}
                buf.put(s_t, a_t, r_t, s_next)
                user._pending = None

        # 逐个决策: 观察→决策→立即执行→拿奖励 (与论文Algorithm 3一致)
        ep_rate = 0.0; ep_hops = 0.0; ep_ho = 0.0; n_users = 0
        for user in all_agents:
            obs = env.Observe(user, 'NETWORK_LOAD')
            user._state = obs
            N_dim = env.topo.total_sat

            if False:  # MGCS init removed
                best_sat, best_q = None, -1
                for sat in user.sat_covered:
                    q = env.ho[user][sat].c_quality
                    if q > best_q: best_q = q; best_sat = sat
                action = (best_sat if best_sat else list(user.sat_covered)[0]).ID - 1
            elif np.random.rand() < epsilon:
                action = np.random.choice([s.ID-1 for s in user.sat_covered])
            else:
                with torch.no_grad():
                    qvals = qnet(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0)
                    for i in range(N_dim):
                        if obs[N_dim + i] <= 0.0: qvals[i] = -float("inf")
                    action = qvals.argmax().item()

            # 立即执行 (后续用户能看到前面的影响)
            env.step({user: action+1}, 'INITIAL' if ep==0 else 'NETWORK')
            local_r = env.Get_Reward(user)
            ep_r += local_r
            user._pending = (user._state, action, local_r)

            # 诊断统计
            if user.sat_connected is not None and user.sat_connected in env.ho[user]:
                access = min(env.ho[user][user.sat_connected].c_quality, 500.0)
                fd = env._get_feeder_sat(user, user.sat_connected)
                isl_b = env.net.N2N_status[user.sat_connected.con_id-1][user.sat_connected.ID-1][fd.ID-1].free_band if fd and user.sat_connected!=fd else 500.0
                rate = min(access, isl_b, 500.0)
                hops = 1.0
                if fd is not None and user.sat_connected != fd:
                    hops = max(1.0, env.Calc_Path_Hops(user.sat_connected, fd))
                ep_rate += rate; ep_hops += hops
                ep_ho += 1.0 if user.sat_connected == user.last_connected else 0.0
                n_users += 1

        env.net.Update_N2N_Load_By_LSDB_All()   # 全部决策完成后刷新N2N
        env.Update_Available_Band()

        if n_users > 0:
            ep_rate /= n_users; ep_hops /= n_users; ep_ho /= n_users

        if len(buf) >= BATCH:
            s,a,r,ns = buf.sample(BATCH)
            s, a, r, ns = s.to(device), a.to(device), r.to(device), ns.to(device)
            # mask next state max
            with torch.no_grad():
                nq = target(ns)
                vis = (ns[:, N_dim:2*N_dim] != 0.0)
                nq[~vis] = -float('inf')
                nmax = nq.max(dim=1).values
                target_q = r + GAMMA * nmax
            q = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = loss_fn(q, target_q)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
            opt.step()
            cur_loss = loss.item()
            loss_f.write(f'{ep},{cur_loss:.6f}\n')
            loss_f.flush()

            if ep % NET_STEP == 0:
                target.load_state_dict(qnet.state_dict())

        actions.clear(); env.Update_Env((ep+1)*30, 'NETWORK_LOAD')  # 30s/集

        csv_f.write(f'{ep},{ep_r:.4f},{ep_rate:.1f},{ep_hops:.2f},{ep_ho:.3f}\n')
        if ep % 50 == 0: csv_f.flush()
        reward_hist.append(ep_r)

        if ep < 20 or ep % 20 == 0:
            loss_str = f'{cur_loss:.4f}' if cur_loss > 0 else '-'
            print(f'ep {ep}: r={ep_r:.1f} avg={sum(reward_hist[-100:])/min(100,len(reward_hist)):.1f} loss={loss_str} rate={ep_rate:.0f} hops={ep_hops:.1f} ho={ep_ho:.2f}', flush=True)

    csv_f.close()
    loss_f.close()
    avg_r = sum(reward_hist)/len(reward_hist)
    print(f'Final: avg={avg_r:.1f} max={max(reward_hist):.1f} time={time.time()-t0:.0f}s', flush=True)

if __name__ == "__main__":
    main()
