#!/usr/bin/env python3
"""DRQN 实验5: cq+free_band独立(2N+1) + MLP + 纯rate reward — 验证特征分离"""
import os, sys, copy, collections
import numpy as np; np.random.seed(0)
import random; random.seed(42)
import torch; torch.manual_seed(0)
import torch.nn as nn

os.environ['LEO_CAHS_VARIANT'] = 'paper'
os.environ['LEO_QUIET_LOGS'] = '1'
os.environ['LEO_C_BAND'] = '100'
sys.path.insert(0, '.')

from train_drqn import build_drqn_env

device = 'cpu'
H, BATCH, TU = 128, 256, 10

class MLP_QNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden=H):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x[:, -1, :])

def test_20eps(qnet, label):
    env_t = build_drqn_env(200, 'C'); env_t.reset(0, 'NETWORK_LOAD')
    users_t = list(env_t.topo.user); N = env_t.topo.total_sat
    class TA:
        def __init__(self, u, q):
            self.user = u; self.qnet = q
        def decide(self, obs):
            qv = self.qnet(torch.as_tensor(obs, dtype=torch.float).unsqueeze(0).unsqueeze(0)).squeeze()
            for i in range(N):
                if obs[i] <= 0.0: qv[i] = -float('inf')
            return qv.argmax().item()
    agents = [TA(u, qnet) for u in users_t]
    for ep in range(5):
        for a in agents:
            env_t.step({a.user: a.decide(env_t.Observe(a.user, 'NETWORK_LOAD'))+1}, 'NETWORK')
        env_t.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    rates, isl_ranks = [], []
    for ep in range(20):
        ep_rate, n = 0.0, 0
        for a in agents:
            obs = env_t.Observe(a.user, 'NETWORK_LOAD'); act = a.decide(obs)
            env_t.step({a.user: act+1}, 'NETWORK')
            if a.user.sat_connected and a.user.sat_connected in env_t.ho[a.user]:
                fd = env_t._get_feeder_sat(a.user, a.user.sat_connected)
                b = (env_t.net.N2N_status[a.user.sat_connected.con_id-1][a.user.sat_connected.ID-1][fd.ID-1].free_band
                     if fd and a.user.sat_connected != fd else 9999)
                ep_rate += min(env_t.ho[a.user][a.user.sat_connected].c_quality, b); n += 1
            feeder = env_t._get_feeder_sat(a.user)
            if feeder and a.user.sat_connected:
                vis = []
                for sat in a.user.sat_covered:
                    fb = env_t.net.N2N_status[sat.con_id-1][sat.ID-1][feeder.ID-1].free_band
                    vis.append((sat.ID-1, fb))
                vis.sort(key=lambda x: -x[1])
                cr = next((r for r,(sid,_) in enumerate(vis) if sid==a.user.sat_connected.ID-1), len(vis)-1)
                isl_ranks.append(cr/max(1,len(vis)-1) if len(vis)>1 else 0.0)
        rates.append(ep_rate/n if n else 0)
        env_t.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    avg = sum(rates)/20; l50 = sum(rates[-10:])/10; isl = sum(isl_ranks)/len(isl_ranks) if isl_ranks else 0.5
    print(f'  [TEST {label}] avg={avg:.0f} last50={l50:.0f} ISLrank={isl:.3f}', flush=True)
    return avg, l50, isl

print('DRQN Exp6: cq+fb独立(2N+1) + MLP + γ=0 + ISL-congestion-aware reward', flush=True)
env = build_drqn_env(200, 'C'); env.reset(0, 'NETWORK_LOAD')
N = env.topo.total_sat; users = list(env.topo.user)
GAMMA, EPSILON = 0.0, 0.01

qnet = MLP_QNet(2*N+1, N).to(device)
target_net = copy.deepcopy(qnet).to(device)
opt = torch.optim.Adam(qnet.parameters(), lr=0.001)
loss_fn = nn.SmoothL1Loss()
replay_buf = collections.deque(maxlen=10000)

class Agent:
    def __init__(self, u, q):
        self.user = u; self.qnet = q; self.epsilon = EPSILON
    def decide(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice([s.ID-1 for s in self.user.sat_covered]) if self.user.sat_covered else 0
        qv = self.qnet(torch.as_tensor(obs, dtype=torch.float).unsqueeze(0).unsqueeze(0)).squeeze()
        for i in range(N):
            if obs[i] <= 0.0: qv[i] = -float('inf')
        return qv.argmax().item()

agents = [Agent(u, qnet) for u in users]
rates, test_results, last_loss = [], [], 0.0

for ep in range(200):
    ep_r, ep_rate, n = 0.0, 0.0, 0
    for a in agents:
        obs = env.Observe(a.user, 'NETWORK_LOAD'); act = a.decide(obs)
        env.step({a.user: act+1}, 'INITIAL' if ep==0 else 'NETWORK')
        r = env.Get_Reward(a.user); ep_r += r
        if a.user.sat_connected and a.user.sat_connected in env.ho[a.user]:
            fd = env._get_feeder_sat(a.user, a.user.sat_connected)
            b = (env.net.N2N_status[a.user.sat_connected.con_id-1][a.user.sat_connected.ID-1][fd.ID-1].free_band
                 if fd and a.user.sat_connected != fd else 9999)
            ep_rate += min(env.ho[a.user][a.user.sat_connected].c_quality, b); n += 1
        replay_buf.append((obs, act, r, obs))
    rates.append(ep_rate/n if n else 0)

    if len(replay_buf) >= BATCH:
        batch = random.sample(replay_buf, BATCH)
        s_b = torch.FloatTensor([b[0] for b in batch]).unsqueeze(1)
        a_b = torch.LongTensor([b[1] for b in batch])
        r_b = torch.FloatTensor([b[2] for b in batch])
        ns_b = torch.FloatTensor([b[3] for b in batch]).unsqueeze(1)
        with torch.no_grad():
            nq = target_net(ns_b)
            nq[~(ns_b[:, -1, :N] > 0.0)] = -float('inf')
            target = r_b + GAMMA * nq.max(dim=1).values
        q = qnet(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
        loss = loss_fn(q, target); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(qnet.parameters(), 5.0); opt.step()
        last_loss = loss.item()
        if ep % TU == 0: target_net.load_state_dict(qnet.state_dict())
    env.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    a_r = sum(rates[-100:])/min(100,len(rates))
    print(f'[ep{ep:3d}] rate={ep_rate/n:3.0f} avgR={a_r:3.0f} loss={last_loss:.4f}', flush=True)
    if ep > 0 and ep % 50 == 0:
        torch.save(qnet.state_dict(), f'./log/model/mlp_cqfb_ep{ep}.pkl')
        avg, l50, isl = test_20eps(qnet, f'ep{ep}')
        test_results.append((ep, avg, l50, isl))

avg, l50, isl = test_20eps(qnet, 'FINAL')
test_results.append((200, avg, l50, isl))
print(f'\nMGCS: 254  MaxISL: 362', flush=True)
for ep, avg, l50, isl in test_results:
    print(f'MLP ep{ep}: test={avg:.0f} last50={l50:.0f} ISLrank={isl:.3f}', flush=True)
