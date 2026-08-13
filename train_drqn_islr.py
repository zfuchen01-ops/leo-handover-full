#!/usr/bin/env python3
"""DRQN 杨论文原版特征 + 显式ISL reward"""
import os, sys, copy, collections
import numpy as np; np.random.seed(0)
import random; random.seed(42)
import torch; torch.manual_seed(0)

os.environ['LEO_CAHS_VARIANT'] = 'paper'
os.environ['LEO_QUIET_LOGS'] = '1'
os.environ['LEO_C_BAND'] = '100'
sys.path.insert(0, '.')

from train_drqn import build_drqn_env
from DRQNAgent import Q_net, UserReplayer, CenterReplayer, resolve_device

device = resolve_device('cpu')
H, SEQ, BATCH, TU = 128, 6, 256, 10

def test_20eps(qnet, label):
    env_t = build_drqn_env(200, 'C'); env_t.reset(0, 'NETWORK_LOAD')
    users_t = list(env_t.topo.user); N = env_t.topo.total_sat
    class TA:
        def __init__(self, u, q):
            self.user = u; self.qnet = q; self.sequence = SEQ; self.device = device
            self.state_fifo = collections.deque(maxlen=SEQ)
            self.lstm_h = None; self.lstm_c = None
        def decide(self, obs):
            if self.lstm_h is None:
                self.lstm_h = torch.zeros(1, 1, H).to(device)
                self.lstm_c = torch.zeros(1, 1, H).to(device)
            self.state_fifo.append(obs)
            sl = list(self.state_fifo) if len(self.state_fifo) >= SEQ else [obs]*(SEQ-len(self.state_fifo)) + list(self.state_fifo)
            qv, self.lstm_h, self.lstm_c = self.qnet(torch.as_tensor(sl, dtype=torch.float).to(device).unsqueeze(0), self.lstm_h, self.lstm_c)
            qv = qv[:, -1, :].squeeze(0)
            for i in range(N):
                if obs[N+i] <= 0.0: qv[i] = -float('inf')
            return qv.argmax().item()
    agents = [TA(u, qnet) for u in users_t]
    for ep in range(5):
        for a in agents:
            obs = env_t.Observe(a.user, 'NETWORK_LOAD'); act = a.decide(obs)
            env_t.step({a.user: act+1}, 'NETWORK')
        env_t.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    rates = []
    for ep in range(20):
        ep_rate, n = 0.0, 0
        for a in agents:
            obs = env_t.Observe(a.user, 'NETWORK_LOAD'); act = a.decide(obs)
            env_t.step({a.user: act+1}, 'NETWORK')
            if a.user.sat_connected and a.user.sat_connected in env_t.ho[a.user]:
                fd = env_t._get_feeder_sat(a.user, a.user.sat_connected)
                b = (env_t.net.N2N_status[a.user.sat_connected.con_id-1][a.user.sat_connected.ID-1][fd.ID-1].free_band if fd and a.user.sat_connected!=fd else 9999)
                ep_rate += min(env_t.ho[a.user][a.user.sat_connected].c_quality, b); n += 1
        rates.append(ep_rate/n if n else 0)
        env_t.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    avg = sum(rates)/20; l50 = sum(rates[-10:])/10
    print(f'  [TEST {label}] avg={avg:.0f} last50={l50:.0f}', flush=True)
    return avg, l50

print('DRQN Yang+Cavai+ISLreward ISL=2000 200u 200eps', flush=True)
env = build_drqn_env(200, 'C'); env.reset(0, 'NETWORK_LOAD')
N = env.topo.total_sat; users = list(env.topo.user)

qnet = Q_net(4*N+1, N, H).to(device)
target_net = copy.deepcopy(qnet).to(device)
opt = torch.optim.Adam(qnet.parameters(), lr=0.001)
loss_fn = torch.nn.SmoothL1Loss()
center_buf = CenterReplayer(5000, SEQ)

class A:
    def __init__(self, u, q):
        self.user = u; self.qnet = q; self.epsilon = 0.01; self.sequence = SEQ
        self.state_fifo = collections.deque(maxlen=SEQ)
        self.trajectory = collections.deque(maxlen=6)
        self.replayer = UserReplayer(2000, SEQ)
        self.lstm_h = None; self.lstm_c = None; self.device = device
    def decide(self, obs):
        if self.lstm_h is None:
            self.lstm_h = torch.zeros(1, 1, H).to(device)
            self.lstm_c = torch.zeros(1, 1, H).to(device)
        self.state_fifo.append(obs)
        if np.random.rand() < self.epsilon:
            return np.random.choice([s.ID-1 for s in self.user.sat_covered]) if self.user.sat_covered else 0
        sl = list(self.state_fifo) if len(self.state_fifo) >= SEQ else [obs]*(SEQ-len(self.state_fifo)) + list(self.state_fifo)
        qv, self.lstm_h, self.lstm_c = self.qnet(torch.as_tensor(sl, dtype=torch.float).to(device).unsqueeze(0), self.lstm_h, self.lstm_c)
        qv = qv[:, -1, :].squeeze(0)
        for i in range(N):
            if obs[N+i] <= 0.0: qv[i] = -float('inf')
        return qv.argmax().item()
    def add(self, obs, r, act):
        self.trajectory += [obs, r, act]
        if len(self.trajectory) == 6:
            self.replayer.put([self.trajectory[0], self.trajectory[2], self.trajectory[4], self.trajectory[3]])
            return 1
        return 0

agents = [A(u, qnet) for u in users]
center_buf.reset(agents)
rates, test_results, last_loss = [], [], 0.0

for ep in range(200):
    ep_r, ep_rate, n = 0.0, 0.0, 0
    for a in agents:
        obs = env.Observe(a.user, 'NETWORK_LOAD'); act = a.decide(obs)
        env.step({a.user: act+1}, 'INITIAL' if ep==0 else 'NETWORK')
        r = env.Get_Reward(a.user); ep_r += r
        if a.user.sat_connected and a.user.sat_connected in env.ho[a.user]:
            fd = env._get_feeder_sat(a.user, a.user.sat_connected)
            b = (env.net.N2N_status[a.user.sat_connected.con_id-1][a.user.sat_connected.ID-1][fd.ID-1].free_band if fd and a.user.sat_connected!=fd else 9999)
            ep_rate += min(env.ho[a.user][a.user.sat_connected].c_quality, b); n += 1
        if a.add(obs, r, act) and ep >= SEQ: center_buf.put(a, [obs, act, r, obs])
    rates.append(ep_rate/n if n else 0)

    if ep >= SEQ+1:
        s_l, a_l, r_l, ns_l = center_buf.sample(BATCH)
        if len(s_l) > 0:
            s_t = torch.FloatTensor(s_l).to(device); a_t = torch.LongTensor(a_l).to(device)
            r_t = torch.FloatTensor(r_l).to(device); ns_t = torch.FloatTensor(ns_l).to(device)
            a_last = a_t[:, -1]; r_last = r_t[:, -1]
            with torch.no_grad():
                h_t = torch.zeros(1, BATCH, H).to(device); c_t = torch.zeros(1, BATCH, H).to(device)
                nq, _, _ = target_net(ns_t, h_t, c_t); nq = nq[:, -1, :]
                nq[~(ns_t[:, -1, N:2*N] > 0.0)] = -float('inf')
                target = r_last + 0.9 * nq.max(dim=1).values
            h_q = torch.zeros(1, BATCH, H).to(device); c_q = torch.zeros(1, BATCH, H).to(device)
            q, _, _ = qnet(s_t, h_q, c_q); q = q[:, -1, :].gather(1, a_last.unsqueeze(1)).squeeze(1)
            loss = loss_fn(q, target); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), 5.0); opt.step()
            last_loss = loss.item()
            if ep % TU == 0: target_net.load_state_dict(qnet.state_dict())
    env.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    a_r = sum(rates[-100:])/min(100,len(rates))
    print(f'[ep{ep:3d}] rate={ep_rate/n:3.0f} avgR={a_r:3.0f} loss={last_loss:.4f}', flush=True)

    if ep > 0 and ep % 50 == 0:
        torch.save(qnet.state_dict(), f'./log/model/drqn_yang_ep{ep}.pkl')
        avg, l50 = test_20eps(qnet, f'ep{ep}')
        test_results.append((ep, avg, l50))

avg, l50 = test_20eps(qnet, 'FINAL')
test_results.append((200, avg, l50))
print(f'\nMGCS: 254  MaxISL: 362', flush=True)
for ep, avg, l50 in test_results:
    print(f'DRQN ep{ep}: test={avg:.0f} last50={l50:.0f}', flush=True)
