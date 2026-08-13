#!/usr/bin/env python3
"""
DRQN Yang对齐: 无preload + 预分配reward(Eq.50) + MLP + γ=0.99
每50ep测试输出: rate, last50, ISLrank, HO/ep
"""
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
GAMMA = 0.99
Q_CLAMP = 10.0
HO_PENALTY = 1.5  # Yang Eq.50: ω_ue_1
EPSILON = 0.05
EPISODES = 500
FEAT_DIM = 2  # cq + fb per satellite
N_MGCS_BG = 100   # 一半用户用MGCS当背景流量, 对齐杨"background traffic"

class MLP_QNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden=H):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x[:, -1, :])

def test_20eps(qnet, label):
    env_t = build_drqn_env(200, 'C'); env_t.reset(0, 'NETWORK_LOAD')
    all_users = list(env_t.topo.user); N = env_t.topo.total_sat
    mgcs_t = all_users[:N_MGCS_BG]
    drqn_t = all_users[N_MGCS_BG:]

    class TestAgent:
        def __init__(self, u, q):
            self.user = u; self.qnet = q
        def decide(self, obs):
            qv = self.qnet(torch.as_tensor(obs, dtype=torch.float).unsqueeze(0).unsqueeze(0)).squeeze()
            for i in range(N):
                if obs[i] <= 0.0: qv[i] = -float('inf')
            return qv.argmax().item()

    drqn_test = [TestAgent(u, qnet) for u in drqn_t]
    for ep in range(5):
        random.shuffle(mgcs_t)
        for u in mgcs_t:
            if u.sat_covered:
                env_t.step({u: max(u.sat_covered, key=lambda s: env_t.ho[u][s].c_quality).ID}, 'NETWORK')
        random.shuffle(drqn_test)
        for a in drqn_test:
            env_t.step({a.user: a.decide(env_t.Observe(a.user, 'NETWORK_LOAD'))+1}, 'NETWORK')
        env_t.Update_Env((ep+1)*30, 'NETWORK_LOAD')

    rates, isl_ranks, ho_counts = [], [], []
    for ep in range(20):
        ep_rate, n_users, ep_ho = 0.0, 0, 0
        random.shuffle(mgcs_t)
        for u in mgcs_t:
            if u.sat_covered:
                env_t.step({u: max(u.sat_covered, key=lambda s: env_t.ho[u][s].c_quality).ID}, 'NETWORK')
        random.shuffle(drqn_test)
        for a in drqn_test:
            obs = env_t.Observe(a.user, 'NETWORK_LOAD')
            act = a.decide(obs)
            last_sat = a.user.sat_connected
            env_t.step({a.user: act+1}, 'NETWORK')
            if a.user.sat_connected != last_sat:
                ep_ho += 1
            if a.user.sat_connected and a.user.sat_connected in env_t.ho[a.user]:
                fd = env_t._get_feeder_sat(a.user, a.user.sat_connected)
                fb = (env_t.net.N2N_status[a.user.sat_connected.con_id-1][a.user.sat_connected.ID-1][fd.ID-1].free_band
                      if fd and a.user.sat_connected != fd else 9999)
                ep_rate += min(env_t.ho[a.user][a.user.sat_connected].c_quality, fb)
                n_users += 1
            feeder = env_t._get_feeder_sat(a.user)
            if feeder and a.user.sat_connected:
                vis = [(s.ID-1, env_t.net.N2N_status[s.con_id-1][s.ID-1][feeder.ID].free_band)
                       for s in a.user.sat_covered]
                vis.sort(key=lambda x: -x[1])
                chose_id = a.user.sat_connected.ID - 1
                rank = next((r for r, (sid, _) in enumerate(vis) if sid == chose_id), len(vis)-1)
                isl_ranks.append(rank / max(1, len(vis)-1) if len(vis) > 1 else 0.0)
        rates.append(ep_rate / n_users if n_users else 0)
        ho_counts.append(ep_ho)
        env_t.Update_Env((ep+1)*30, 'NETWORK_LOAD')

    avg = sum(rates) / 20; l50 = sum(rates[-10:]) / 10
    isl = sum(isl_ranks) / len(isl_ranks) if isl_ranks else 0.5
    ho = sum(ho_counts) / 20
    print(f'  [TEST {label}] rate={avg:.0f} last50={l50:.0f} ISLrank={isl:.3f} HO={ho:.0f}/ep')
    return avg, l50, isl, ho

print(f'DRQN Yang-align: MGCS-bg(100)+DRQN(100) reward=min(cq,fb_obs)/1000-1.5ho γ=0.99 ep={EPISODES}')
env = build_drqn_env(200, 'C'); env.reset(0, 'NETWORK_LOAD')
N = env.topo.total_sat; users = list(env.topo.user)

# 前N_MGCS_BG个用户固定用MCC做背景流量
mgcs_bg_users = users[:N_MGCS_BG]
drqn_users = users[N_MGCS_BG:]

qnet = MLP_QNet(FEAT_DIM*N+1, N).to(device)
target_net = copy.deepcopy(qnet).to(device)
opt = torch.optim.Adam(qnet.parameters(), lr=0.001)
loss_fn = nn.SmoothL1Loss()
replay_buf = collections.deque(maxlen=50000)

class DRQNAgent:
    def __init__(self, u, q):
        self.user = u; self.qnet = q; self.epsilon = EPSILON
    def decide(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.choice([s.ID-1 for s in self.user.sat_covered]) if self.user.sat_covered else 0
        qv = self.qnet(torch.as_tensor(obs, dtype=torch.float).unsqueeze(0).unsqueeze(0)).squeeze()
        for i in range(N):
            if obs[i] <= 0.0: qv[i] = -float('inf')
        return qv.argmax().item()

def mgcs_decide(user):
    """MGCS策略: 选最高cq的卫星"""
    if user.sat_covered:
        return max(user.sat_covered, key=lambda s: env.ho[user][s].c_quality).ID - 1
    return 0

drqn_agents = [DRQNAgent(u, qnet) for u in drqn_users]
episode_rates = []; test_results = []; last_loss = 0.0

for ep in range(EPISODES):
    ep_reward, ep_rate, n_valid = 0.0, 0.0, 0
    mode = 'INITIAL' if ep == 0 else 'NETWORK'
    random.shuffle(mgcs_bg_users)
    ep_trans = []

    # Phase 1: MGCS用户先行, 制造天然ISL拥堵
    for u in mgcs_bg_users:
        obs = env.Observe(u, 'NETWORK_LOAD')
        act = mgcs_decide(u)
        env.step({u: act+1}, mode)

    # Phase 2: DRQN用户决策, 学习避开拥堵
    random.shuffle(drqn_agents)
    for a in drqn_agents:
        obs = env.Observe(a.user, 'NETWORK_LOAD')
        act = a.decide(obs)
        env.step({a.user: act+1}, mode)

        if a.user.sat_connected and a.user.sat_connected in env.ho[a.user]:
            cq_chosen = obs[act] * 1000.0
            fb_chosen = obs[N + act] * 2000.0
            cavai = min(cq_chosen, fb_chosen) / 1000.0
            ho_pen = HO_PENALTY if a.user.sat_connected != a.user.last_connected else 0.0
            r = cavai - ho_pen
        else:
            r = 0.0
        ep_reward += r

        if a.user.sat_connected and a.user.sat_connected in env.ho[a.user]:
            fd = env._get_feeder_sat(a.user, a.user.sat_connected)
            fb = (env.net.N2N_status[a.user.sat_connected.con_id-1][a.user.sat_connected.ID-1][fd.ID-1].free_band
                  if fd and a.user.sat_connected != fd else 9999)
            ep_rate += min(env.ho[a.user][a.user.sat_connected].c_quality, fb); n_valid += 1

        ep_trans.append((obs, act, r))
    episode_rates.append(ep_rate / n_valid if n_valid else 0)

    for i in range(len(ep_trans)):
        obs, act, r = ep_trans[i]
        next_obs = ep_trans[i+1][0] if i+1 < len(ep_trans) else ep_trans[i][0]
        replay_buf.append((obs, act, r, next_obs))

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
            target = torch.clamp(target, -Q_CLAMP, Q_CLAMP)
        q = qnet(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
        loss = loss_fn(q, target); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(qnet.parameters(), 5.0); opt.step()
        last_loss = loss.item()
        if ep % TU == 0: target_net.load_state_dict(qnet.state_dict())

    env.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    avgR = sum(episode_rates[-100:]) / min(100, len(episode_rates))
    print(f'[ep{ep:4d}] rate={ep_rate/n_valid:3.0f} avgR={avgR:3.0f} loss={last_loss:.4f}')

    if ep > 0 and ep % 50 == 0:
        torch.save(qnet.state_dict(), f'./log/model/mlp_ep{ep}.pkl')
        r, l50, isl, ho = test_20eps(qnet, f'ep{ep}')
        test_results.append((ep, r, l50, isl, ho))

r, l50, isl, ho = test_20eps(qnet, 'FINAL')
test_results.append((EPISODES, r, l50, isl, ho))

print()
print(f'Setup: {N_MGCS_BG} MGCS(bg) + {len(drqn_users)} DRQN | Baselines(clean): MGCS=253 MaxISL=358 Random=173')
print(f'{"Ep":>5s}  {"rate":>5s}  {"last50":>6s}  {"ISLrank":>8s}  {"HO/ep":>5s}')
print(f'{"="*45}')
for ep, avg, l50, isl, ho in test_results:
    print(f'{ep:5d}  {avg:5.0f}  {l50:6.0f}  {isl:8.3f}  {ho:5.0f}')
print(f'\nTarget: rate > MGCS(253) + HO < MGCS(46)')
