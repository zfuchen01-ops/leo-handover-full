#!/usr/bin/env python3
"""ε=0 测试已训练模型."""
import os, sys, time
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")
os.environ.setdefault("LEO_ORTH_LAMBDA", "0.1")
os.environ.setdefault("LEO_HO_PENALTY", "0")

from math import pi as _pi
from pathlib import Path; BASE_DIR = Path(__file__).resolve().parent
import random as _random; import numpy as np; np.random.seed(0); _random.seed(42)
import torch; torch.manual_seed(0); import copy, collections

# 环境
from Topology import Topology; from User import User
from Network import Network; from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from Handover import Handover, RATE_UPPER
from DRQNAgent import UserAgent, CenterAgent, TOP_K, MAX_SATS, FEAT_PER_SAT, VARLEN

def reset_ids(): Topology.index_con = 0; User.uid = 0

def build_env(n=200, constellation='C'):
    reset_ids(); t = Topology()
    if constellation == 'A': o, s = 8, 9
    elif constellation == 'B': o, s = 12, 12
    elif constellation == 'C': o, s = 16, 16
    elif constellation == 'D': o, s = 20, 20
    else: raise ValueError(f"unknown constellation: {constellation}")
    p = 1
    fp = 2.0 * p * _pi / (o * s); ln = 54.0 / 180.0 * _pi; th = 2.0 * _pi / o
    t.Add_Constellation(o, s, SAT_HEIGHT, fp, ln, th, TYPE_2PI); t.Each_Satellite()
    for lon, lat, nm in [(0,0,"G1"),(60,0,"G2"),(120,0,"G3"),(180,0,"G4"),(-120,0,"G5"),(-60,0,"G6")]:
        t.Add_Gateway_Loc(lon/180.0*_pi, lat/180.0*_pi, 5, nm)
    users = [(_random.uniform(-120,120)*_pi/180.0, _random.uniform(-60,60)*_pi/180.0) for _ in range(n)]
    t.Add_User_From_Input(users)
    for u in t.user: u.assigned_gateway = t.gateway[_random.randrange(len(t.gateway))]
    return PaperRebuildHandover(net=Network(t))

# Load model — 支持LEO_USE_TRANSFORMER切换
model_path = Path(os.environ.get('LEO_MODEL_PATH', str(BASE_DIR / "log/model/drqn_C_u200.pkl")))
feat = int(os.environ.get('LEO_FEAT_PER_SAT', '4'))
use_tf = os.environ.get('LEO_USE_TRANSFORMER', '0') == '1'
device = os.environ.get('LEO_TEST_DEVICE', 'cpu')
n_users = int(os.environ.get('LEO_TEST_USERS', '200'))
constellation = os.environ.get('LEO_TEST_CONSTELLATION', 'C')
print(f"加载模型: {model_path} (FEAT={feat}, TF={use_tf}, dev={device}, n={n_users}, const={constellation})", flush=True)
if not model_path.exists():
    print("模型不存在!"); sys.exit(1)

env = build_env(n_users, constellation)
c_agent = CenterAgent(env, gamma=0.9, epsilon=0.01, batch=256, buffer=20000,
                      hidden_size=128, lr=0.001, seq=6, device=device)
c_agent.evaluate_net = torch.load(str(model_path), map_location='cpu', weights_only=False)
c_agent.target_net = copy.deepcopy(c_agent.evaluate_net)
c_agent.evaluate_net.eval()

u_agents = []
for i in range(n_users):
    agent = UserAgent(env.topo.user[i], env, c_agent, gamma=0.9, epsilon=0.0,
                      batch=256, buffer=2000, hidden_size=128, seq=6,
                      device='cuda', head_idx=i)
    agent.use_transformer = use_tf
    agent.evaluate_net = copy.deepcopy(c_agent.evaluate_net).to(agent.device)
    u_agents.append(agent)

N_TEST = 100
print(f"测试 {N_TEST}ep (ε=0, 贪心)...", flush=True)
env.reset(0, 'NETWORK_LOAD')
c_agent.reset(u_agents)
ob_re = {}
for agent in u_agents:
    agent.reset(mode='test')
    agent.epsilon = 0.0
    ob_re[agent] = [agent.observe('NETWORK_LOAD'), 0.0]

rates, hos, q_spreads = [], [], []
for ep in range(N_TEST):
    ep_rate, ep_ho, n_conn = 0.0, 0.0, 0
    for agent in u_agents:
        ob_re[agent][0] = agent.observe('NETWORK_LOAD')
        action_idx = agent.step(ob_re[agent][0], ob_re[agent][1], None, None)
        K = MAX_SATS if (VARLEN and use_tf) else TOP_K
        sat_ids = getattr(agent.user, '_topK_sat_ids', [0]*K)
        sat_id = sat_ids[action_idx] if action_idx < len(sat_ids) else sat_ids[0]
        env.step({agent.user: sat_id}, 'INITIAL' if ep == 0 else 'NETWORK')

    for agent in u_agents:
        local_r = agent.get_reward()
        ob_re[agent][1] = local_r
        u = agent.user
        if u.sat_connected and u.sat_connected in env.ho[u]:
            fd = env._get_feeder_sat(u, u.sat_connected)
            fb = 9999.0
            if fd and u.sat_connected != fd:
                fb = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band
            ep_rate += min(env.ho[u][u.sat_connected].c_quality, fb)
            n_conn += 1
            if u.last_connected and u.sat_connected != u.last_connected:
                ep_ho += 1

    if n_conn > 0:
        rates.append(ep_rate / n_conn)
        hos.append(ep_ho)
    q_spreads.append(sum(getattr(a, '_last_q_spread', 0) for a in u_agents) / n_users)

    env.Update_Env((ep+1)*50, 'NETWORK_LOAD')
    if ep % 10 == 9:
        print(f"  ep{ep+1}: rate={rates[-1]:.0f} HO={hos[-1]} q={q_spreads[-1]:.4f}", flush=True)

avg_rate = sum(rates) / len(rates)
avg_ho = sum(hos) / len(hos) / n_users
avg_q = sum(q_spreads) / len(q_spreads)
print(f"\n=== 测试结果 ({N_TEST}ep, ε=0) ===")
print(f"  平均 rate:  {avg_rate:.0f} Mbps")
print(f"  平均 HO率:  {avg_ho:.3f}")
print(f"  平均 q_spread: {avg_q:.4f}")
print(f"  rate范围:   {min(rates):.0f} ~ {max(rates):.0f}")
