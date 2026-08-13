#!/usr/bin/env python3
"""最小化梯度冲突诊断测试 — 只跑10ep填buffer, 然后直接调诊断."""
import os, sys
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

from math import pi as _pi
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

import random as _random
import numpy as np; np.random.seed(0)
_random.seed(42)
import torch; torch.manual_seed(0)

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from Handover import Handover, RATE_UPPER
from DRQNAgent import UserAgent, CenterAgent

# ── 快速构建环境 ──
def reset_ids():
    Topology.index_con = 0; User.uid = 0

def build_env(n=200):
    reset_ids()
    t = Topology()
    o, s = 16, 16  # C星座 256星
    p = 1; fp = 2.0 * p * _pi / (o * s)
    ln = 54.0 / 180.0 * _pi; th = 2.0 * _pi / o
    t.Add_Constellation(o, s, SAT_HEIGHT, fp, ln, th, TYPE_2PI)
    t.Each_Satellite()
    for lon, lat, nm in [(0, 0, "G1"), (60, 0, "G2"), (120, 0, "G3"),
                          (180, 0, "G4"), (-120, 0, "G5"), (-60, 0, "G6")]:
        t.Add_Gateway_Loc(lon / 180.0 * _pi, lat / 180.0 * _pi, 5, nm)
    users = [(_random.uniform(-120, 120) * _pi / 180.0,
              _random.uniform(-60, 60) * _pi / 180.0) for _ in range(n)]
    t.Add_User_From_Input(users)
    for u in t.user:
        u.assigned_gateway = t.gateway[_random.randrange(len(t.gateway))]
    return PaperRebuildHandover(net=Network(t))

print("构建环境...", flush=True)
env = build_env(200)
total_sats = env.topo.total_sat
print(f"  卫星={total_sats} 用户={len(env.topo.user)}", flush=True)

print("创建 CenterAgent...", flush=True)
c_agent = CenterAgent(env, gamma=0.9, epsilon=0.01, batch=256, buffer=20000,
                      hidden_size=128, lr=0.001, seq=6, device='cpu')

print("创建 UserAgents (200个)...", flush=True)
u_agents = []
for i, user in enumerate(env.topo.user):
    agent = UserAgent(user, env, c_agent, gamma=0.9, epsilon=0.01,
                      batch=256, buffer=2000, hidden_size=128, seq=6,
                      device='cpu', head_idx=i)
    u_agents.append(agent)

# ── 跑10ep填buffer ──
print("\n填buffer (10ep)...", flush=True)
env.reset(0, 'NETWORK_LOAD')
c_agent.reset(u_agents)
ob_re = {}
for agent in u_agents:
    agent.reset(mode='train')
    ob_re[agent] = [agent.observe('NETWORK_LOAD'), 0.0]

for ep in range(10):
    for agent in u_agents:
        ob_re[agent][0] = agent.observe('NETWORK_LOAD')
        action = agent.step(ob_re[agent][0], ob_re[agent][1], None, None)
        single_action = {agent.user: action + 1}
        if ep == 0:
            env.step(single_action, 'INITIAL')
        else:
            env.step(single_action, 'NETWORK')

    # 算reward
    for agent in u_agents:
        local_r = agent.get_reward()
        ob_re[agent][1] = local_r

    if ep % 3 == 0:
        total_in_buffer = sum(len(a.replayer.memory) for a in u_agents)
        print(f"  ep{ep}: buffer总量={total_in_buffer}", flush=True)

# ── 检查buffer然后手动触发几次learn ──
print(f"\nbuffer就绪, 开始learn+诊断...", flush=True)
for i in range(12):
    if all(len(a.replayer.memory) >= c_agent.sequence + 1 for a in u_agents):
        c_agent.learn()
    else:
        print(f"  learn #{i}: buffer不足,跳过", flush=True)

print("\n=== 完成 ===", flush=True)
print(f"learn_cnt={c_agent.learn_cnt}", flush=True)
