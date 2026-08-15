#!/usr/bin/env python3
"""L0-b 噪声切换冻结 replay: 加载训练好的 DRQN, 对比正常 vs 冻结噪声切换的 rate/ho。

核心假设: 很多切换是 Q 值噪声驱动 (_last_q_gap 很小), 提 ho 不提 rate。
冻结这些切换 (q_gap < threshold 时 hold 当前星), 应能降 ho 而 rate 基本不掉
—— 等价于把 rate↔ho trade-off 曲线整体上抬。

用法:
  python3 l0_replay.py --model log/model/drqn_C_u200_master2000.pkl \
      --const C --users 200 --eps 100 --freeze-gaps 0.01,0.02,0.05,0.10
"""
import os, sys, numpy as np, random, torch
os.environ['LEO_CAHS_VARIANT'] = 'paper'; os.environ['LEO_QUIET_LOGS'] = '1'
os.environ['LEO_C_BAND'] = '100'
sys.path.insert(0, '.')
np.random.seed(0); random.seed(42)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='log/model/drqn_C_u200_master2000.pkl')
ap.add_argument('--const', default='C')
ap.add_argument('--users', type=int, default=200)
ap.add_argument('--eps', type=int, default=100)
ap.add_argument('--freeze-gaps', default='0.01,0.02,0.05,0.10')
args = ap.parse_args()

from train_drqn import build_drqn_env
from DRQNAgent import UserAgent, CenterAgent
from Handover import RATE_UPPER

env = build_drqn_env(args.users, args.const)
c_agent = CenterAgent(env, gamma=0.9, epsilon=0.01, batch=256, buffer=20000,
                      hidden_size=128, seq=6, device='cpu')
u_agents = []
for i, user in enumerate(env.topo.user):
    u_agents.append(UserAgent(user, env, c_agent, gamma=0.9, epsilon=0.01,
                              batch=256, buffer=2000, hidden_size=128, seq=6,
                              device='cpu', head_idx=i))

net = torch.load(args.model, map_location='cpu', weights_only=False)
for a in u_agents:
    a.evaluate_net = net  # 共享权重, head_idx 区分, 顺序 forward 无竞态

def run_eval(freeze_gap, eps):
    env.reset(0, 'NETWORK_LOAD')
    for a in u_agents:
        a.reset('eval')  # mode='eval': argmax, 不走 epsilon/replayer
    # warmup: 让 env + LSTM 进入稳态
    for ep in range(10):
        for a in u_agents:
            obs = a.observe('NETWORK_LOAD')
            act = a.step(obs, 0.0, None, None)
            sids = a.user._topK_sat_ids
            env.step({a.user: sids[act] if act < len(sids) else sids[0]},
                     'INITIAL' if ep == 0 else 'NETWORK')
        env.Update_Env((ep + 1) * 50, 'NETWORK_LOAD')
    sum_rate = 0.0; n = 0; ho = 0; ho_den = 0
    for ep in range(eps):
        for a in u_agents:
            obs = a.observe('NETWORK_LOAD')
            act = a.step(obs, 0.0, None, None)
            sids = a.user._topK_sat_ids
            # 冻结噪声切换: 这次会切换且 q_gap 极小 → hold 当前星
            if freeze_gap >= 0 and a.user.sat_connected is not None:
                cur_id = a.user.sat_connected.ID
                if cur_id in sids:
                    old_idx = sids.index(cur_id)
                    if old_idx != act:
                        g = a._last_q_gap
                        if 0 < g < freeze_gap:
                            act = old_idx
            sid = sids[act] if act < len(sids) else sids[0]
            env.step({a.user: sid}, 'NETWORK')
            # 立即统计 (同 train 循环口径)
            u = a.user
            if u.sat_connected is not None and u.sat_connected in env.ho[u]:
                access = min(env.ho[u][u.sat_connected].c_quality, RATE_UPPER)
                fd = env._get_feeder_sat(u, u.sat_connected)
                isl_b = (env.net.N2N_status[u.sat_connected.con_id - 1][u.sat_connected.ID - 1][fd.ID - 1].free_band
                         if fd and u.sat_connected != fd else RATE_UPPER)
                sum_rate += min(access, isl_b, RATE_UPPER); n += 1
                ho += 1.0 if u.sat_connected != u.last_connected else 0.0
                ho_den += 1
        env.Update_Env((ep + 1) * 50, 'NETWORK_LOAD')
    return sum_rate / n, ho / ho_den

print(f"=== L0-b replay: {args.const}x{args.users} eps={args.eps} ===", flush=True)
print(f"model = {args.model}", flush=True)
r, h = run_eval(-1.0, args.eps)
print(f"[freeze=-1    baseline] rate={r:.0f}  ho={h:.3f}", flush=True)
for g in [float(x) for x in args.freeze_gaps.split(',') if x.strip()]:
    r, h = run_eval(g, args.eps)
    print(f"[freeze={g:<5}         ] rate={r:.0f}  ho={h:.3f}", flush=True)
print("=== L0-b done ===", flush=True)
