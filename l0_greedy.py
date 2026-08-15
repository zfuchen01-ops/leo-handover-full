#!/usr/bin/env python3
"""L0-c 贪心联合 baseline: 真实 env 下对比两种策略的 rate + ho。

  - maxisl  : 每用户挑 fb(ISL 带宽)最大的可见星 —— RL 现在学的近似
  - greedy  : 每用户贪心挑 min(cq, fb) 最大的可见星 —— 联合最优的贪心版
              (考虑 feeder 冲突/切换的真实代价, 是"只看 rate 不管 ho"的上界)

输出两行 rate + ho, 直接回答:
  1. 联合优化的肉在真实 env 里还剩下多少 (greedy.rate - maxisl.rate)?
  2. 吃这块肉要付多少 ho 代价 (greedy.ho - maxisl.ho)?
"""
import os, sys, numpy as np, random
os.environ['LEO_CAHS_VARIANT'] = 'paper'; os.environ['LEO_QUIET_LOGS'] = '1'
os.environ['LEO_C_BAND'] = '100'
sys.path.insert(0, '.')
np.random.seed(0); random.seed(42)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--const', default='C')
ap.add_argument('--users', type=int, default=200)
ap.add_argument('--eps', type=int, default=50)
args = ap.parse_args()

from train_drqn import build_drqn_env
env = build_drqn_env(args.users, args.const)
users = list(env.topo.user)

def fb_of(u, sat):
    fd = env._get_feeder_sat(u, sat)
    return (env.net.N2N_status[sat.con_id-1][sat.ID-1][fd.ID-1].free_band
            if fd is not None and sat != fd else 9999)

def maxisl_pick(u):
    best_sat, best_fb = None, -1
    for sat in u.sat_covered:
        fb = fb_of(u, sat)
        if fb > best_fb:
            best_fb = fb; best_sat = sat
    return best_sat.ID-1 if best_sat else 0

def greedy_pick(u):
    best_sat, best_val = None, -1
    for sat in u.sat_covered:
        cq = env.ho[u][sat].c_quality
        val = min(cq, fb_of(u, sat))
        if val > best_val:
            best_val = val; best_sat = sat
    return best_sat.ID-1 if best_sat else 0

def run(pick_fn, eps):
    env.reset(0, 'NETWORK_LOAD')
    for ep in range(10):
        for u in users:
            env.step({u: pick_fn(u)+1}, 'NETWORK')
        env.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    sum_rate = 0.0; n = 0; ho = 0; ho_den = 0
    prev = {u: None for u in users}
    for ep in range(eps):
        for u in users:
            env.step({u: pick_fn(u)+1}, 'NETWORK')
            if u.sat_connected:
                cq = env.ho[u][u.sat_connected].c_quality
                sum_rate += min(cq, fb_of(u, u.sat_connected)); n += 1
            cur = u.sat_connected.ID if u.sat_connected else None
            if prev[u] is not None and cur != prev[u]:
                ho += 1
            ho_den += 1
            prev[u] = cur
        env.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    return sum_rate/n, ho/ho_den

print(f"=== L0 greedy: {args.const}x{args.users} eps={args.eps} ===", flush=True)
r, h = run(maxisl_pick, args.eps)
print(f"[maxisl ] rate={r:.0f}  ho={h:.3f}", flush=True)
r, h = run(greedy_pick, args.eps)
print(f"[greedy ] rate={r:.0f}  ho={h:.3f}", flush=True)
print("=== L0 greedy done ===", flush=True)
