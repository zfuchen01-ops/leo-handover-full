#!/usr/bin/env python3
"""L0 诊断: 瓶颈统计 + 上界估计 (零训练, 直接复用 env 内部数据)

回答两个问题:
  1. 当前 rate = min(access, fb) 里, 到底是下行(access)还是 feeder(fb)在卡?
  2. 如果只松开一侧瓶颈, rate 天花板是多少?

三个量 (都在某个策略的连接下逐用户统计):
  actual        = mean(min(access, fb))      当前实际速率
  downlink_ceil = mean(access)               假设 feeder 无限(fb=inf)时的上限
  feeder_ceil   = mean(fb)                   假设下行无限(access=inf)时的上限
  access-bound / fb-bound 占比: min 取到哪一侧
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
ap.add_argument('--eps', type=int, default=100)
args = ap.parse_args()

from train_drqn import build_drqn_env
env = build_drqn_env(args.users, args.const)
users = list(env.topo.user)

def max_isl_pick(u):
    best_sat, best_fb = None, -1
    for sat in u.sat_covered:
        fd = env._get_feeder_sat(u, sat)
        fb = (env.net.N2N_status[sat.con_id-1][sat.ID-1][fd.ID-1].free_band
              if fd is not None and sat != fd else 9999)
        if fb > best_fb:
            best_fb = fb; best_sat = sat
    return best_sat.ID-1 if best_sat else 0

def mgcs_pick(u):
    best_sat, best_cq = None, -1
    for sat in u.sat_covered:
        cq = env.ho[u][sat].c_quality
        if cq > best_cq:
            best_cq = cq; best_sat = sat
    return best_sat.ID-1 if best_sat else 0

def run(strategy, pick):
    env.reset(0, 'NETWORK_LOAD')
    for ep in range(10):
        for u in users:
            env.step({u: pick(u)+1}, 'NETWORK')
        env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

    sum_min = sum_access = sum_fb = 0.0
    n_access = n_fb = n = 0
    for ep in range(args.eps):
        for u in users:
            env.step({u: pick(u)+1}, 'NETWORK')
            if u.sat_connected and u.sat_connected in env.ho[u]:
                fd = env._get_feeder_sat(u, u.sat_connected)
                fb = (env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band
                      if fd and u.sat_connected != fd else 9999)
                access = env.ho[u][u.sat_connected].c_quality
                sum_min += min(access, fb); sum_access += access; sum_fb += fb
                n += 1
                if access <= fb:
                    n_access += 1
                else:
                    n_fb += 1
        env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

    print(f"[{strategy:7s}] actual={sum_min/n:.0f}  downlink_ceil={sum_access/n:.0f}  "
          f"feeder_ceil={sum_fb/n:.0f}  |  access-bound={n_access/n*100:.0f}%  "
          f"fb-bound={n_fb/n*100:.0f}%", flush=True)

print(f"=== L0 diag: {args.const}x{args.users} eps={args.eps} ===", flush=True)
run("maxisl", max_isl_pick)
run("mgcs", mgcs_pick)
print("=== L0 done ===", flush=True)
