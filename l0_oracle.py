#!/usr/bin/env python3
"""L0 oracle 上界: 每个用户在所有可达卫星里取 min(access,fb) 最大。

这给出 rate 的"物理上界"(忽略切换代价/冲突), 用来判断还有多少肉:
  oracle - maxisl 的差 = 联合优化(下行+ISL)还能挖的空间。
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

def fb_of(u, sat):
    fd = env._get_feeder_sat(u, sat)
    return (env.net.N2N_status[sat.con_id-1][sat.ID-1][fd.ID-1].free_band
            if fd is not None and sat != fd else 9999)

# 用 maxisl 驱动 env 前进, 每步额外算 oracle 上界
env.reset(0, 'NETWORK_LOAD')
for ep in range(10):
    for u in users:
        env.step({u: max_isl_pick(u)+1}, 'NETWORK')
    env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

sum_maxisl = sum_oracle = 0.0
n = 0
n_oracle_access = n_oracle_fb = 0
for ep in range(args.eps):
    for u in users:
        env.step({u: max_isl_pick(u)+1}, 'NETWORK')
        # actual (maxisl 连接)
        if u.sat_connected and u.sat_connected in env.ho[u]:
            acc = env.ho[u][u.sat_connected].c_quality
            fb = fb_of(u, u.sat_connected)
            sum_maxisl += min(acc, fb)
            n += 1
        # oracle: max over covered sats of min(access, fb)
        best = -1
        best_access = False
        for sat in u.sat_covered:
            acc2 = env.ho[u][sat].c_quality
            fb2 = fb_of(u, sat)
            m = min(acc2, fb2)
            if m > best:
                best = m
                best_access = (acc2 <= fb2)
        sum_oracle += best
        if best_access:
            n_oracle_access += 1
        else:
            n_oracle_fb += 1
    env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

print(f"=== L0 oracle: {args.const}x{args.users} eps={args.eps} ===", flush=True)
print(f"maxisl_actual = {sum_maxisl/n:.0f}", flush=True)
print(f"oracle_upper  = {sum_oracle/n:.0f}   (联合优化空间 = {sum_oracle/n - sum_maxisl/n:+.0f})", flush=True)
print(f"oracle 连接下: access-bound={n_oracle_access/(n or 1)*100:.0f}%  fb-bound={n_oracle_fb/(n or 1)*100:.0f}%", flush=True)
print("=== L0 oracle done ===", flush=True)
