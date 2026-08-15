#!/usr/bin/env python3
"""L0 状态诊断: 量化 RL 动作空间裁剪的损失 + 特征饱和。

回答:
  1. 每用户可见星(sat_covered) / 可动作星(ho 排序前12) 各多少颗?
  2. 下行 cq 与 feeder fb 的分布, fb 是否频繁 > RATE_UPPER (特征饱和)?
  3. top-12 by ID 裁剪后, 丢失的星里有没有高 cq 的星 (RL 看不到的肉)?
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
ap.add_argument('--eps', type=int, default=30)
args = ap.parse_args()

from train_drqn import build_drqn_env
from Handover import RATE_UPPER
env = build_drqn_env(args.users, args.const)
users = list(env.topo.user)

print(f"RATE_UPPER = {RATE_UPPER:.1f}", flush=True)

def max_isl_pick(u):
    best_sat, best_fb = None, -1
    for sat in u.sat_covered:
        fd = env._get_feeder_sat(u, sat)
        fb = (env.net.N2N_status[sat.con_id-1][sat.ID-1][fd.ID-1].free_band
              if fd is not None and sat != fd else 9999)
        if fb > best_fb:
            best_fb = fb; best_sat = sat
    return best_sat.ID-1 if best_sat else 0

env.reset(0, 'NETWORK_LOAD')
for ep in range(10):
    for u in users:
        env.step({u: max_isl_pick(u)+1}, 'NETWORK')
    env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

# 统计
cover_sizes = []
ho_sizes = []
all_cq = []
all_fb = []
cq_lost = []   # 被 top-12 裁剪掉的高 cq 星 (相对该用户可见集)
fb_ge_upper = 0
fb_total = 0
n_snap = 0

for ep in range(args.eps):
    for u in users:
        env.step({u: max_isl_pick(u)+1}, 'NETWORK')
        cover_sizes.append(len(u.sat_covered))
        ho_sizes.append(len(env.ho[u]))
        # 可见星的 cq 分布 + 全量最优
        cqs = []
        best_cq_full = -1
        for sat in u.sat_covered:
            cq = env.ho[u][sat].c_quality
            cqs.append((cq, sat.ID))
            all_cq.append(cq)
            if cq > best_cq_full:
                best_cq_full = cq
            fd = env._get_feeder_sat(u, sat)
            fb = (env.net.N2N_status[sat.con_id-1][sat.ID-1][fd.ID-1].free_band
                  if fd is not None and sat != fd else 9999)
            all_fb.append(fb)
            fb_total += 1
            if fb >= RATE_UPPER:
                fb_ge_upper += 1
        # top-12 by ID 裁剪: 模拟 Observe 的 sort by ID 取前12
        cqs_sorted = sorted(cqs, key=lambda x: x[1])[:12]
        best_cq_top12 = max(c for c, _ in cqs_sorted) if cqs_sorted else -1
        # 丢失的高 cq 星: 全量最优 cq 在 top-12 之外
        if best_cq_full > best_cq_top12:
            cq_lost.append(best_cq_full - best_cq_top12)
        n_snap += 1
    env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

cq = np.array(all_cq); fb = np.array(all_fb)
print(f"=== L0 state: {args.const}x{args.users} ({n_snap} user-snapshots) ===", flush=True)
print(f"可见星 sat_covered: mean={np.mean(cover_sizes):.1f}  p50={np.median(cover_sizes):.0f}  p90={np.percentile(cover_sizes,90):.0f}  max={max(cover_sizes)}", flush=True)
print(f"可动作星 ho[user]: mean={np.mean(ho_sizes):.1f}  p50={np.median(ho_sizes):.0f}  max={max(ho_sizes)}", flush=True)
print(f"下行 cq: mean={cq.mean():.0f}  p50={np.median(cq):.0f}  p90={np.percentile(cq,90):.0f}  max={cq.max():.0f}", flush=True)
print(f"feeder fb: mean={fb.mean():.0f}  p50={np.median(fb):.0f}  p90={np.percentile(fb,90):.0f}  max={fb.max():.0f}", flush=True)
print(f"fb >= RATE_UPPER({RATE_UPPER:.0f}) 的比例 = {fb_ge_upper/fb_total*100:.1f}%  (特征饱和比例)", flush=True)
if cq_lost:
    cl = np.array(cq_lost)
    print(f"top-12 by ID 丢失的高cq星: {len(cq_lost)}/{n_snap} 用户 ({(len(cq_lost)/n_snap)*100:.1f}%), 平均丢失 cq={cl.mean():.0f}", flush=True)
else:
    print(f"top-12 by ID 未丢失高cq星", flush=True)
print("=== L0 state done ===", flush=True)
