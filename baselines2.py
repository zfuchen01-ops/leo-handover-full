#!/usr/bin/env python3
"""直接用 env 内部数据算 MaxISL/MGCS/Random, 不经过 Observe 特征"""
import os, sys, numpy as np, random
os.environ['LEO_CAHS_VARIANT'] = 'paper'; os.environ['LEO_QUIET_LOGS'] = '1'
os.environ['LEO_C_BAND'] = '100'
sys.path.insert(0, '.')
np.random.seed(0); random.seed(42)

from train_drqn import build_drqn_env

env = build_drqn_env(200, 'C')
users = list(env.topo.user)

def max_isl_pick(u):
    """直接查 N2N free_band, 跟 rate 计算一样的路径"""
    best_sat, best_fb = None, -1
    for sat in u.sat_covered:
        fd = env._get_feeder_sat(u, sat)
        if fd is not None and sat != fd:
            fb = env.net.N2N_status[sat.con_id-1][sat.ID-1][fd.ID-1].free_band
        else:
            fb = 9999
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

def run_baseline(name, pick_fn):
    env.reset(0, 'NETWORK_LOAD')
    for ep in range(10):
        for u in users:
            env.step({u: pick_fn(u)+1}, 'NETWORK')
        env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

    rates = []
    for ep in range(100):
        ep_rate, n = 0.0, 0
        for u in users:
            act = pick_fn(u)
            env.step({u: act+1}, 'NETWORK')
            if u.sat_connected and u.sat_connected in env.ho[u]:
                fd = env._get_feeder_sat(u, u.sat_connected)
                b = (env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band
                     if fd and u.sat_connected != fd else 9999)
                ep_rate += min(env.ho[u][u.sat_connected].c_quality, b)
                n += 1
        rates.append(ep_rate/n if n else 0)
        env.Update_Env((ep+1)*30, 'NETWORK_LOAD')
    avg = sum(rates)/100
    print(f"{name:8s}: rate={avg:.0f}", flush=True)
    return avg

run_baseline("MGCS", mgcs_pick)
run_baseline("MaxISL", max_isl_pick)
run_baseline("Random", lambda u: random.choice([s.ID-1 for s in u.sat_covered]) if u.sat_covered else 0)
