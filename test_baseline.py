#!/usr/bin/env python3
"""本地验证修正后baseline, 10ep."""
import os, sys, time
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

from math import pi as _pi
import random as _random, numpy as np; np.random.seed(0); _random.seed(42)
from collections import defaultdict

from Topology import Topology; from User import User
from Network import Network; from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from Handover import Handover, RATE_UPPER, Calc_Sphere_Elevation

def reset_ids():
    Topology.index_con = 0; User.uid = 0

def build_env(n=200):
    reset_ids(); t = Topology(); o, s = 16, 16; p = 1
    fp = 2.0 * p * _pi / (o * s); ln = 54.0 / 180.0 * _pi; th = 2.0 * _pi / o
    t.Add_Constellation(o, s, SAT_HEIGHT, fp, ln, th, TYPE_2PI); t.Each_Satellite()
    for lon, lat, nm in [(0,0,"G1"),(60,0,"G2"),(120,0,"G3"),(180,0,"G4"),(-120,0,"G5"),(-60,0,"G6")]:
        t.Add_Gateway_Loc(lon/180.0*_pi, lat/180.0*_pi, 5, nm)
    users = [(_random.uniform(-120,120)*_pi/180.0, _random.uniform(-60,60)*_pi/180.0) for _ in range(n)]
    t.Add_User_From_Input(users)
    for u in t.user: u.assigned_gateway = t.gateway[_random.randrange(len(t.gateway))]
    return PaperRebuildHandover(net=Network(t))

def user_rate_and_ho(env, user, ep):
    rate = ho = 0.0
    if user.sat_connected and user.sat_connected in env.ho[user]:
        fd = env._get_feeder_sat(user, user.sat_connected)
        fb = 9999.0
        if fd and user.sat_connected != fd:
            fb = env.net.N2N_status[user.sat_connected.con_id-1][user.sat_connected.ID-1][fd.ID-1].free_band
        rate = min(env.ho[user][user.sat_connected].c_quality, fb)
        if ep > 0 and user.last_connected is not None and user.sat_connected != user.last_connected:
            ho = 1.0
    return rate, ho

def run_baseline(env, n_ep, strategy):
    n_users = len(env.topo.user)
    rates, hos = [], []
    env.reset(0, 'NETWORK_LOAD')
    for ep in range(n_ep):
        ep_rate, ep_ho = 0.0, 0.0
        for u in env.topo.user:
            visible = list(env.ho[u].keys())
            if not visible: continue
            best_sat = None
            if strategy == "MGCS":
                best_val = -1
                for sat in visible:
                    cq = env.ho[u][sat].c_quality
                    if cq > best_val: best_val = cq; best_sat = sat
            elif strategy == "RAND":
                best_sat = _random.choice(visible)
            env.step({u: best_sat.ID}, 'INITIAL' if ep == 0 else 'NETWORK')
            r, h = user_rate_and_ho(env, u, ep)
            ep_rate += r; ep_ho += h
        rates.append(ep_rate / n_users)  # ← /200
        hos.append(ep_ho / n_users)
        env.Update_Env((ep + 1) * 50, 'NETWORK_LOAD')
    return sum(rates)/len(rates), sum(hos)/len(hos)

print(f"RATE_UPPER={RATE_UPPER:.0f}\n", flush=True)
t0 = time.time()
env = build_env(200)

for st in ["MGCS", "RAND"]:
    r, h = run_baseline(env, 10, st)
    print(f"{st}: rate={r:.0f} HO={h:.3f} (10ep, /200用户)", flush=True)

print(f"\n{time.time()-t0:.0f}s")
print("修正后 MGCS应该~300, 不再是477")
