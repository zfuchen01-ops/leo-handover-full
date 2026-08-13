#!/usr/bin/env python3
"""三基线: MGCS + Random + MaxISL, 各3×50ep, 去前10ep取平均"""
import sys, os
from math import pi as _pi
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from Handover import RATE_UPPER
import random

def run_one_trial(N_EP, strategy):
    """strategy: 'mgcs' | 'random' | 'maxisl'"""
    Topology.index_con = 0; User.uid = 0
    topo = Topology()
    o, s = 16, 16; phase = 1
    first_phi = 2.0*phase*_pi/(o*s); lean = 54.0/180.0*_pi; theta = 2.0*_pi/o
    topo.Add_Constellation(o, s, SAT_HEIGHT, first_phi, lean, theta, TYPE_2PI)
    topo.Each_Satellite()
    for lon,lat,name in [(0,0,"GW1"),(60,0,"GW2"),(120,0,"GW3"),(180,0,"GW4"),(-120,0,"GW5"),(-60,0,"GW6")]:
        topo.Add_Gateway_Loc(lon/180*_pi, lat/180*_pi, antenna_Num=5, name_str=name)
    random.seed(42)
    users = [(random.uniform(-120,120)*_pi/180, random.uniform(-60,60)*_pi/180) for _ in range(200)]
    topo.Add_User_From_Input(users)
    for u in topo.user: u.assigned_gateway = topo.gateway[random.randrange(6)]
    env = PaperRebuildHandover(net=Network(topo))
    rates, hos, uniqs, beams = [], [], [], []
    env.reset(0, 'NETWORK_LOAD'); time = 0

    for ep in range(N_EP):
        for user in env.topo.user:
            if strategy == 'mgcs':
                best, best_val = None, -1
                for sat in env.ho[user]:
                    q = env.ho[user][sat].c_quality
                    if q > best_val: best_val = q; best = sat
            elif strategy == 'random':
                cand = [sat for sat in env.ho[user]]
                best = random.choice(cand) if cand else None
            elif strategy == 'maxisl':
                fd = env._get_feeder_sat(user)
                best, best_val = None, -1
                for sat in env.ho[user]:
                    fb = 0
                    if fd and sat != fd:
                        fb = env.net.N2N_status[sat.con_id-1][sat.ID-1][fd.ID-1].free_band
                    elif fd and sat == fd:
                        fb = 2000.0  # 直连feeder, ISL无限
                    if fb > best_val: best_val = fb; best = sat
            if best is None: continue
            env.step({user: best.ID}, 'INITIAL' if ep==0 else 'NETWORK')

        ep_r, ep_ho, n = 0.0, 0.0, 0
        for u in env.topo.user:
            if u.sat_connected and u.sat_connected in env.ho[u]:
                fd = env._get_feeder_sat(u, u.sat_connected)
                isl = RATE_UPPER
                if fd and u.sat_connected != fd:
                    isl = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band
                r = min(env.ho[u][u.sat_connected].c_quality, isl, RATE_UPPER)
                ep_r += r; ep_ho += 0.0 if u.sat_connected == u.last_connected else 1.0; n += 1
        rates.append(ep_r/n if n else 0)
        hos.append(ep_ho/n if n else 0)
        conn = [u.sat_connected for u in env.topo.user if u.sat_connected]
        uniq = len(set(s.ID for s in conn))
        uniqs.append(uniq)
        beams.append(n / max(1, uniq * 64.0))  # 每星平均波束利用率
        time += 50; env.Update_Env(time, 'NETWORK_LOAD')
    env.close()
    return rates, hos, uniqs, beams

for strategy in ['mgcs', 'random', 'maxisl']:
    tr, th, tu, tb = [], [], [], []
    for t in range(3):
        r, h, u, b = run_one_trial(50, strategy)
        tr.append(r); th.append(h); tu.append(u); tb.append(b)
    vr = [x for r in tr for x in r[10:]]
    vh = [x for h in th for x in h[10:]]
    vu = [x for u in tu for x in u[10:]]
    vb = [x for b in tb for x in b[10:]]
    print(f"{strategy:>7}: rate={sum(vr)/len(vr):.0f} ho={sum(vh)/len(vh):.3f} uniq={sum(vu)/len(vu):.0f} beam={sum(vb)/len(vb):.3f}")
