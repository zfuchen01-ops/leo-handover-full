#!/usr/bin/env python3
"""环境诊断: 打印关键物理参数 + cq/ISL范围."""
import os
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

from Handover import RATE_UPPER, C_BAND, EIRP, GS, LIGHT_VEL, SAT_HEIGHT, CARRIER, GAUSE_NOISE, RICE, RAIN
from Defination import USER_ELEVATION
from math import pi as _pi
import random; random.seed(42)
import numpy as np; np.random.seed(0)

print(f"C_BAND={C_BAND}")
print(f"RATE_UPPER={RATE_UPPER:.0f}")
print(f"EIRP={EIRP:.1e} GS={GS:.1e}")
print(f"SAT_HEIGHT={SAT_HEIGHT}")
print(f"CARRIER={CARRIER:.1e}")
print(f"GAUSE_NOISE={GAUSE_NOISE:.1e}")
print(f"RICE={RICE} RAIN={RAIN}")
print(f"USER_ELEVATION={USER_ELEVATION}")
print(f"LIGHT_VEL={LIGHT_VEL:.1e}")

# 构建环境采样cq
from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT as SH, TYPE_2PI

Topology.index_con = 0; User.uid = 0
t = Topology()
o, s = 16, 16; p = 1
fp = 2.0 * p * _pi / (o * s)
ln = 54.0 / 180.0 * _pi
th = 2.0 * _pi / o
t.Add_Constellation(o, s, SH, fp, ln, th, TYPE_2PI)
t.Each_Satellite()
for lon, lat, nm in [(0,0,"G1"),(60,0,"G2"),(120,0,"G3"),(180,0,"G4"),(-120,0,"G5"),(-60,0,"G6")]:
    t.Add_Gateway_Loc(lon/180.0*_pi, lat/180.0*_pi, 5, nm)
users = [(random.uniform(-120,120)*_pi/180.0, random.uniform(-60,60)*_pi/180.0) for _ in range(200)]
t.Add_User_From_Input(users)
for u in t.user: u.assigned_gateway = t.gateway[random.randrange(len(t.gateway))]

env = PaperRebuildHandover(net=Network(t))
env.reset(0, 'NETWORK_LOAD')

cqs, fbs = [], []
for u in env.topo.user:
    if u.sat_connected and u.sat_connected in env.ho[u]:
        cqs.append(env.ho[u][u.sat_connected].c_quality)
        fd = env._get_feeder_sat(u, u.sat_connected)
        if fd and u.sat_connected != fd:
            fb = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band
            fbs.append(fb)

cqs.sort()
fbs.sort()
print(f"cq: [{cqs[0]:.0f}, {cqs[-1]:.0f}] med={cqs[len(cqs)//2]:.0f} n={len(cqs)}")
print(f"fb: [{fbs[0]:.0f}, {fbs[-1]:.0f}] med={fbs[len(fbs)//2]:.0f} n={len(fbs)}")
print(f"rate_if_noISL: cq min={min(cqs):.0f}, cq avg={sum(cqs)/len(cqs):.0f}")
print(f"rate_if_ISL: min(cq,fb) avg={sum(min(a,b) for a,b in zip(cqs,fbs))/len(cqs):.0f}")
