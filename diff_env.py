#!/usr/bin/env python3
"""对比本地 vs 远程关键参数, 找出 rate 差异根因."""
import paramiko
HOST = "connect.westd.seetacloud.com"
PORT = 34172

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password="f/oqAQ+57I15", timeout=15)

def remote(cmd):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=60)
    return stdout.read().decode(), stderr.read().decode()

# 诊断脚本: 打印所有关键参数
diag = """
import os
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")
from Handover import RATE_UPPER, C_BAND, EIRP, GS, LIGHT_VEL, SAT_HEIGHT, CARRIER, GAUSE_NOISE, RICE, RAIN
from Defination import USER_ELEVATION, SAT_HEIGHT as SH_DEF
print(f"C_BAND={C_BAND}")
print(f"RATE_UPPER={RATE_UPPER:.0f}")
print(f"EIRP={EIRP:.1f} GS={GS:.1f}")
print(f"SAT_HEIGHT={SAT_HEIGHT}")
print(f"CARRIER={CARRIER:.1e}")
print(f"GAUSE_NOISE={GAUSE_NOISE:.1e}")
print(f"RICE={RICE} RAIN={RAIN}")
print(f"USER_ELEVATION={USER_ELEVATION}")

# 打印 cq 范围
from math import pi as _pi, log2
import random
random.seed(42)
import numpy as np; np.random.seed(0)

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI

Topology.index_con = 0; User.uid = 0
t = Topology()
o, s = 16, 16; p = 1
fp = 2.0 * p * _pi / (o * s)
ln = 54.0 / 180.0 * _pi
th = 2.0 * _pi / o
t.Add_Constellation(o, s, SAT_HEIGHT, fp, ln, th, TYPE_2PI)
t.Each_Satellite()
for lon, lat, nm in [(0,0,"G1"),(60,0,"G2"),(120,0,"G3"),(180,0,"G4"),(-120,0,"G5"),(-60,0,"G6")]:
    t.Add_Gateway_Loc(lon/180.0*_pi, lat/180.0*_pi, 5, nm)
users = [(random.uniform(-120,120)*_pi/180.0, random.uniform(-60,60)*_pi/180.0) for _ in range(5)]
t.Add_User_From_Input(users)
for u in t.user: u.assigned_gateway = t.gateway[random.randrange(len(t.gateway))]

env = PaperRebuildHandover(net=Network(t))
env.reset(0, 'NETWORK_LOAD')

# 采样 cq 值
cqs = []
for u in env.topo.user:
    for sat in env.ho[u]:
        cqs.append(env.ho[u][sat].c_quality)
if cqs:
    cqs.sort()
    print(f"cq 范围: [{cqs[0]:.0f}, {cqs[-1]:.0f}] 中位数={cqs[len(cqs)//2]:.0f} 样本数={len(cqs)}")
    print(f"cq samples: {[f'{x:.0f}' for x in cqs[:5]]} ... {[f'{x:.0f}' for x in cqs[-5:]]}")

# ISL free_band
fbs = []
for u in env.topo.user:
    if u.sat_connected and u.sat_connected in env.ho[u]:
        fd = env._get_feeder_sat(u, u.sat_connected)
        if fd and u.sat_connected != fd:
            fb = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band
            fbs.append(fb)
if fbs:
    fbs.sort()
    print(f"ISL free_band: [{fbs[0]:.0f}, {fbs[-1]:.0f}] 中位数={fbs[len(fbs)//2]:.0f}")
"""

out, err = remote(f"cd /root/leo_handover && python3 -c '{diag}'")
print("=== 远程GPU ===")
print(out)
if err.strip():
    print(f"ERR: {err.strip()[:500]}")
c.close()
