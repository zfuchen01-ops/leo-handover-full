#!/usr/bin/env python3
import os, random, math
from math import pi as _pi
os.environ['LEO_CAHS_VARIANT'] = 'paper'
os.environ['LEO_QUIET_LOGS'] = '1'
os.environ['LEO_C_BAND'] = '100'
os.environ['LEO_PAPER_PT_GT_GS_DB'] = '60'
from Topology import Topology; from User import User
from Network import Network; from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI

for s in ['mgcs', 'maxisl']:
    Topology.index_con = 0; User.uid = 0
    topo = Topology(); o, s2 = 16, 16; ph = 1
    fp = 2.0*ph*_pi/(o*s2); lean = 54.0/180.0*_pi; theta = 2.0*_pi/o
    topo.Add_Constellation(o, s2, SAT_HEIGHT, fp, lean, theta, TYPE_2PI)
    topo.Each_Satellite()
    for lo,la in [(0,0),(60,0),(120,0),(180,0),(-120,0),(-60,0)]:
        topo.Add_Gateway_Loc(lo/180*_pi, la/180*_pi, antenna_Num=5, name_str='GW')
    random.seed(42)
    us = [(random.uniform(-120,120)*_pi/180, random.uniform(-60,60)*_pi/180) for _ in range(200)]
    topo.Add_User_From_Input(us)
    for u in topo.user: u.assigned_gateway = topo.gateway[random.randrange(6)]
    env = PaperRebuildHandover(net=Network(topo))
    rates, cqs = [], []
    env.reset(0, 'NETWORK_LOAD'); tm = 0
    for ep in range(20):
        for user in env.topo.user:
            fd = env._get_feeder_sat(user); best, bv = None, -1
            for sat in env.ho[user]:
                if s == 'mgcs': v = env.ho[user][sat].c_quality
                else:
                    v = 0
                    if fd and sat != fd: v = env.net.N2N_status[sat.con_id-1][sat.ID-1][fd.ID-1].free_band
                    elif fd and sat == fd: v = 2000.0
                if v > bv: bv = v; best = sat
            if best: env.step({user: best.ID}, 'INITIAL' if ep == 0 else 'NETWORK')
        er, n = 0.0, 0
        for u in env.topo.user:
            if u.sat_connected and u.sat_connected in env.ho[u]:
                cqs.append(env.ho[u][u.sat_connected].c_quality)
                fd = env._get_feeder_sat(u, u.sat_connected); isl = 9999
                if fd and u.sat_connected != fd:
                    isl = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band
                er += min(env.ho[u][u.sat_connected].c_quality, isl); n += 1
        rates.append(er/n if n else 0)
        tm += 50; env.Update_Env(tm, 'NETWORK_LOAD')
    env.close()
    print(f'{s}: avg={sum(rates[5:])/len(rates[5:]):.0f} cq={min(cqs):.0f}-{max(cqs):.0f}({sum(cqs)/len(cqs):.0f})', flush=True)
