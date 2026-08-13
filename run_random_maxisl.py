#!/usr/bin/env python3
"""Random + MaxISL baselines (with RATE_UPPER)"""
import sys, os, random
from math import pi as _pi
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")
from Topology import Topology; from User import User
from Network import Network; from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI

for strat in ['random', 'maxisl']:
    tr, th, tu, tb = [], [], [], []
    for t in range(3):
        Topology.index_con = 0; User.uid = 0
        topo = Topology()
        o, s = 16, 16; ph = 1; fp = 2.0*ph*_pi/(o*s)
        topo.Add_Constellation(o, s, SAT_HEIGHT, fp, 54.0/180.0*_pi, 2.0*_pi/o, TYPE_2PI)
        topo.Each_Satellite()
        for lo, la in [(0,0),(60,0),(120,0),(180,0),(-120,0),(-60,0)]:
            topo.Add_Gateway_Loc(lo/180*_pi, la/180*_pi, antenna_Num=5, name_str=f'GW_{lo}_{la}')
        random.seed(42)
        us = [(random.uniform(-120,120)*_pi/180, random.uniform(-60,60)*_pi/180) for _ in range(200)]
        topo.Add_User_From_Input(us)
        for u in topo.user: u.assigned_gateway = topo.gateway[random.randrange(6)]
        env = PaperRebuildHandover(net=Network(topo))
        rt, ho, uni, be = [], [], [], []
        env.reset(0, 'NETWORK_LOAD'); tm = 0
        for ep in range(50):
            for user in env.topo.user:
                if strat == 'random':
                    cand = [sat for sat in env.ho[user]]
                    best = random.choice(cand) if cand else None
                else:
                    fd = env._get_feeder_sat(user); best, bv = None, -1
                    for sat in env.ho[user]:
                        fb = 0
                        if fd and sat != fd: fb = env.net.N2N_status[sat.con_id-1][sat.ID-1][fd.ID-1].free_band
                        elif fd and sat == fd: fb = 2000.0
                        if fb > bv: bv = fb; best = sat
                if best is None: continue
                env.step({user: best.ID}, 'INITIAL' if ep == 0 else 'NETWORK')
            er, eh, n = 0.0, 0.0, 0
            for u in env.topo.user:
                if u.sat_connected and u.sat_connected in env.ho[u]:
                    fd = env._get_feeder_sat(u, u.sat_connected)
                    isl = 2000.0
                    if fd and u.sat_connected != fd:
                        isl = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band
                    er += min(env.ho[u][u.sat_connected].c_quality, isl)
                    eh += 0.0 if u.sat_connected == u.last_connected else 1.0
                    n += 1
            rt.append(er/n if n else 0); ho.append(eh/n if n else 0)
            cn = [u.sat_connected for u in env.topo.user if u.sat_connected]
            un = len(set(s.ID for s in cn)); uni.append(un)
            be.append(n/max(1, un*64.0))
            tm += 50; env.Update_Env(tm, 'NETWORK_LOAD')
        env.close()
        tr.append(rt); th.append(ho); tu.append(uni); tb.append(be)
        print(f'  {strat} trial {t+1}/3 done', flush=True)
    vr = [x for r in tr for x in r[10:]]; vh = [x for h in th for x in h[10:]]
    vu = [x for u in tu for x in u[10:]]; vb = [x for b in tb for x in b[10:]]
    print(f'{strat:>7}: rate={sum(vr)/len(vr):.0f} ho={sum(vh)/len(vh):.3f} uniq={sum(vu)/len(vu):.0f} beam={sum(vb)/len(vb):.3f}', flush=True)
print('DONE')
