#!/usr/bin/env python3
"""卫星切换论文标准Baseline: MGCS/MRVT/Max-Elevation/Min-Load/Random.
引用: He2020, Liu2022(SDQL), Lee2025(MADQN), Badini2025(SMASH), Yang2024(MAF-DDQN), Tong2025(A2C)
用法: python baselines.py [--users 200] [--ep 20]
"""
import argparse, os, sys
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

from math import pi as _pi
import random as _random, numpy as np; np.random.seed(0); _random.seed(42)
from collections import defaultdict

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from Handover import Handover, RATE_UPPER, Calc_Sphere_Elevation

def reset_ids():
    Topology.index_con = 0; User.uid = 0

def build_env(n=200, constellation='C'):
    reset_ids(); t = Topology()
    if constellation == 'A': o, s = 8, 9
    elif constellation == 'B': o, s = 12, 12
    elif constellation == 'C': o, s = 16, 16
    elif constellation == 'D': o, s = 20, 20
    else: raise ValueError(f"unknown constellation: {constellation}")
    p = 1
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
    """策略名对应论文:
    MGCS  (Max CQ)         - Lee2025, Tong2025, Yang2024, YangTSGH-REB
    MRVT  (Max RVT)        - He2020, Lee2025, Badini2025, Yang2024
    ELEV  (Max Elevation)  - Liu2022(SDQL), Liu2024, Badini2024
    MINL  (Min Load)       - He2020, Liu2022(SDQL), Badini2025, Tong2025
    RAND  (Random)         - Lee2025, Tong2025, Yang2024, Liu2024
    """
    n_users = len(env.topo.user)
    rates, hos, beam_loads, uniq_sats = [], [], [], []
    env.reset(0, 'NETWORK_LOAD')

    for ep in range(n_ep):
        ep_rate, ep_ho, n_conn = 0.0, 0.0, 0
        sat_loads = defaultdict(int)

        for u in env.topo.user:
            visible = list(env.ho[u].keys())
            if not visible:
                env.step({u: 0}, 'INITIAL' if ep == 0 else 'NETWORK')
                continue

            best_sat = None
            if strategy == "MGCS":
                best_val = -1
                for sat in visible:
                    cq = env.ho[u][sat].c_quality
                    if cq > best_val: best_val = cq; best_sat = sat
            elif strategy == "MRVT":
                best_val = -1
                for sat in visible:
                    rvt = env._compute_rvt(u, sat)
                    if rvt > best_val: best_val = rvt; best_sat = sat
            elif strategy == "ELEV":
                best_val = -1
                for sat in visible:
                    elev = Calc_Sphere_Elevation(sat.we_pos, u.we_pos)
                    if elev > best_val: best_val = elev; best_sat = sat
            elif strategy == "MINL":
                best_val = 999
                for sat in visible:
                    ld = sum(1 for u2 in env.topo.user if u2.sat_connected and u2.sat_connected.ID == sat.ID)
                    if ld < best_val: best_val = ld; best_sat = sat
            elif strategy == "RAND":
                best_sat = _random.choice(visible)

            env.step({u: best_sat.ID}, 'INITIAL' if ep == 0 else 'NETWORK')
            r, h = user_rate_and_ho(env, u, ep)
            ep_rate += r; ep_ho += h
            if r > 0: n_conn += 1
            if u.sat_connected:
                sat_loads[u.sat_connected.ID] += 1

        rates.append(ep_rate / n_users)  # 除以全部用户, 含未连接的
        hos.append(ep_ho / n_users)
        beam_loads.append(sum(v / 128.0 for v in sat_loads.values()) / max(1, len(sat_loads)))
        uniq_sats.append(len(sat_loads))

        env.Update_Env((ep + 1) * 50, 'NETWORK_LOAD')

    return (sum(rates)/len(rates), sum(hos)/len(hos),
            sum(beam_loads)/len(beam_loads), sum(uniq_sats)/len(uniq_sats))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--ep", type=int, default=20)
    parser.add_argument("--constellation", default="C", choices=["A","B","C","D"])
    args = parser.parse_args()

    print(f"构建环境 (星座{args.constellation} N={args.users})...", flush=True)
    import time; t0 = time.time()
    env = build_env(args.users, args.constellation)
    print(f"  卫星={env.topo.total_sat}, {time.time()-t0:.0f}s\n", flush=True)

    strategies = [
        ("MGCS", "Max CQ (所有论文)"),
        ("MRVT", "Max RVT (He, Lee, Badini)"),
        ("ELEV", "Max Elevation (Liu22, Badini)"),
        ("MINL", "Min Load (He, Liu22, Badini)"),
        ("RAND", "Random (Lee, Tong, Yang)"),
    ]

    results = {}
    for st, label in strategies:
        print(f"--- {label} ({args.ep}ep) ---", flush=True)
        t0 = time.time()
        rate, ho, beam, uniq = run_baseline(env, args.ep, st)
        results[st] = (rate, ho, beam, uniq)
        print(f"  rate={rate:.0f}  HO={ho:.3f}  beam_load={beam:.3f}  uniq_sat={uniq:.0f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 70)
    print(f"{'策略':<20s} {'Rate':>8s} {'HO率':>8s} {'Beam':>8s} {'Sat#':>8s}")
    print("-" * 70)
    for st, label in strategies:
        rate, ho, beam, uniq = results[st]
        tag = f"{label.split(' ')[0]}"
        print(f"{label:<20s} {rate:>8.0f} {ho:>8.3f} {beam:>8.3f} {uniq:>8.0f}")
    print("=" * 70)

    mgcs_rate = results["MGCS"][0]
    best_rate = max(r[0] for r in results.values())
    best_name = [l for s, l in strategies if results[s][0] == best_rate][0]
    print(f"\nMGCS={mgcs_rate:.0f}, 最优传统方法={best_name}({best_rate:.0f})")
    print(f"RL 需超过 {max(mgcs_rate, best_rate):.0f} 才能体现价值")
