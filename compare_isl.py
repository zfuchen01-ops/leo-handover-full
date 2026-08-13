#!/usr/bin/env python3
"""MGCS基线 + RL(无ISL) + RL(有ISL) 三组对比. 子进程调用, 避免import缓存."""
import os, sys, subprocess, time
from math import pi as _pi
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

import random as _random
import numpy as np; np.random.seed(0)
_random.seed(42)

os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from Handover import Handover, RATE_UPPER

def reset_ids():
    Topology.index_con = 0; User.uid = 0

def build_env(n=200, constellation="C"):
    reset_ids(); t = Topology()
    if constellation == "C": o, s = 16, 16
    p = 1; fp = 2.0 * p * _pi / (o * s)
    ln = 54.0 / 180.0 * _pi; th = 2.0 * _pi / o
    t.Add_Constellation(o, s, SAT_HEIGHT, fp, ln, th, TYPE_2PI); t.Each_Satellite()
    for lon, lat, nm in [(0,0,"G1"),(60,0,"G2"),(120,0,"G3"),(180,0,"G4"),(-120,0,"G5"),(-60,0,"G6")]:
        t.Add_Gateway_Loc(lon/180.0*_pi, lat/180.0*_pi, 5, nm)
    users = [(_random.uniform(-120,120)*_pi/180.0, _random.uniform(-60,60)*_pi/180.0) for _ in range(n)]
    t.Add_User_From_Input(users)
    for u in t.user:
        u.assigned_gateway = t.gateway[_random.randrange(len(t.gateway))]
    return PaperRebuildHandover(net=Network(t))

def run_mgcs(env, n_ep=10):
    """MGCS: 所有用户选最高c_quality的星."""
    rates, hos = [], []
    for ep in range(n_ep):
        env.reset(ep * 50, 'NETWORK_LOAD')
        ep_rate, ep_ho, n_conn = 0.0, 0.0, 0
        for u in env.topo.user:
            best_sat, best_cq = None, -1
            for sat in env.ho[u]:
                cq = env.ho[u][sat].c_quality
                if cq > best_cq: best_cq = cq; best_sat = sat
            if best_sat is None: continue
            env.step({u: best_sat.ID}, 'INITIAL' if ep == 0 else 'NETWORK')
            if u.sat_connected is not None and u.sat_connected in env.ho[u]:
                fd = env._get_feeder_sat(u, u.sat_connected)
                fb_val = 9999.0
                if fd is not None and u.sat_connected != fd:
                    fb_val = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band
                val = min(env.ho[u][u.sat_connected].c_quality, fb_val)
                ep_rate += val; n_conn += 1
                if u.last_connected is not None and u.sat_connected != u.last_connected:
                    ep_ho += 1
        if n_conn > 0: rates.append(ep_rate / n_conn); hos.append(ep_ho)
    return sum(rates) / len(rates), sum(hos) / len(hos) / len(env.topo.user)

def run_rl_subprocess(n_users, n_ep, tag, feat_per_sat):
    """子进程跑train_drqn.py."""
    env_vars = {
        'LEO_ORTH_LAMBDA': '0.1',
        'LEO_HO_PENALTY': '0',
        'LEO_FEAT_PER_SAT': str(feat_per_sat),
        'LEO_CAHS_VARIANT': 'paper',
        'LEO_QUIET_LOGS': '0',
        'LEO_C_BAND': '100',
        'PYTHONUNBUFFERED': '1',
    }
    cmd = [
        sys.executable, '-u', str(BASE_DIR / 'train_drqn.py'),
        '--slots', str(n_ep), '--users', str(n_users),
        '--tag', tag, '--min-episodes', '0', '--patience', '999',
    ]
    print(f"  启动: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE_DIR),
                           env={**os.environ, **env_vars},
                           capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"  完成: {elapsed:.0f}s", flush=True)

    # 读CSV
    import csv
    csv_path = BASE_DIR / f'log/RL/DRQN_{tag}_per_ep.csv'
    qs, rates = [], []
    if csv_path.exists():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                try:
                    qs.append(float(r['q_spread']))
                    rates.append(float(r['rate_avg']))
                except: pass
    q_spread = max(qs) if qs else 0
    rate = max(rates) if rates else 0
    return q_spread, rate, result.stdout, result.stderr

if __name__ == '__main__':
    N_USERS = 20
    N_EP = 80

    # 1. MGCS (用同一环境)
    print(f"构建环境 (N={N_USERS}用户)...", flush=True)
    t0 = time.time()
    env = build_env(N_USERS, "C")
    print(f"  完成: {time.time()-t0:.0f}s, 卫星={env.topo.total_sat}", flush=True)

    print("\n=== 1/3 MGCS 基线 ===", flush=True)
    t0 = time.time()
    mgcs_rate, mgcs_ho = run_mgcs(env, 10)
    print(f"  rate={mgcs_rate:.0f} Mbps, HO率={mgcs_ho:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # 2. RL 无ISL
    print(f"\n=== 2/3 RL 无ISL (3特征) {N_EP}ep ===", flush=True)
    rl3_qs, rl3_rate, out3, err3 = run_rl_subprocess(N_USERS, N_EP, "cmp_noisl", 3)

    # 3. RL 有ISL
    print(f"\n=== 3/3 RL 有ISL (4特征) {N_EP}ep ===", flush=True)
    rl4_qs, rl4_rate, out4, err4 = run_rl_subprocess(N_USERS, N_EP, "cmp_isl", 4)

    print("\n" + "="*60)
    print("对比汇总")
    print("="*60)
    print(f"{'':20s} {'rate(Mbps)':>10s} {'HO率':>8s} {'q_spread':>10s}")
    print(f"{'MGCS 基线':20s} {mgcs_rate:>10.0f} {mgcs_ho:>8.3f} {'-':>10s}")
    print(f"{'RL 无ISL(3特征)':20s} {rl3_rate:>10.0f} {'?':>8s} {rl3_qs:>10.4f}")
    print(f"{'RL 有ISL(4特征)':20s} {rl4_rate:>10.0f} {'?':>8s} {rl4_qs:>10.4f}")
    if rl4_rate > rl3_rate:
        print(f"\n✓ ISL提升: +{rl4_rate-rl3_rate:.0f} Mbps ({(rl4_rate/rl3_rate-1)*100:.1f}%)")
    else:
        print(f"\n✗ ISL未提升: {rl4_rate-rl3_rate:.0f} Mbps")
    if max(rl3_rate, rl4_rate) > mgcs_rate:
        best = max(rl3_rate, rl4_rate)
        print(f"✓ RL超MGCS: +{best-mgcs_rate:.0f} Mbps ({(best/mgcs_rate-1)*100:.1f}%)")
    else:
        print(f"✗ RL未超MGCS")
    print("="*60)
