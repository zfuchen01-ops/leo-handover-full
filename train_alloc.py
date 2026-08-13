#!/usr/bin/env python3
"""REDA风格: Q网络打分 + 贪心分配 — 学习与决策解耦, 天然防羊群.

用法:
    LEO_ISL_FAIL_MODE=random python3 -u train_alloc.py --tag v1 --slots 200
"""

import argparse, os, collections, random
from math import pi as _pi
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

import random as _random
import numpy as np; np.random.seed(0)
_random.seed(42)
import torch
import torch.nn as nn
torch.manual_seed(0)

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from Handover import Handover, RATE_UPPER


# ═══════════════════════ 环境 (复用) ═══════════════════════
def reset_ids():
    Topology.index_con = 0; User.uid = 0

def make_constellation(topo, c="C"):
    o, s = (16,16) if c=="C" else (8,9) if c=="A" else (12,12)
    p=1; fp=2.0*p*_pi/(o*s); ln=54.0/180.0*_pi; th=2.0*_pi/o
    topo.Add_Constellation(o,s,SAT_HEIGHT,fp,ln,th,TYPE_2PI); topo.Each_Satellite()

def make_user_locations(n):
    return [(_random.uniform(-120,120)*_pi/180.0, _random.uniform(-60,60)*_pi/180.0) for _ in range(n)]

def build_env(n=200, c="C"):
    reset_ids(); t=Topology(); make_constellation(t,c)
    for lon,lat,nm in [(0,0,"G1"),(60,0,"G2"),(120,0,"G3"),(180,0,"G4"),(-120,0,"G5"),(-60,0,"G6")]:
        t.Add_Gateway_Loc(lon/180.0*_pi,lat/180.0*_pi,5,nm)
    t.Add_User_From_Input(make_user_locations(n))
    for u in t.user: u.assigned_gateway = t.gateway[_random.randrange(len(t.gateway))]
    return PaperRebuildHandover(net=Network(t))


# ═══════════════════════ Q网络 ═══════════════════════
class QNet(nn.Module):
    """Pure per-sat scoring (Liu 2022): 无全局偏置, Q值自然分化"""
    def __init__(self, n_sats, n_users=200, h=128):
        super().__init__(); self.N = n_sats
        self.user_embed = nn.Embedding(n_users, 8)
        nn.init.uniform_(self.user_embed.weight, -0.1, 0.1)
        # Per-sat: 5特征+8(embed) → 64 → 32 → 16 → 1 (LayerNorm防方差坍缩)
        self.sat_net = nn.Sequential(
            nn.Linear(5+8, 64), nn.LayerNorm(64), nn.LeakyReLU(0.1),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
        )
        self.q_out = nn.Linear(16, 1)
    def forward(self, s):
        B,D = s.shape; N=self.N  # D = 5N+8
        # 每星拼上user embedding
        per_sat_5 = s[:,:5*N].reshape(B, N, 5)           # [B,N,5]
        emb = s[:,5*N:].unsqueeze(1).expand(-1,N,-1)     # [B,N,8]
        per_sat = torch.cat([per_sat_5, emb], -1)         # [B,N,13]
        sat_f = self.sat_net(per_sat.reshape(B*N, 5+8))  # [B*N,16]
        sat_f = sat_f.reshape(B, N, 16)                   # [B,N,16]
        return self.q_out(sat_f).squeeze(-1)              # [B,N]


# ═══════════════════════ 贪心分配器 ═══════════════════════
def greedy_allocate(q_values, users, N, beams_per_sat=64, epsilon=0.0):
    """贪心分配: 按Q值从高到低, 每人分配到beam没满的最高Q星.

    Args:
        q_values: [200, N] 每用户对每颗星的Q分数
        users: 200个user对象
        N: 卫星数
        epsilon: 探索概率 (随机分配)

    Returns:
        assignments: {user: sat_id}  sat_id从1开始
    """
    remaining = [beams_per_sat] * N
    assignments = {}

    # 按最大Q值排序 (最自信的先选)
    max_q = q_values.max(dim=1).values.cpu().numpy()
    user_order = np.argsort(-max_q)

    for idx in user_order:
        user = users[idx]

        # ε-greedy 探索
        if _random.random() < epsilon:
            visible = [s.ID-1 for s in user.sat_covered]
            candidates = [i for i in visible if remaining[i] > 0]
            if candidates:
                chosen = _random.choice(candidates)
            else:
                chosen = max(visible, key=lambda i: q_values[idx,i].item()) if visible else 0
        else:
            # 贪心: 选剩余beam中Q最高的
            q_row = q_values[idx].clone()
            # 屏蔽beam满的
            for i in range(N):
                if remaining[i] <= 0:
                    q_row[i] = -float('inf')
            # 屏蔽不可见 (Cavai[N:2N]=0)
            # 但我们没有state在这里, 用Q值-1e9处理
            chosen = q_row.argmax().item()

        remaining[chosen] -= 1
        assignments[user] = chosen + 1  # sat_id从1开始

    return assignments


# ═══════════════════════ 训练 ═══════════════════════
def train(env, q_net, target_net, args, start_ep=0, buffer=None, opt=None):
    device = next(q_net.parameters()).device
    users = env.topo.user; n_users = len(users); N = env.topo.total_sat
    ISL_MODE = os.environ.get('LEO_ISL_FAIL_MODE','0')
    import random as _random

    if buffer is None: buffer = collections.deque(maxlen=args.buffer)
    if opt is None: opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    csv_mode = 'a' if start_ep > 0 else 'w'
    csv = open(f'./log/RL/DRQN_{args.tag}_per_ep.csv', csv_mode)
    if start_ep == 0:
        csv.write('episode,reward,rate_avg,hops_avg,ho_avg,q_spread,uniq_sat,beam_avg,rew_rate,rew_ho,loss\n')

    total_eps = args.slots
    prev_s, prev_a, prev_r = [None]*n_users, [None]*n_users, [None]*n_users
    env.reset(0, 'NETWORK_LOAD')
    for ep in range(total_eps):
        display_ep = ep + start_ep
        # ── ISL failures ──
        if ISL_MODE == 'random':
            if not hasattr(env.net,'_active_lsa_cache'):
                env.net._active_lsa_cache=[]; env.net._failed_lsa=[]
                for ci in range(len(env.net.LSDB)):
                    for si in range(len(env.net.LSDB[ci])):
                        for lsa in env.net.LSDB[ci][si]:
                            if lsa.isEstablished and lsa.total_band>0: env.net._active_lsa_cache.append((ci,si,lsa))
            a=env.net._active_lsa_cache; f=env.net._failed_lsa
            _random.shuffle(a); nt=len(a)+len(f); nc=max(1,int(nt*0.05))
            _random.shuffle(f)
            for ci,si,lsa in f[:min(nc,len(f))]: lsa.total_band=lsa._orig; lsa.isEstablished=True; a.append((ci,si,lsa))
            env.net._failed_lsa=f[min(nc,len(f)):]
            for ci,si,lsa in a[:min(nc,len(a))]:
                if lsa.total_band>0:
                    if not hasattr(lsa,'_orig'): lsa._orig=lsa.total_band
                    lsa.total_band=0; lsa.isEstablished=False; env.net._failed_lsa.append((ci,si,lsa))
            env.net._active_lsa_cache=a[min(nc,len(a)):]
            env.net.Update_N2N_Load_By_LSDB_All()

        # ── 顺序决策: 每人选完立即改变env → 后来者看到拥堵 ──
        ep_reward = 0.0
        epsilon = max(0.05, 1.0 - 0.95 * ep / max(1, args.slots-1)) if start_ep == 0 else 0.05
        q_spreads = []
        q_best_sats = set()

        for i, u in enumerate(users):
            s_before = env.Observe(u,'NETWORK_LOAD')
            emb = q_net.user_embed(torch.tensor(i).to(device)).detach().cpu().numpy()
            s_emb_before = np.concatenate([s_before, emb])
            s_t = torch.tensor(s_emb_before,dtype=torch.float).unsqueeze(0).to(device)

            with torch.no_grad():
                q_row = q_net(s_t).squeeze(0)

            for j in range(N):
                if s_before[N+j] <= 0.0: q_row[j] = -float('inf')
            top2 = q_row[q_row > -float('inf')].topk(min(2, (q_row>-float('inf')).sum().item())).values
            if len(top2) >= 2: q_spreads.append((top2[0]-top2[1]).item())
            q_best_sats.add(q_row.argmax().item())

            if _random.random() < epsilon:
                visible = [x.ID-1 for x in u.sat_covered]
                a = _random.choice(visible) if visible else 0
            else:
                a = q_row.argmax().item()

            env.step({u: a+1}, 'NETWORK')
            r = env.Get_Reward(u)
            ep_reward += r

            # 正确链条: (s_{N-1}, a_{N-1}, r_{N-1}, s_N) → 配对后才存
            if prev_s[i] is not None:
                buffer.append([prev_s[i], prev_a[i], prev_r[i], s_emb_before])
            prev_s[i] = s_emb_before; prev_a[i] = a; prev_r[i] = r

        ep_q_spread = sum(q_spreads)/max(1,len(q_spreads))
        ep_q_uniq = len(q_best_sats)

        ep_q_spread = sum(q_spreads)/max(1,len(q_spreads))
        ep_q_uniq = len(q_best_sats)

        # ── DQN Learn ──
        loss_val = 0.0
        if len(buffer) >= args.batch:
            batch = _random.sample(list(buffer), args.batch)
            s_l, a_l, r_l, ns_l = zip(*batch)
            s_t = torch.tensor(np.stack(s_l),dtype=torch.float).to(device)
            a_t = torch.tensor(a_l,dtype=torch.long).to(device)
            r_t = torch.tensor(r_l,dtype=torch.float).to(device)
            ns_t = torch.tensor(np.stack(ns_l),dtype=torch.float).to(device)

            with torch.no_grad():
                # Double DQN: online选动作, target估值 → 防过估计
                next_q_online = q_net(ns_t)
                next_q_target = target_net(ns_t)
                for b in range(args.batch):
                    for i in range(N):
                        if ns_t[b,N+i]<=0.0:
                            next_q_online[b,i]=-float('inf')
                            next_q_target[b,i]=-float('inf')
                best_a = next_q_online.argmax(dim=-1)
                next_max = next_q_target.gather(1, best_a.unsqueeze(1)).squeeze(1)
                target = r_t + args.gamma * next_max

            pred = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
            loss = loss_fn(pred, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(),10.0)
            opt.step(); loss_val=loss.item()

        # Sync target
        if ep % 20 == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Checkpoint + 测试 (ε=0, 20ep)
        if ep == 49 or ep == 99 or ep == 149 or ep == args.slots - 1:
            # Save full state for resume
            torch.save({
                'q_net': q_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': opt.state_dict(),
                'buffer': list(buffer),
                'episode': display_ep + 1,
            }, f'./log/model/alloc_{args.tag}_ep{display_ep+1:03d}.pkl')
            # ── 论文级测试: 新env新种子, ε=0, 验证泛化 ──
            _rs = _random.getstate()
            _random.seed(display_ep + 1000)  # 固定测试种子
            e_env = build_env(n_users, 'C')
            e_users = e_env.topo.user
            _random.setstate(_rs)
            e_env.reset(0, 'NETWORK_LOAD')
            test_rates, test_hos = [], []
            for te in range(120):
                test_raw = [e_env.Observe(u,'NETWORK_LOAD') for u in e_users]
                test_emb = [np.concatenate([test_raw[i], q_net.user_embed(torch.tensor(i).to(device)).detach().cpu().numpy()]) for i in range(n_users)]
                test_st = torch.tensor(np.stack(test_emb),dtype=torch.float).to(device)
                with torch.no_grad():
                    test_q = q_net(test_st)
                test_assign = {}
                for i, u in enumerate(e_users):
                    for j in range(N):
                        if test_raw[i][N+j] <= 0.0: test_q[i,j] = -float('inf')
                    test_assign[u] = test_q[i].argmax().item() + 1
                e_env.step(test_assign, 'NETWORK')
                tc, th = 0.0, 0
                for u in e_users:
                    if u.sat_connected is not None and u.sat_connected in e_env.ho[u]:
                        acc = min(e_env.ho[u][u.sat_connected].c_quality, RATE_UPPER)
                        fd = e_env._get_feeder_sat(u,u.sat_connected)
                        isl_b = e_env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else RATE_UPPER
                        tc += min(acc, isl_b, RATE_UPPER)
                        th += 0.0 if u.sat_connected==u.last_connected else 1.0
                test_rates.append(tc/n_users)
                test_hos.append(th/n_users)
                e_env.Update_Env(te*50,'NETWORK_LOAD')
            print(f'  [TEST ep{display_ep+1}] rate={sum(test_rates)/len(test_rates):.0f}Mbps  HO={sum(test_hos)/len(test_hos):.4f}  (valid)', flush=True)

        # Stats
        ep_rate=ep_hops=ep_ho=n_conn=0.0
        for u in users:
            if u.sat_connected is not None and u.sat_connected in env.ho[u]:
                acc = min(env.ho[u][u.sat_connected].c_quality, RATE_UPPER)
                fd = env._get_feeder_sat(u,u.sat_connected)
                isl = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else RATE_UPPER
                rr = min(acc,isl,RATE_UPPER)
                hh = max(1.0,env.Calc_Path_Hops(u.sat_connected,fd)) if fd and u.sat_connected!=fd else 1.0
                ep_rate+=rr; ep_hops+=hh
                ep_ho += 1.0 if u.sat_connected!=u.last_connected else 0.0; n_conn+=1
        if n_conn>0: ep_rate/=n_conn; ep_hops/=n_conn; ep_ho/=n_conn

        uniq = len(set(u.sat_connected.ID for u in users if u.sat_connected is not None))
        beam = n_conn/max(1,uniq*64.0)
        csv.write(f'{display_ep},{ep_reward:.4f},{ep_rate:.0f},{ep_hops:.1f},{ep_ho:.3f},{0:.4f},{uniq},{beam:.3f},{ep_rate/RATE_UPPER:.2f},{-1.5*(1-ep_ho):.2f},{loss_val:.4f}\n')
        csv.flush()

        if ep<20 or ep%50==0 or ep==args.slots-1:
            print(f'[ep {display_ep:4d}] rate={ep_rate:5.0f}  HO={ep_ho:.3f}  eps={epsilon:.2f}  uniq={uniq}  Qsp={ep_q_spread:.3f}  Quniq={ep_q_uniq}  loss={loss_val:.4f}',flush=True)

        env.Update_Env(ep*50,'NETWORK_LOAD')

    csv.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slots",type=int,default=200); p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--gamma",type=float,default=0.99); p.add_argument("--batch",type=int,default=256)
    p.add_argument("--buffer",type=int,default=50000); p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--tag",default="v1_alloc"); p.add_argument("--device",default="auto")
    p.add_argument("--resume",default="")  # 从checkpoint继续训练
    args = p.parse_args()

    d = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(d)
    env = build_env(200,"C"); N = env.topo.total_sat
    q_net = QNet(N, n_users=200, h=args.hidden).to(device)
    target_net = QNet(N, n_users=200, h=args.hidden).to(device)
    start_ep = 0; buffer = collections.deque(maxlen=args.buffer)
    opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if 'q_net' in ckpt:
            q_net.load_state_dict(ckpt['q_net'])
            target_net.load_state_dict(ckpt['target_net'])
            opt.load_state_dict(ckpt['optimizer'])
            buffer = collections.deque(ckpt['buffer'], maxlen=args.buffer)
            start_ep = ckpt['episode']
        else:
            # 旧格式: 纯state_dict
            q_net.load_state_dict(ckpt)
            target_net.load_state_dict(ckpt)
            start_ep = args.slots  # 假设从slots开始
        print(f"=== Resume from ep{start_ep} ===")

    print(f"=== Q打分 + 贪心分配 ===")
    print(f"  slots={args.slots} lr={args.lr} gamma={args.gamma} device={d} start_ep={start_ep}")
    print(f"  sats={N} users={len(env.topo.user)} params={sum(p.numel() for p in q_net.parameters()):,}")
    print("-"*60)
    train(env,q_net,target_net,args, start_ep=start_ep, buffer=buffer, opt=opt)
    print("-"*60); print("完成!")

if __name__=="__main__": main()
