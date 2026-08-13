#!/usr/bin/env python3
"""DRQN 学习诊断 —— 加载已保存模型, 用5个指标判断是否在学习.

用法:
    python diagnose_drqn.py                          # 诊断所有已保存的checkpoint
    python diagnose_drqn.py --model ep50             # 只诊断 ep50
    python diagnose_drqn.py --compare                # 对比: MGCS + MaxISL + Random 基线
"""
import os, sys, argparse, copy, collections
import numpy as np; np.random.seed(0)
import random; random.seed(42)
import torch; torch.manual_seed(0)

os.environ['LEO_CAHS_VARIANT'] = 'paper'
os.environ['LEO_QUIET_LOGS'] = '1'
os.environ['LEO_C_BAND'] = '100'
sys.path.insert(0, '.')

from train_drqn import build_drqn_env
from DRQNAgent import Q_net, resolve_device

H, SEQ = 128, 6
device = resolve_device('cpu')


def diagnose_model(model_path, label, users=200):
    """加载模型, 跑100ep测试 + 诊断指标"""
    env = build_drqn_env(users, 'C')
    env.reset(0, 'NETWORK_LOAD')
    users_t = list(env.topo.user)
    N = env.topo.total_sat

    qnet = Q_net(4*N+1, N, H).to(device)
    qnet.load_state_dict(torch.load(model_path, map_location=device))
    qnet.eval()

    # Test agent (ε=0 greedy)
    class TA:
        def __init__(self, u, q):
            self.user = u; self.qnet = q
            self.state_fifo = collections.deque(maxlen=SEQ)
            self.lstm_h = None; self.lstm_c = None
        def decide(self, obs):
            if self.lstm_h is None:
                self.lstm_h = torch.zeros(1, 1, H).to(device)
                self.lstm_c = torch.zeros(1, 1, H).to(device)
            self.state_fifo.append(obs)
            sl = list(self.state_fifo) if len(self.state_fifo) >= SEQ \
                else [obs]*(SEQ-len(self.state_fifo)) + list(self.state_fifo)
            qv, self.lstm_h, self.lstm_c = self.qnet(
                torch.as_tensor(sl, dtype=torch.float).to(device).unsqueeze(0),
                self.lstm_h, self.lstm_c)
            qv = qv[:, -1, :].squeeze(0)
            for i in range(N):
                if obs[N+i] <= 0.0:
                    qv[i] = -float('inf')
            return qv.argmax().item(), qv.detach().clone()

    agents = [TA(u, qnet) for u in users_t]

    # Warmup 10 eps
    for ep in range(10):
        for a in agents:
            obs = env.Observe(a.user, 'NETWORK_LOAD')
            act, _ = a.decide(obs)
            env.step({a.user: act+1}, 'NETWORK')
        env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

    # Test 100 eps with diagnostics
    rates = []
    # 诊断指标累积器
    q_stds = []       # Q值标准差 (区分度)
    isl_ranks = []    # ISL排名 (归一化)
    conf_gaps = []    # maxQ - 2ndMaxQ
    top3_hits = []    # Q-top3包含ISL最优卫星
    td_errors = []    # 原始TD误差

    for ep in range(100):
        ep_rate, n = 0.0, 0
        for a in agents:
            obs = env.Observe(a.user, 'NETWORK_LOAD')
            act, qv = a.decide(obs)
            env.step({a.user: act+1}, 'NETWORK')

            # ---- 诊断指标收集 ----
            obs_arr = np.array(obs)  # convert list to array for vectorized ops
            visible_mask = obs_arr[N:2*N] > 0.0
            qv_vis = qv[torch.from_numpy(visible_mask).to(device)]
            if len(qv_vis) >= 2:
                # 1) Q-std: 可见卫星Q值区分度
                q_stds.append(qv_vis.std().item())

                # 2) ISL-rank: 选择的卫星在free_band中的排名
                free_bands = obs_arr[2*N:3*N]  # free_band at position [2N:3N]
                fb_vis = []
                for i in range(N):
                    if visible_mask[i]:
                        fb_vis.append((i, free_bands[i]))
                fb_vis.sort(key=lambda x: -x[1])  # 按free_band降序
                rank_map = {sat_id: rank+1 for rank, (sat_id, _) in enumerate(fb_vis)}
                chosen_rank = rank_map.get(act, len(fb_vis))
                # 归一化: 1/visible_count = best, 1.0 = worst
                isl_ranks.append((chosen_rank - 1) / max(1, len(fb_vis) - 1) if len(fb_vis) > 1 else 0.0)

                # 3) Conf-gap: maxQ - 2ndMaxQ
                qv_sorted = qv_vis.sort(descending=True).values
                conf_gaps.append((qv_sorted[0] - qv_sorted[1]).item() if len(qv_sorted) >= 2 else 0.0)

                # 4) Top3-hit: Q-top3是否包含ISL最优卫星
                top3_q = qv_vis.sort(descending=True).indices[:3].tolist()
                # 映射回全局sat ID
                vis_indices = [i for i in range(N) if visible_mask[i]]
                top3_global = [vis_indices[i] for i in top3_q if i < len(vis_indices)]
                best_isl_sat = fb_vis[0][0]  # free_band最大的卫星
                top3_hits.append(1.0 if best_isl_sat in top3_global else 0.0)

            # ---- Rate + TD-err 计算 ----
            b_val = 9999.0
            if a.user.sat_connected and a.user.sat_connected in env.ho[a.user]:
                fd = env._get_feeder_sat(a.user, a.user.sat_connected)
                b_val = (env.net.N2N_status[a.user.sat_connected.con_id-1][a.user.sat_connected.ID-1][fd.ID-1].free_band
                     if fd and a.user.sat_connected != fd else 9999)
                ep_rate += min(env.ho[a.user][a.user.sat_connected].c_quality, b_val)
                n += 1

            # 5) TD-err: |Q(s,a) - rate_norm|
            with torch.no_grad():
                r_val = min(env.ho[a.user][a.user.sat_connected].c_quality, b_val) / 1000.0 \
                    if a.user.sat_connected and a.user.sat_connected in env.ho[a.user] else 0.0
                td_errors.append(abs(qv[act].item() - r_val) if not torch.isinf(qv[act]) else 0.0)

        rates.append(ep_rate/n if n else 0)
        env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

    avg = sum(rates) / 100
    l50 = sum(rates[-50:]) / 50

    # 汇总
    print(f"\n{'='*60}")
    print(f"  [{label}]  Rate: avg={avg:.0f}  last50={l50:.0f}")
    print(f"{'='*60}")
    print(f"  Q-std:      {np.mean(q_stds):.3f}  (↑ 区分度, 随机≈0.1, 学会>0.5)")
    print(f"  ISL-rank:   {np.mean(isl_ranks):.3f}  (↓ ISL感知, MaxISL=0.0, 随机≈0.5, MGCS≈0.3)")
    print(f"  Conf-gap:   {np.mean(conf_gaps):.4f}  (↑ 决策信心)")
    print(f"  Top3-hit:   {np.mean(top3_hits)*100:.1f}%  (↑ Q与实际ISL一致性)")
    print(f"  TD-err:     median={np.median(td_errors):.4f}  mean={np.mean(td_errors):.4f}  (↓ 预测误差)")
    print(f"  Q值范围:    min={np.min([qv.min().item() for _ in range(1)]):.2f}  "
          f"(clamped to [-10,10])")
    return {
        'label': label, 'avg': avg, 'last50': l50,
        'q_std': np.mean(q_stds), 'isl_rank': np.mean(isl_ranks),
        'conf_gap': np.mean(conf_gaps), 'top3_hit': np.mean(top3_hits),
        'td_err_median': np.median(td_errors), 'td_err_mean': np.mean(td_errors),
    }


def compare_baselines(users=200):
    """计算 MGCS, MaxISL, Random 三条基线 + 诊断指标"""
    env = build_drqn_env(users, 'C')
    N = env.topo.total_sat
    users_t = list(env.topo.user)

    strategies = {
        'MaxISL': lambda arr, u: max(
            [(s.ID-1, arr[2*N + s.ID-1]) for s in u.sat_covered],
            key=lambda x: x[1])[0] if u.sat_covered else 0,
        'MGCS': lambda arr, u: max(
            [(s.ID-1, arr[N + s.ID-1]) for s in u.sat_covered],
            key=lambda x: x[1])[0] if u.sat_covered else 0,
        'Random': lambda arr, u: random.choice([s.ID-1 for s in u.sat_covered]) if u.sat_covered else 0,
    }

    for name, strategy in strategies.items():
        env.reset(0, 'NETWORK_LOAD')
        # Warmup
        for ep in range(10):
            for u in users_t:
                obs_list = env.Observe(u, 'NETWORK_LOAD')
                obs_arr = np.array(obs_list)
                act = strategy(obs_arr, u)
                env.step({u: act+1}, 'NETWORK')
            env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

        rates = []
        isl_ranks = []
        for ep in range(100):
            ep_rate, n = 0.0, 0
            for u in users_t:
                obs_list = env.Observe(u, 'NETWORK_LOAD')
                obs_arr = np.array(obs_list)
                visible_mask = obs_arr[N:2*N] > 0.0
                fb = obs_arr[2*N:3*N]
                # ISL rank
                fb_vis = [(i, fb[i]) for i in range(N) if visible_mask[i]]
                fb_vis.sort(key=lambda x: -x[1])
                rank_map = {sid: r+1 for r, (sid, _) in enumerate(fb_vis)}
                act = strategy(obs_arr, u)
                chosen_rank = rank_map.get(act, len(fb_vis))
                isl_ranks.append((chosen_rank-1)/max(1, len(fb_vis)-1) if len(fb_vis)>1 else 0.0)

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
        print(f"  {name:8s}: rate={avg:.0f}  ISL-rank={np.mean(isl_ranks):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all', help='checkpoint to diagnose (ep50, ep100, all)')
    parser.add_argument('--compare', action='store_true', help='also run baseline comparison')
    args = parser.parse_args()

    print("DRQN 学习诊断")
    print(f"  Model: H={H} seq={SEQ}  ISL=2000 no-cap 200u")
    print()

    if args.compare:
        print("═══ 基线对比 ═══")
        compare_baselines()

    print("\n═══ 模型诊断 ═══")

    checkpoints = {
        'ep50':  './log/model/drqn_200u_ep50.pkl',
        'ep100': './log/model/drqn_200u_ep100.pkl',
        'ep150': './log/model/drqn_200u_ep150.pkl',
    }

    if args.model != 'all':
        checkpoints = {args.model: checkpoints[args.model]}

    results = []
    for label, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"  [{label}] SKIP: model not found at {path}")
            continue
        r = diagnose_model(path, label)
        results.append(r)

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  趋势汇总")
        print(f"  {'Checkpoint':<10} {'Rate':>6} {'Qstd':>6} {'ISLrk':>6} {'ConfGap':>7} {'Top3Hit':>7} {'TDerr':>6}")
        print(f"  {'-'*50}")
        for r in results:
            print(f"  {r['label']:<10} {r['avg']:6.0f} {r['q_std']:6.3f} {r['isl_rank']:6.3f} "
                  f"{r['conf_gap']:7.4f} {r['top3_hit']:6.1%} {r['td_err_median']:6.4f}")
        print()
        print("  判断学习:")
        if results[-1]['q_std'] > results[0]['q_std'] + 0.1:
            print(f"    ✓ Q-std上升 ({results[0]['q_std']:.3f}→{results[-1]['q_std']:.3f}) — Q值在学会区分卫星")
        else:
            print(f"    ✗ Q-std未上升 ({results[1]['q_std']:.3f}) — Q值未形成区分度")
        if results[-1]['isl_rank'] < results[0]['isl_rank'] - 0.05:
            print(f"    ✓ ISL-rank下降 ({results[0]['isl_rank']:.3f}→{results[-1]['isl_rank']:.3f}) — 在学ISL拥塞避免")
        else:
            print(f"    ✗ ISL-rank未下降 ({results[-1]['isl_rank']:.3f}) — 未学会ISL感知")
        if results[-1]['top3_hit'] > results[0]['top3_hit'] + 0.05:
            print(f"    ✓ Top3-hit上升 ({results[0]['top3_hit']:.1%}→{results[-1]['top3_hit']:.1%}) — Q值与ISL一致")
        else:
            print(f"    ✗ Top3-hit未上升 ({results[-1]['top3_hit']:.1%})")


if __name__ == '__main__':
    main()
