#!/usr/bin/env python3
"""对比论文四种切换策略: DRQN, DQN, MGCS(最大信道容量), 最大时长"""
import os, sys, csv, time
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "500")
import numpy as np
from math import pi as _pi
from train_dqn import build_env

N_EPISODES = 500

def run_strategy(name, decision_fn, episodes=N_EPISODES):
    """运行策略, 返回 (rates, hops_list, ho_list, rewards)"""
    env = build_env(200, 'C')
    all_users = list(env.topo.user)
    actions = {}
    env.reset(0, 'NETWORK_LOAD')

    rates, hops_list, ho_list, rewards = [], [], [], []

    for ep in range(episodes):
        ep_r = 0.0; ep_rate = 0.0; ep_hops = 0.0; ep_ho = 0.0; n = 0

        for user in all_users:
            act = decision_fn(user, env, ep, all_users)
            env.step({user: act+1}, 'INITIAL' if ep==0 else 'NETWORK')
            actions[user] = act+1

        for user in all_users:
            r = env.Get_Reward(user)
            ep_r += r
            if user.sat_connected is not None:
                access = min(env.ho[user][user.sat_connected].c_quality, 500.0)
                feeder = env._get_feeder_sat(user, user.sat_connected)
                isl_b = env.net.N2N_status[user.sat_connected.con_id-1][user.sat_connected.ID-1][feeder.ID-1].free_band if feeder and user.sat_connected!=feeder else 500.0
                ep_rate += min(access, isl_b, 500.0)
                hops = 1.0
                if feeder is not None and user.sat_connected != feeder:
                    hops = max(1.0, env.Calc_Path_Hops(user.sat_connected, feeder))
                ep_hops += hops
                ep_ho += 1.0 if user.sat_connected == user.last_connected else 0.0
                n += 1

        actions.clear()
        env.Update_Env((ep+1)*50, 'NETWORK_LOAD')

        if n > 0:
            rates.append(ep_rate/n)
            hops_list.append(ep_hops/n)
            ho_list.append(ep_ho/n)
            rewards.append(ep_r)

        if ep < 5 or ep % 100 == 0:
            print(f'  [{name}] ep {ep}: rate={rates[-1]:.0f} hops={hops_list[-1]:.1f} ho={ho_list[-1]:.2f} r={ep_r:.1f}', flush=True)

    return rates, hops_list, ho_list, rewards

# ── 决策函数 ──
def mgcs_decision(user, env, ep, all_users):
    """最大信道容量策略"""
    best_sat, best_q = None, -1
    for sat in user.sat_covered:
        q = env.ho[user][sat].c_quality
        if q > best_q: best_q = q; best_sat = sat
    return (best_sat if best_sat else list(user.sat_covered)[0]).ID - 1

def max_duration_decision(user, env, ep, all_users):
    """最大可见时长策略: 不切换, 只在卫星移出可见范围时切到最新进入的卫星"""
    if user.sat_connected is not None and user.sat_connected in user.sat_covered:
        return user.sat_connected.ID - 1  # 保持当前卫星
    # 当前卫星不可见, 选最大信道容量作为fallback
    best_sat, best_q = None, -1
    for sat in user.sat_covered:
        q = env.ho[user][sat].c_quality
        if q > best_q: best_q = q; best_sat = sat
    return (best_sat if best_sat else list(user.sat_covered)[0]).ID - 1

# ── 运行 ──
if __name__ == '__main__':
    print("=" * 60)
    print(f"论文策略对比: 256星, 200用户, {N_EPISODES}集")
    print("=" * 60)

    results = {}

    # 1. MGCS (最大信道容量)
    print("\n[1/4] MGCS 最大信道容量...")
    rates, hops, ho, rewards = run_strategy("MGCS", mgcs_decision)
    results['MGCS'] = (rates, hops, ho, rewards)

    # 2. 最大时长
    print("\n[2/4] 最大可见时长...")
    rates, hops, ho, rewards = run_strategy("MaxDuration", max_duration_decision)
    results['MaxDuration'] = (rates, hops, ho, rewards)

    # 3. DQN - 从已有训练结果读取 (取最后500集)
    print("\n[3/4] DQN (读取训练日志)...")
    dqn_rates, dqn_hops, dqn_ho, dqn_r = [], [], [], []
    dqn_csv = './log/RL/DQN_per_ep.csv'
    if os.path.exists(dqn_csv):
        with open(dqn_csv) as f:
            rows = list(csv.DictReader(f))
            for row in rows[-500:]:
                dqn_rates.append(float(row.get('rate_avg', 0)))
                dqn_hops.append(float(row.get('hops_avg', 0)))
                dqn_ho.append(float(row.get('ho_avg', 0)))
                dqn_r.append(float(row.get('reward', 0)))
        print(f"  读取DQN最后{len(dqn_rates)}集")
    results['DQN'] = (dqn_rates, dqn_hops, dqn_ho, dqn_r)

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("结果汇总 (最后100集平均)")
    print("=" * 60)
    print(f"{'策略':<15} {'速率(Mbps)':>10} {'跳数':>8} {'切换率':>8} {'Reward':>8}")
    print("-" * 55)

    summary = {}
    for name, (rates, hops, ho, rewards) in results.items():
        n = min(100, len(rates))
        if n > 0:
            r_avg = sum(rates[-n:])/n
            h_avg = sum(hops[-n:])/n
            ho_avg = sum(ho[-n:])/n
            rw_avg = sum(rewards[-n:])/n
            summary[name] = (r_avg, h_avg, ho_avg, rw_avg)
            print(f"{name:<15} {r_avg:>10.0f} {h_avg:>8.1f} {ho_avg:>8.2f} {rw_avg:>8.1f}")

    # 保存CSV
    with open('./log/RL/compare_results.csv', 'w') as f:
        f.write('strategy,rate_avg,hops_avg,ho_avg,reward_avg\n')
        for name, (r, h, ho, rw) in summary.items():
            f.write(f'{name},{r:.1f},{h:.2f},{ho:.3f},{rw:.2f}\n')
    print(f"\n结果已保存到 log/RL/compare_results.csv")
