#!/usr/bin/env python3
"""DRQN 训练脚本 —— 早停收敛版.

用法:
    python train_drqn.py                           # 默认 A 星座, 200 终端
    python train_drqn.py --constellation B         # B 星座
    python train_drqn.py --users 200 --slots 5000  # 自定义
"""

import argparse
import os
import sys
from math import pi as _pi
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "0")  # DRQN训练需要日志输出
os.environ.setdefault("LEO_C_BAND", "100")
os.environ.setdefault("LEO_HO_PENALTY", "0.2")
os.environ.setdefault("LEO_ORTH_LAMBDA", "0.1")
os.environ.setdefault("LEO_FEAT_PER_SAT", "4")  # 4特征: elev, RVT, cq, ISL
os.environ.setdefault("LEO_USE_TRANSFORMER", "0")  # 0=DRQN(LSTM), 1=Transformer
os.environ.setdefault("LEO_VARLEN", "0")  # 1=变长输入(全部可见星)

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI


# ── run_paper_rebuild 本地函数 ──
def reset_ids():
    Topology.index_con = 0
    User.uid = 0


def make_constellation(topo: Topology, constellation: str):
    if constellation == "A":
        orbit_num, sat_per_orbit = 8, 9
    elif constellation == "B":
        orbit_num, sat_per_orbit = 12, 12
    elif constellation == "C":
        orbit_num, sat_per_orbit = 16, 16  # 博士论文: 256星
    elif constellation == "D":
        orbit_num, sat_per_orbit = 20, 20  # 400星
    else:
        raise ValueError("constellation must be A, B, C, or D")
    phase = 1
    first_phi = 2.0 * phase * _pi / (orbit_num * sat_per_orbit)
    lean = 54.0 / 180.0 * _pi
    theta = 2.0 * _pi / orbit_num
    topo.Add_Constellation(
        orbit_num, sat_per_orbit, SAT_HEIGHT, first_phi, lean, theta, TYPE_2PI
    )
    topo.Each_Satellite()


# 用户位置生成 (从 run_paper_rebuild 简化)
def make_user_locations(count: int):
    """生成随机用户位置, 返回 [(lon_rad, lat_rad), ...]."""
    import random
    random.seed(42)
    users = []
    for _ in range(count):
        lat = random.uniform(-60, 60) * _pi / 180.0  # 与v1.1/DQN一致
        lon = random.uniform(-120, 120) * _pi / 180.0
        users.append((lon, lat))
    return users


def build_drqn_env(user_count: int, constellation: str):
    """构建DRQN训练环境，200终端 + 6地面站."""
    reset_ids()
    topo = Topology()
    make_constellation(topo, constellation)

    # 6 个地面站 (博士论文配置)
    gw_coords = [
        (0, 0, "GW1"),
        (60, 0, "GW2"),
        (120, 0, "GW3"),
        (180, 0, "GW4"),
        (-120, 0, "GW5"),
        (-60, 0, "GW6"),
    ]
    for lon_deg, lat_deg, name in gw_coords:
        topo.Add_Gateway_Loc(
            lon_deg / 180.0 * _pi,
            lat_deg / 180.0 * _pi,
            antenna_Num=5,  # 论文5天线
            name_str=name,
        )

    # 添加用户
    user_locs = make_user_locations(user_count)
    topo.Add_User_From_Input(user_locs)

    # 每个用户随机分配地面站 (论文未指定具体规则)
    import random as _random
    for user in topo.user:
        user.assigned_gateway = topo.gateway[_random.randrange(len(topo.gateway))]

    # 论文gateway场景: 无配对, 每用户直连gateway
    env = PaperRebuildHandover(net=Network(topo))
    return env


def main():
    parser = argparse.ArgumentParser(description="DRQN 训练 (早停收敛)")
    parser.add_argument("--constellation", default="C", choices=["A", "B", "C", "D"])
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--slots", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5,
                        help="连续N个窗口无改善即早停 (每窗口=400 ep)")
    parser.add_argument("--min-episodes", type=int, default=2000,
                        help="早停前最少训练的episode数")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    import DRQNAgent
    DRQNAgent.per_ep_log = open(f'./log/RL/DRQN_{args.tag}_per_ep.csv', 'w')
    DRQNAgent.per_ep_log.write('episode,reward,rate_avg,hops_avg,ho_avg,q_spread,uniq_sat,beam_avg,rew_rate,rew_ho,loss,q_gap_avg,q_gap_noise,n_valid\n')
    from DRQNAgent import UserAgent, CenterAgent, train_episode

    # 检查 CUDA
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    print(f"=== DRQN 训练配置 ===")
    print(f"  星座: {args.constellation}")
    print(f"  终端数: {args.users}")
    print(f"  总时隙: {args.slots} (end_time={args.slots*50}ms)")
    print(f"  学习率: {args.lr}")
    print(f"  早停patience: {args.patience} 窗口 (={args.patience*400} ep)")
    print(f"  最少episode: {args.min_episodes}")
    print(f"  ε衰减: 1.0 → 0.05 (线性, 0→min_episodes)")
    print(f"  设备: {device_str}")
    print()

    # 构建环境
    print("构建环境中...")
    env = build_drqn_env(args.users, args.constellation)
    total_sats = env.topo.total_sat
    print(f"  卫星数: {total_sats}")
    print(f"  用户数: {len(env.topo.user)}")
    print(f"  地面站: {len(env.topo.gateway)}")
    print()

    # 创建 CenterAgent
    c_agent = CenterAgent(
        env,
        gamma=0.9,
        epsilon=0.01,
        batch=256,
        buffer=20000,
        hidden_size=128,  # 论文Table3-1: LSTM单元=6
        lr=args.lr,
        seq=6,
        device=device_str,
    )

    # 创建 UserAgents
    u_agents = []
    for i, user in enumerate(env.topo.user):
        agent = UserAgent(
            user, env, c_agent,
            gamma=0.9,
            epsilon=0.01,
            batch=256,
            buffer=2000,
            hidden_size=128,  # 论文Table3-1: LSTM单元=6
            seq=6,
            device=device_str,
            head_idx=i,
        )
        u_agents.append(agent)

    # Resume: 加载预训练权重 (训练中断后继续训练)
    if args.resume:
        import torch, copy
        print(f"Loading pretrained: {args.resume}", flush=True)
        c_agent.evaluate_net = torch.load(args.resume, map_location='cpu', weights_only=False)
        c_agent.evaluate_net = c_agent.evaluate_net.to(c_agent.device)
        c_agent.target_net = copy.deepcopy(c_agent.evaluate_net).to(c_agent.device)
        for agent in u_agents:
            agent.evaluate_net = copy.deepcopy(c_agent.evaluate_net).to(agent.device)
            agent.lstm_h = None; agent.lstm_c = None
        print("  loaded ok, synced to all agents", flush=True)

    K = DRQNAgent.TOP_K; F = DRQNAgent.FEAT_PER_SAT
    print(f"CenterAgent: input={c_agent.evaluate_net.input_size} (top{K}×{F}+embed), output={K}, hidden={c_agent.evaluate_net.hidden_size}, seq=6")
    print(f"独立head: {len(u_agents)}个 × (v_head+a_head+ctx_fc+sat_net) | agent_embed: {len(u_agents)}×64={len(u_agents)*64}")
    print(f"UserAgents: {len(u_agents)} 个")
    print(f"总参数量: ~{sum(p.numel() for p in c_agent.evaluate_net.parameters())} (Center)")
    print()

    # 训练
    end_time_ms = args.slots * 50
    print(f"开始训练... ({args.slots} episodes, {end_time_ms}ms)")
    print("-" * 60)

    reward_list = train_episode(
        env, u_agents, c_agent,
        model=f"drqn_{args.constellation}_u{args.users}_{args.tag}",
        mode='train',
        start_time=0,
        end_time=end_time_ms,
        time_step=50,
        net_step=20,
        batch=256,
        patience=args.patience,
        min_episodes=args.min_episodes,
        conv_threshold=0.005,
    )

    print("-" * 60)
    print(f"训练完成! 共 {len(reward_list)} episodes")

    # 检查是否收敛
    # reward_list 中前两个元素是 -1 (标记位), 跳过
    valid_rewards = [r for r in reward_list if r != -1]
    if valid_rewards:
        print(f"最终平均奖励: {sum(valid_rewards[-400:]) / min(400, len(valid_rewards)):.4f}")
        print(f"最高单episode奖励: {max(valid_rewards):.4f}")


if __name__ == "__main__":
    main()
