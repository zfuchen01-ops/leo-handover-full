#!/usr/bin/env python3
"""PPO 单Agent顺序决策200用户 — 随机策略天然防羊群.

用法:
    LEO_ISL_FAIL_MODE=random python3 -u train_ppo.py --tag v1_ppo --slots 300
"""

import argparse, os, sys
from math import pi as _pi
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "100")

import numpy as np; np.random.seed(0)
import random as _random; _random.seed(42)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Topology import Topology
from User import User
from Network import Network
from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from Handover import Handover, RATE_UPPER


# ══════════════════════════════════════════════════════════════
# 环境构建 (复用)
# ══════════════════════════════════════════════════════════════
def reset_ids():
    Topology.index_con = 0; User.uid = 0

def make_constellation(topo, constellation="C"):
    orbit_num, sat_per_orbit = (16, 16) if constellation == "C" else (8, 9) if constellation == "A" else (12, 12)
    phase = 1; first_phi = 2.0 * phase * _pi / (orbit_num * sat_per_orbit)
    lean = 54.0 / 180.0 * _pi; theta = 2.0 * _pi / orbit_num
    topo.Add_Constellation(orbit_num, sat_per_orbit, SAT_HEIGHT, first_phi, lean, theta, TYPE_2PI)
    topo.Each_Satellite()

def make_user_locations(count):
    users = []
    for _ in range(count):
        lat = _random.uniform(-60, 60) * _pi / 180.0
        lon = _random.uniform(-120, 120) * _pi / 180.0
        users.append((lon, lat))
    return users

def build_env(user_count=200, constellation="C"):
    reset_ids(); topo = Topology(); make_constellation(topo, constellation)
    gw_coords = [(0,0,"GW1"),(60,0,"GW2"),(120,0,"GW3"),(180,0,"GW4"),(-120,0,"GW5"),(-60,0,"GW6")]
    for lon_deg, lat_deg, name in gw_coords:
        topo.Add_Gateway_Loc(lon_deg/180.0*_pi, lat_deg/180.0*_pi, antenna_Num=5, name_str=name)
    topo.Add_User_From_Input(make_user_locations(user_count))
    for user in topo.user:
        user.assigned_gateway = topo.gateway[_random.randrange(len(topo.gateway))]
    return PaperRebuildHandover(net=Network(topo))


# ══════════════════════════════════════════════════════════════
# Actor-Critic 网络
# ══════════════════════════════════════════════════════════════
class ActorCritic(nn.Module):
    """共享backbone + Actor(每星logit) + Critic(标量V)"""
    def __init__(self, n_sats, hidden=128, temperature=0.5):
        super().__init__()
        self.N = n_sats
        self.temperature = temperature
        # Per-sat feature extractor: 5 → 32 → 16
        self.sat_net = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        # Global context: 5N+1(位置) → hidden → 16
        self.ctx_fc = nn.Sequential(
            nn.Linear(5 * n_sats + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 16), nn.ReLU(),
        )
        # Actor: per-sat logit
        self.actor_head = nn.Linear(32, 1)
        # Critic: state value
        self.critic = nn.Sequential(
            nn.Linear(5 * n_sats + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        # 正交初始化 (PPO标准)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)
        # Actor head: small weights for stable start
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        # Critic final layer: small weights
        last_critic = list(self.critic.children())[-1]
        nn.init.orthogonal_(last_critic.weight, gain=1.0)
        nn.init.constant_(last_critic.bias, 0.0)

    def forward(self, state):
        """返回 (action_logits [B,N], state_value [B,1])"""
        B, D = state.shape; N = self.N
        # Per-sat: [B, 5N] → [B, N, 5] → [B*N, 5] → [B*N, 16]
        per_sat = state[:, :5*N].reshape(B, N, 5)
        sat_feat = self.sat_net(per_sat.reshape(B * N, 5)).reshape(B, N, 16)
        # Global context: [B, 16]
        ctx = self.ctx_fc(state).unsqueeze(1).expand(-1, N, -1)
        # Actor logits: [B, N, 32] → [B, N]
        combined = torch.cat([sat_feat, ctx], dim=-1)
        logits = self.actor_head(combined).squeeze(-1)
        # Critic value
        value = self.critic(state)
        return logits, value

    def get_action(self, state, mask_invisible=True, top_k=0):
        """采样动作, 返回 (action, log_prob, value, entropy)"""
        logits, value = self.forward(state)
        N = self.N
        # 矢量化屏蔽不可见卫星: Cavai[N:2N]=0 → 不可见
        if mask_invisible:
            bad_mask = (state[:, N:2*N] <= 0.0)  # [B, N]
            logits = logits.masked_fill(bad_mask, -float('inf'))
        # Top-K: 每行只保留Cavai最高的K颗
        if top_k > 0:
            cavai = state[:, N:2*N]  # [B, N]
            bad_mask = torch.ones_like(logits, dtype=torch.bool)
            _, top_idx = torch.topk(cavai, k=min(top_k, N), dim=-1)  # [B, K]
            bad_mask.scatter_(1, top_idx, False)  # top-K 不mask
            logits = logits.masked_fill(bad_mask, -float('inf'))
        scaled_logits = logits / self.temperature
        dist = Categorical(logits=scaled_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value.squeeze(-1), entropy, scaled_logits


# ══════════════════════════════════════════════════════════════
# PPO 训练
# ══════════════════════════════════════════════════════════════
def compute_gae(rewards, values, gamma, lam):
    """GAE advantage estimation."""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t+1] if t+1 < len(values) else 0) - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def train_ppo(env, ac, args):
    device = next(ac.parameters()).device
    users = env.topo.user; n_users = len(users); N = env.topo.total_sat
    ISL_FAIL_MODE = os.environ.get('LEO_ISL_FAIL_MODE', '0')
    top_k = args.top_k
    optimizer = torch.optim.Adam(ac.parameters(), lr=args.lr)

    csv = open(f'./log/RL/DRQN_{args.tag}_per_ep.csv', 'w')
    csv.write('episode,reward,rate_avg,hops_avg,ho_avg,q_spread,uniq_sat,beam_avg,rew_rate,rew_ho,loss\n')

    episode_rewards = []
    env.reset(0, 'NETWORK_LOAD')

    for ep in range(args.slots):
        # ── ISL failures ──
        if ISL_FAIL_MODE == 'random':
            if not hasattr(env.net, '_active_lsa_cache'):
                env.net._active_lsa_cache = []; env.net._failed_lsa = []
                for ci in range(len(env.net.LSDB)):
                    for si in range(len(env.net.LSDB[ci])):
                        for lsa in env.net.LSDB[ci][si]:
                            if lsa.isEstablished and lsa.total_band > 0:
                                env.net._active_lsa_cache.append((ci, si, lsa))
            active = env.net._active_lsa_cache; failed = env.net._failed_lsa
            _random.shuffle(active)
            n_total = len(active) + len(failed); n_change = max(1, int(n_total * 0.05))
            _random.shuffle(failed)
            for ci, si, lsa in failed[:min(n_change, len(failed))]:
                lsa.total_band = lsa._orig; lsa.isEstablished = True; active.append((ci, si, lsa))
            env.net._failed_lsa = failed[min(n_change, len(failed)):]
            for ci, si, lsa in active[:min(n_change, len(active))]:
                if lsa.total_band > 0:
                    if not hasattr(lsa, '_orig'): lsa._orig = lsa.total_band
                    lsa.total_band = 0; lsa.isEstablished = False; env.net._failed_lsa.append((ci, si, lsa))
            env.net._active_lsa_cache = active[min(n_change, len(active)):]
            env.net.Update_N2N_Load_By_LSDB_All()

        # ── 采集 trajectory: 200用户顺序决策 ──
        states, actions, log_probs, rewards, values, old_logits_list = [], [], [], [], [], []
        ep_reward = 0.0

        for user_idx, user in enumerate(users):
            state = env.Observe(user, 'NETWORK_LOAD')
            state = np.append(state, user_idx / max(1, n_users - 1))
            state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, value, _, logits = ac.get_action(state_t, top_k=top_k)

            a = action.item()
            states.append(state); actions.append(a)
            log_probs.append(log_prob.item()); values.append(value.item())
            old_logits_list.append(logits.squeeze(0).cpu())  # [N]

            env.step({user: a + 1}, 'NETWORK')
            reward = env.Get_Reward(user)
            rewards.append(reward); ep_reward += reward

        # GAE bootstrap
        last_s = env.Observe(users[-1], 'NETWORK_LOAD')
        last_s = np.append(last_s, 1.0)
        with torch.no_grad():
            _, _, fv, _, _ = ac.get_action(
                torch.tensor(last_s, dtype=torch.float).unsqueeze(0).to(device), top_k=top_k
            )
        values.append(fv.item())
        advantages, returns = compute_gae(rewards, values, args.gamma, args.gae_lambda)

        advantages_t = torch.tensor(advantages, dtype=torch.float).to(device)
        returns_t = torch.tensor(returns, dtype=torch.float).to(device)
        old_log_probs_t = torch.tensor(log_probs, dtype=torch.float).to(device)
        old_logits_t = torch.stack([t.to(device) for t in old_logits_list])  # [200, N]
        states_t = torch.tensor(np.stack(states), dtype=torch.float).to(device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ── PPO update (mini-batch + value clipping) ──
        total_loss = 0.0; n_updates = 0
        actions_t = torch.tensor(actions, dtype=torch.long).to(device)
        n_steps = len(states)
        mb_size = args.mini_batch

        # 熵退火: 线性衰减
        ent_coef = args.ent_coef * max(0.05, 1.0 - ep / max(1, args.slots - 1))

        for _ in range(args.ppo_epochs):
            # 随机打乱 mini-batch
            perm = torch.randperm(n_steps)
            for start in range(0, n_steps, mb_size):
                idx = perm[start:start + mb_size]
                s_batch = states_t[idx]
                a_batch = actions_t[idx]
                adv_batch = advantages_t[idx]
                ret_batch = returns_t[idx]
                old_lp_batch = old_log_probs_t[idx]

                logits, values_pred = ac(s_batch)
                N_sat = ac.N

                # 矢量化: Mask invisible + Top-K
                bad_mask = (s_batch[:, N_sat:2*N_sat] <= 0.0)
                logits = logits.masked_fill(bad_mask, -float('inf'))
                if top_k > 0:
                    cavai = s_batch[:, N_sat:2*N_sat].clone()
                    cavai[bad_mask] = -1.0  # 不可见置负,不进topk
                    _, top_idx = torch.topk(cavai, k=min(top_k, N_sat), dim=-1)
                    tk_mask = torch.ones_like(logits, dtype=torch.bool)
                    tk_mask.scatter_(1, top_idx, False)
                    logits = logits.masked_fill(tk_mask, -float('inf'))
                scaled_logits = logits / ac.temperature
                dist = Categorical(logits=scaled_logits)
                new_log_probs = dist.log_prob(a_batch)
                entropy = dist.entropy().mean()

                # KL penalty (KL-PPO): 阻止策略过早锁定次优星
                old_l_batch = old_logits_t[idx] / ac.temperature
                old_dist = Categorical(logits=old_l_batch)
                kl = torch.distributions.kl_divergence(old_dist, dist).mean()

                ratio = torch.exp(new_log_probs - old_lp_batch)
                clip_adv = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_batch
                actor_loss = -torch.min(ratio * adv_batch, clip_adv).mean()

                # Value clipping: 限制critic更新幅度
                old_values = ret_batch - adv_batch  # values = returns - advantages
                values_pred = values_pred.squeeze(-1)
                values_clipped = old_values + torch.clamp(
                    values_pred - old_values, -args.clip_eps, args.clip_eps
                )
                critic_loss1 = F.mse_loss(values_pred, ret_batch)
                critic_loss2 = F.mse_loss(values_clipped, ret_batch)
                critic_loss = torch.max(critic_loss1, critic_loss2)

                loss = actor_loss + args.vf_coef * critic_loss - ent_coef * entropy + args.kl_coef * kl

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
                n_updates += 1

        avg_loss = total_loss / max(1, n_updates)

        # ── Stats ──
        ep_rate = ep_hops = ep_ho = n_conn = 0.0
        for u in users:
            if u.sat_connected is not None and u.sat_connected in env.ho[u]:
                access = min(env.ho[u][u.sat_connected].c_quality, RATE_UPPER)
                fd = env._get_feeder_sat(u, u.sat_connected)
                isl_b = env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else RATE_UPPER
                r = min(access, isl_b, RATE_UPPER)
                h = max(1.0, env.Calc_Path_Hops(u.sat_connected, fd)) if fd and u.sat_connected!=fd else 1.0
                ep_rate += r; ep_hops += h
                ep_ho += 1.0 if u.sat_connected != u.last_connected else 0.0
                n_conn += 1
        if n_conn > 0: ep_rate /= n_conn; ep_hops /= n_conn; ep_ho /= n_conn

        uniq = len(set(u.sat_connected.ID for u in users if u.sat_connected is not None))
        beam = n_conn / max(1, uniq * 64.0)
        csv.write(f'{ep},{ep_reward:.4f},{ep_rate:.0f},{ep_hops:.1f},{ep_ho:.3f},'
                  f'{0.0:.4f},{uniq},{beam:.3f},{ep_rate/RATE_UPPER:.2f},{-1.5*(1-ep_ho):.2f},{avg_loss:.4f}\n')
        csv.flush()

        if ep < 20 or ep % 50 == 0 or ep == args.slots - 1:
            print(f'[ep {ep:4d}] rate={ep_rate:5.0f}  HO={ep_ho:.3f}  reward={ep_reward:7.1f}  loss={avg_loss:.4f}  uniq={uniq}',
                  flush=True)

        episode_rewards.append(ep_reward)
        env.Update_Env(ep * 50, 'NETWORK_LOAD')

    csv.close()
    return episode_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0003)  # PPO typically uses lower LR
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.02)  # 初始熵系数(会退火)
    parser.add_argument("--kl_coef", type=float, default=0.01)   # KL散度系数
    parser.add_argument("--mini_batch", type=int, default=64)    # mini-batch size
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=0)   # 0=不用top-K, PPO自己从可见星选
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--tag", default="v11_klppo")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"=== PPO + top-K ===")
    print(f"  slots={args.slots}  lr={args.lr}  gamma={args.gamma}  clip={args.clip_eps}")
    print(f"  ppo_epochs={args.ppo_epochs}  mini_batch={args.mini_batch}  temp={args.temperature}  top_k={args.top_k}")
    print(f"  +正交初始化 +value_clipping +熵退火  device={device_str}")

    env = build_env(200, "C")
    N = env.topo.total_sat
    ac = ActorCritic(N, args.hidden, args.temperature).to(device)
    print(f"  params={sum(p.numel() for p in ac.parameters()):,}  sats={N}")
    print("-" * 60)

    train_ppo(env, ac, args)
    print("-" * 60)
    print("完成!")


if __name__ == "__main__":
    main()
