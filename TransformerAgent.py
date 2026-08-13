"""
Yang TSGH-REB 完整复现:
  PPO + Transformer + 4特征/星 + Cavai reward + 无背景流量
  State  (Eq.46): {prev_conn, Cavai/C_norm, RVT/600, reliability(=1/(1+hops))}
  Reward (Eq.50): Cavai/C_norm - 1.5*ho - 2.0*backhaul_interrupt
  Training:       Algorithm 3 — PPO, K-episode batch, γ=0.99
"""
import collections, copy, math, os, sys
import numpy as np; np.random.seed(0)
import random; random.seed(42)
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.99
LAMBDA = 0.95          # GAE
CLIP_EPS = 0.2
LR = 3e-4
PPO_EPOCHS = 4
K_EPISODES = 10        # Paper: update every K episodes
UE_USERS = 300         # Paper: 300 users

# ──────────── Transformer Policy+Value Network ────────────
class TransformerPPONet(nn.Module):
    def __init__(self, total_sats, feature_dim=4, d_model=128, nhead=8,
                 num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.total_sats = total_sats
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.embedding = nn.Linear(feature_dim, d_model)
        self.register_buffer('grid_pe', self._build_grid_pe(total_sats, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        # Policy head: per-sat scores → softmax probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        # Value head: global state value
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def _build_grid_pe(self, total_sats, d_model):
        orbits = int(math.sqrt(total_sats))
        sats_per = orbits
        d_half = d_model // 2
        pe = torch.zeros(total_sats, d_model)
        for i in range(total_sats):
            oidx = i // sats_per
            sidx = i % sats_per
            for k in range(d_half // 2):
                div = 10000.0 ** (2.0 * k / max(1, d_half // 2))
                pe[i, 2*k] = math.sin(oidx/div); pe[i, 2*k+1] = math.cos(oidx/div)
                pe[i, d_half+2*k] = math.sin(sidx/div); pe[i, d_half+2*k+1] = math.cos(sidx/div)
        return pe.unsqueeze(0)

    def forward(self, x, action_mask=None):
        N = self.total_sats; batch = x.shape[0]
        per_sat = x[:, :3*N].reshape(batch, 3, N).transpose(1, 2).contiguous()
        is_visible = per_sat[:, :, 1] > 0.0
        if not is_visible.any():
            return None, None, is_visible
        emb = self.embedding(per_sat) + self.grid_pe[:, :N, :]
        out = self.encoder(emb, src_key_padding_mask=~is_visible)
        # Policy
        logits = self.policy_head(out).squeeze(-1)  # [B, N]
        # Mask invisible → -inf
        logits[~is_visible] = -float('inf')
        if action_mask is not None:
            logits[action_mask] = -float('inf')
        # Global value: mean pool over visible
        mask = is_visible.float().unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value, is_visible

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            logits, value, _ = self.forward(obs.unsqueeze(0) if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs).unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                return probs.argmax().item(), value.item(), probs.squeeze(0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), value.item(), dist.log_prob(action)

# ──────────── PPO Buffer ────────────
class PPOBuffer:
    def __init__(self, max_size=100000):
        self.states = []; self.actions = []; self.rewards = []
        self.values = []; self.log_probs = []; self.dones = []
        self.max_size = max_size

    def store(self, s, a, r, v, logp, done):
        if len(self.states) >= self.max_size:
            self.states.pop(0); self.actions.pop(0); self.rewards.pop(0)
            self.values.pop(0); self.log_probs.pop(0); self.dones.pop(0)
        self.states.append(np.array(s, dtype=np.float32))
        self.actions.append(a); self.rewards.append(r)
        self.values.append(v); self.log_probs.append(logp)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value=0.0):
        """GAE: A_t = sum_{l} (γλ)^l * δ_{t+l}"""
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + GAMMA * (0 if self.dones[t] else
                     (self.values[t+1] if t+1 < len(self.rewards) else last_value)) - self.values[t]
            gae = delta + GAMMA * LAMBDA * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
        return returns, advantages

    def get_batch(self):
        s = torch.FloatTensor(np.array(self.states))
        a = torch.LongTensor(self.actions)
        r = torch.FloatTensor(self.rewards)
        v = torch.FloatTensor(self.values)
        lp = torch.stack(self.log_probs)
        d = torch.BoolTensor(self.dones)
        return s, a, r, v, lp, d

    def clear(self):
        self.states.clear(); self.actions.clear(); self.rewards.clear()
        self.values.clear(); self.log_probs.clear(); self.dones.clear()

    def __len__(self):
        return len(self.states)

# ──────────── PPO Update ────────────
def ppo_update(policy_net, optimizer, buffer, device='cpu', epochs=PPO_EPOCHS, clip_eps=CLIP_EPS):
    if len(buffer) == 0:
        return 0.0
    s, a, _, v_old, logp_old, _ = buffer.get_batch()
    s, a, v_old, logp_old = s.to(device), a.to(device), v_old.to(device), logp_old.to(device)
    returns, advantages = buffer.compute_returns_and_advantages()
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    adv_std = advantages.std()
    if adv_std > 1e-8:
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

    total_loss = 0.0
    for _ in range(epochs):
        logits, values, _ = policy_net(s)
        if logits is None:
            continue
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        logp_new = dist.log_prob(a)
        entropy = dist.entropy().mean()

        ratio = (logp_new - logp_old).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5*value_loss - 0.2*entropy  # paper: c1=0.5 c2=0.2

        if torch.isnan(loss) or torch.isinf(loss):
            print('  [WARN] NaN/Inf loss, skipping update', flush=True)
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    buffer.clear()
    return total_loss / max(1, epochs)

# ──────────── Training ────────────
def train_yang_ppo(env, policy_net, optimizer, episodes=500, device='cpu'):
    N = env.topo.total_sat
    users = list(env.topo.user)[:UE_USERS]  # paper: 300 users
    buffer = PPOBuffer(max_size=UE_USERS * K_EPISODES * 10)
    episode_rates = []; last_policy_loss = 0.0

    for ep in range(episodes):
        mode = 'INITIAL' if ep == 0 else 'NETWORK'
        random.shuffle(users)
        ep_rate, n_valid, ep_reward = 0.0, 0, 0.0

        for u in users:
            obs = env.Observe_Yang(u, 'NETWORK_LOAD')
            act, value, logp = policy_net.act(obs)
            prev_sat = u.sat_connected

            env.step({u: act+1}, mode)

            # Rate tracking
            if u.sat_connected and u.sat_connected in env.ho[u]:
                fd = env._get_feeder_sat(u, u.sat_connected)
                fb = (env.net.N2N_status[u.sat_connected.con_id-1]
                      [u.sat_connected.ID-1][fd.ID-1].free_band
                      if fd and u.sat_connected != fd else 9999)
                ep_rate += min(env.ho[u][u.sat_connected].c_quality, fb)
                n_valid += 1

            # Yang Eq.50 simplified: Cavai from OBS - ho penalty (no backhaul interrupt)
            if u.sat_connected and u.sat_connected in env.ho[u]:
                r = obs[N + act] - (1.5 if prev_sat is not None and u.sat_connected != prev_sat else 0.0)
            else:
                r = 0.0
            ep_reward += r

            buffer.store(obs, act, r, value, logp, done=False)

        # Mark end of episode (GAE bootstrap resets here)
        if len(buffer.dones) > 0:
            buffer.dones[-1] = True

        episode_rates.append(ep_rate / n_valid if n_valid else 0)
        env.Update_Env((ep+1)*30, 'NETWORK_LOAD')

        # PPO update every K episodes (paper Algorithm 3 line 22)
        if ep > 0 and ep % K_EPISODES == 0:
            last_policy_loss = ppo_update(policy_net, optimizer, buffer, device)

        avgR = sum(episode_rates[-100:]) / min(100, len(episode_rates))
        print(f'[PPO ep{ep:4d}] rate={ep_rate/n_valid:3.0f} avgR={avgR:3.0f} loss={last_policy_loss:.4f}')

        if ep > 0 and ep % 50 == 0:
            torch.save(policy_net.state_dict(), f'./log/model/ppo_yang_ep{ep}.pkl')
            print(f'  [SAVE ppo_yang_ep{ep}]')

    return episode_rates


if __name__ == '__main__':
    os.environ.setdefault('LEO_CAHS_VARIANT', 'paper')
    os.environ.setdefault('LEO_QUIET_LOGS', '1')
    os.environ.setdefault('LEO_C_BAND', '100')
    np.random.seed(0); random.seed(42); torch.manual_seed(0)
    sys.path.insert(0, '.')
    from train_drqn import build_drqn_env

    env = build_drqn_env(UE_USERS, 'C')
    env.reset(0, 'NETWORK_LOAD')
    N = env.topo.total_sat

    policy_net = TransformerPPONet(N, feature_dim=3, d_model=128, nhead=8, num_layers=2, dim_feedforward=512)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    n_params = sum(p.numel() for p in policy_net.parameters())
    print(f'Yang TSGH-REB (PPO): {N}sats {UE_USERS}users params={n_params:,}')
    print(f'State: 4N+1  Reward: Cavai/1000-1.5ho-2.0*backhaul  γ={GAMMA}')
    print(f'PPO: clip={CLIP_EPS} K={K_EPISODES} epochs={PPO_EPOCHS} lr={LR}')

    rewards = train_yang_ppo(env, policy_net, optimizer, episodes=500)
