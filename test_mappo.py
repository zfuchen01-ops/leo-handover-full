#!/usr/bin/env python3
"""Yang MAPPO验证: 每agent独立网络, 10 agent, 100MHz, ω=1.5"""
import os, sys, random, math
os.environ['LEO_CAHS_VARIANT']='paper'; os.environ['LEO_QUIET_LOGS']='1'
os.environ['LEO_C_BAND']='100'; sys.path.insert(0,'.')
import numpy as np; np.random.seed(0); random.seed(42)
import torch; torch.manual_seed(0)
import torch.nn as nn; import torch.nn.functional as F
from train_drqn import build_drqn_env
from TransformerAgent import TransformerPPONet, PPOBuffer

GAMMA=0.99; LAMBDA=0.95; CLIP=0.2; LR=3e-4; EPOCHS=4; K=1; EP=200; USERS=100; N_STEP=240; RL_RATIO=0.2
device='cuda' if torch.cuda.is_available() else 'cpu'

# 小MLP替代Transformer (每agent ~12K参数, 100agent ~1.2M总参数)
class MLPActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, action_dim)
        self.value = nn.Linear(hidden, 1)
    def forward(self, x):
        if x.dim()==1: x=x.unsqueeze(0)  # [F]→[1,F]
        h=self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1), None  # (logits, value, None=no visible mask)
    def act(self, obs, deterministic=False):
        logits, value, _ = self.forward(obs)
        # Mask invisible: obs[1] (Cavai feature) <=0
        if obs.dim()==1: obs_2d=obs.unsqueeze(0)
        else: obs_2d=obs
        N=obs_2d.shape[-1]//3
        is_vis=obs_2d[:,N:2*N]>0
        logits[~is_vis]=-float('inf')
        probs=F.softmax(logits,dim=-1)
        if deterministic:
            return probs.argmax().item(), value.item(), None
        dist=torch.distributions.Categorical(probs)
        act=dist.sample()
        return act.item(), value.item(), dist.log_prob(act).detach()

# ── MGCS ──
env_b=build_drqn_env(USERS,'C'); env_b.reset(0,'NETWORK_LOAD')
users_b=list(env_b.topo.user)[:USERS]; tr,th=0,0
for ep in range(10):
    er,en,eh=0,0,0; env_b._precompute_hops()
    for u in users_b:
        if u.sat_covered:
            lst=u.sat_connected; best=max(u.sat_covered,key=lambda s:env_b.ho[u][s].c_quality)
            env_b.step({u:best.ID},'NETWORK')
            if u.sat_connected!=lst: eh+=1
            if u.sat_connected and u.sat_connected in env_b.ho[u]:
                fd=env_b._get_feeder_sat(u,u.sat_connected)
                b=env_b.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else 9999
                er+=min(env_b.ho[u][u.sat_connected].c_quality,b); en+=1
    tr+=er/en; th+=eh; env_b.Update_Env((ep+1)*30,'NETWORK_LOAD')
print(f'MGCS({USERS}人): rate={tr/10:.0f} HO={th/10:.0f}',flush=True)

# ── MAPPO: 混合 RL+MGCS (杨: limited agent-driven users + background traffic) ──
env=build_drqn_env(USERS,'C'); env.reset(0,'NETWORK_LOAD')
N=env.topo.total_sat; users=list(env.topo.user)[:USERS]

# 划分 RL 用户和 MGCS 背景用户
n_rl = int(USERS * RL_RATIO); n_mgcs = USERS - n_rl
rl_users = users[:n_rl]           # RL agent 用户
mgcs_users = users[n_rl:]          # MGCS 背景流量用户
print(f'RL agents={n_rl} MGCS bg={n_mgcs} total={USERS}',flush=True)

# 仅 RL agent 有网络+优化器+buffer
input_dim=3*N
nets=[MLPActorCritic(input_dim,N).to(device) for _ in range(n_rl)]
opts=[torch.optim.Adam(n.parameters(),lr=LR) for n in nets]
bufs=[PPOBuffer(N_STEP*2) for _ in range(n_rl)]

# 背景流量: 杨 Algorithm 3 line 12 — 每ep重置
_preloaded=[]
def preload_reset():
    global _preloaded
    for lsa,amt in _preloaded: lsa.used_band=max(0.0,lsa.used_band-amt)
    _preloaded.clear()
    for con in env.net.LSDB:
        for node in con:
            for lsa in node:
                if lsa.isEstablished and random.random()<0.05:
                    amt=lsa.total_band*0.5; lsa.used_band+=amt
                    _preloaded.append((lsa,amt))

rates=[]; ho_rates=[]; pls=[0]*n_rl
N_STEP=240

def mgcs_act(env, u):
    """MGCS贪心: 选最高cq的可见卫星"""
    if not u.sat_covered: return None
    return max(u.sat_covered, key=lambda s: env.ho[u][s].c_quality).ID

for ep in range(EP):
    env.reset(0,'NETWORK_LOAD')
    env._precompute_hops()  # key: 预计算所有handover链路的cq
    for step in range(N_STEP):
        preload_reset()
        random.shuffle(rl_users); random.shuffle(mgcs_users)
        er,en,eh=0,0,0
        # ── RL 用户: 神经网络决策 + 记录经验 ──
        for i,u in enumerate(rl_users):
            obs=env.Observe_Yang(u,'NETWORK_LOAD'); act,val,logp=nets[i].act(torch.FloatTensor(obs).to(device)); prev=u.sat_connected
            env.step({u:act+1},'NETWORK')
            if u.sat_connected!=prev: eh+=1
            if u.sat_connected and u.sat_connected in env.ho[u]:
                fd=env._get_feeder_sat(u,u.sat_connected)
                fb=env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else 9999
                er+=min(env.ho[u][u.sat_connected].c_quality,fb); en+=1
                r=obs[N+act]-(1.5 if prev is not None and u.sat_connected!=prev else 0)
            else: r=0
            bufs[i].store(obs,act,r,val,logp,False)
        # ── MGCS 用户: 贪心决策(不记录,充当背景流量) ──
        for u in mgcs_users:
            if u.sat_covered:
                a = mgcs_act(env, u)
                if a is not None:
                    prev=u.sat_connected; env.step({u:a},'NETWORK')
                    if u.sat_connected!=prev: eh+=1
                    if u.sat_connected and u.sat_connected in env.ho[u]:
                        fd=env._get_feeder_sat(u,u.sat_connected)
                        fb=env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else 9999
                        er+=min(env.ho[u][u.sat_connected].c_quality,fb); en+=1
        # 标记done
        for b in bufs:
            if len(b.dones)>0: b.dones[-1]=True
        rates.append(er/en if en else 0); ho_rates.append(eh)
        env.Update_Env((step+1)*30,'NETWORK_LOAD')

    # PPO更新 — 仅RL agent
    if ep>=0 and (ep+1)%K==0:
        for i in range(n_rl):
            b=bufs[i]
            if len(b)==0: continue
            s,a,_,v,lp,_=b.get_batch()
            s=s.to(device); a=a.to(device); v=v.to(device); lp=lp.to(device)
            ret,adv=b.compute_returns_and_advantages()
            ret=torch.FloatTensor(ret).to(device)
            adv=torch.FloatTensor(adv).to(device)
            if adv.std()>1e-8: adv=(adv-adv.mean())/(adv.std()+1e-8)
            ent_coef=0.2
            tl=0
            for _ in range(EPOCHS):
                logits,vals,_=nets[i](s)
                is_vis=s[:,N:2*N]>0; logits[~is_vis]=-float('inf')
                probs=F.softmax(logits,dim=-1); dist=torch.distributions.Categorical(probs)
                lp_new=dist.log_prob(a); ent=dist.entropy().mean()
                ratio=(lp_new-lp).exp(); s1=ratio*adv; s2=torch.clamp(ratio,1-CLIP,1+CLIP)*adv
                loss=-torch.min(s1,s2).mean()+0.5*F.mse_loss(vals,ret)-ent_coef*ent
                if not torch.isnan(loss):
                    opts[i].zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(nets[i].parameters(),0.5); opts[i].step(); tl+=loss.item()
            pls[i]=tl/max(1,EPOCHS); b.clear()
    avgR=sum(rates[-10:])/min(10,len(rates)); avgHO=sum(ho_rates[-10:])/min(10,len(ho_rates))
    avg_loss=sum(pls)/n_rl
    print(f'[MAPPO ep{ep:3d}] rate={er/en:3.0f} avgR={avgR:3.0f} HO={eh:2d} loss={avg_loss:.4f} RL={n_rl}',flush=True)


print(f'\nFINAL: MGCS={tr/10:.0f} HO={th/10:.0f} | MAPPO ep={EP-1} RL={n_rl}/{USERS}',flush=True)
import json; json.dump({'rates':rates,'ho_rates':ho_rates,'losses':pls,'rl_ratio':RL_RATIO,'n_rl':n_rl},open('/tmp/mappo_data.json','w'))
print('Saved /tmp/mappo_data.json',flush=True)
