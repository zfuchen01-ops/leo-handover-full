#!/usr/bin/env python3
"""Shared PPO: 1个Transformer服务所有agent, pool buffer一次更新, K=1, w=3, 200用户"""
import os, sys, random, math, json, time
os.environ['LEO_CAHS_VARIANT']='paper'; os.environ['LEO_QUIET_LOGS']='1'
os.environ['LEO_C_BAND']='100'; sys.path.insert(0,'.')
import numpy as np; np.random.seed(0); random.seed(42)
import torch; torch.manual_seed(0)
import torch.nn as nn; import torch.nn.functional as F
from train_drqn import build_drqn_env
from TransformerAgent import TransformerPPONet, PPOBuffer

GAMMA=0.99; LAMBDA=0.95; CLIP=0.2; LR=3e-4; EPOCHS=4; K=1; EP=500; USERS=200; HO_W=3.0
device='cuda' if torch.cuda.is_available() else 'cpu'

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
mgcs_r=tr/10; mgcs_ho=th/10
print(f'MGCS({USERS}人): rate={mgcs_r:.0f} HO={mgcs_ho:.0f}',flush=True)

# ── PPO ──
env=build_drqn_env(USERS,'C'); env.reset(0,'NETWORK_LOAD')
N=env.topo.total_sat; users=list(env.topo.user)[:USERS]
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
net=TransformerPPONet(N,feature_dim=3,d_model=256,dim_feedforward=512).to(device)
opt=torch.optim.Adam(net.parameters(),lr=LR)
bufs=[PPOBuffer(K*2) for _ in range(USERS)]  # 杨: 每agent独立buffer, 每buffer 40条
rates=[]; ho_rates=[]; pl=0
csv_f=open('./log/RL/PPO_shared_per_ep.csv','w')
csv_f.write('episode,reward,rate_avg,ho_avg,loss\n')

for ep in range(EP):
    mode='INITIAL' if ep==0 else 'NETWORK'; random.shuffle(users); er,en,eh=0,0,0
    for i,u in enumerate(users):
        obs=env.Observe_Yang(u,'NETWORK_LOAD'); act,val,logp=net.act(torch.FloatTensor(obs).to(device)); prev=u.sat_connected
        env.step({u:act+1},mode)
        if u.sat_connected!=prev: eh+=1
        if u.sat_connected and u.sat_connected in env.ho[u]:
            fd=env._get_feeder_sat(u,u.sat_connected)
            fb=env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else 9999
            er+=min(env.ho[u][u.sat_connected].c_quality,fb); en+=1
            r=obs[N+act]-(HO_W if prev is not None and u.sat_connected!=prev else 0)
        else: r=0
        bufs[i].store(obs,act,r,val,logp,False)
    for b in bufs:
        if len(b.dones)>0: b.dones[-1]=True
    rates.append(er/en if en else 0); ho_rates.append(eh); env.Update_Env((ep+1)*30,'NETWORK_LOAD')
    if ep>0 and ep%K==0:
        all_s,all_a,all_v,all_lp,all_ret,all_adv=[],[],[],[],[],[]
        for b in bufs:
            if len(b)==0: continue
            s,a,_,v,lp,_=b.get_batch()
            ret,adv=b.compute_returns_and_advantages()
            all_s.append(s); all_a.append(a); all_v.append(v); all_lp.append(lp)
            all_ret.append(torch.FloatTensor(ret)); all_adv.append(torch.FloatTensor(adv))
            b.clear()
        if not all_s: continue
        s=torch.cat([t.to(device) for t in all_s]); a=torch.cat([t.to(device) for t in all_a])
        vo=torch.cat([t.to(device) for t in all_v]); lpo=torch.cat([t.to(device) for t in all_lp])
        ret=torch.cat(all_ret).to(device)/10.0; adv=torch.cat(all_adv).to(device)/10.0
        if adv.std()>1e-8: adv=(adv-adv.mean())/(adv.std()+1e-8)
        ent_coef=0.05-(0.049*min(ep,int(EP*0.25))/int(EP*0.25))  # 快速衰减, 前50ep收敛
        tl=0
        for _ in range(EPOCHS):
            logits,vals,_=net(s)
            probs=F.softmax(logits,dim=-1); dist=torch.distributions.Categorical(probs)
            lp=dist.log_prob(a); ent=dist.entropy().mean()
            ratio=(lp-lpo).exp(); s1=ratio*adv; s2=torch.clamp(ratio,1-CLIP,1+CLIP)*adv
            loss=-torch.min(s1,s2).mean()+0.5*F.mse_loss(vals,ret)-ent_coef*ent
            if not torch.isnan(loss):
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),0.5); opt.step(); tl+=loss.item()
        pl=tl/max(1,EPOCHS)
    avgR=sum(rates[-10:])/min(10,len(rates)); avgHO=sum(ho_rates[-10:])/min(10,len(ho_rates))
    print(f'[PPO ep{ep:3d}] rate={er/en:3.0f} avgR={avgR:3.0f} HO={eh:3d} loss={pl:.4f}',flush=True)
    csv_f.write(f'{ep},{er/en:.0f},{er/en:.0f},{eh/USERS:.3f},{pl:.4f}\n')
    if ep%20==0: csv_f.flush()
    if ep>0 and ep%20==0:
        gr,gn,gh,gisl=0,0,0,[]
        for u in users:
            obs=env.Observe_Yang(u,'NETWORK_LOAD'); act,_,_=net.act(torch.FloatTensor(obs).to(device),deterministic=True)
            last=u.sat_connected; env.step({u:act+1},'NETWORK')
            if u.sat_connected!=last: gh+=1
            if u.sat_connected and u.sat_connected in env.ho[u]:
                fd=env._get_feeder_sat(u,u.sat_connected)
                fb=env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else 9999
                gr+=min(env.ho[u][u.sat_connected].c_quality,fb); gn+=1
            feeder=env._get_feeder_sat(u)
            if feeder and u.sat_connected:
                vis=[(s.ID-1,env.net.N2N_status[s.con_id-1][s.ID-1][feeder.ID].free_band) for s in u.sat_covered]
                vis.sort(key=lambda x:-x[1])
                cr=next((r for r,(sid,_) in enumerate(vis) if sid==u.sat_connected.ID-1),len(vis)-1)
                gisl.append(cr/max(1,len(vis)-1) if len(vis)>1 else 0.0)
        gisl_avg=sum(gisl)/len(gisl) if gisl else 0.5
        print(f'  [GREEDY ep{ep}] rate={gr/gn:.0f} HO={gh} ISLrank={gisl_avg:.3f}',flush=True)
        env.Update_Env((ep+1)*30,'NETWORK_LOAD')

# ── 正式测试 ──
print(f'\n[TEST ep{EP}] running 20ep...',flush=True)
env_t=build_drqn_env(USERS,'C'); env_t.reset(0,'NETWORK_LOAD')
users_t=list(env_t.topo.user)[:USERS]
test_rates,test_isl,test_ho=[],[],[]
for ep in range(20):
    er,en,eh=0,0,0; env_t._precompute_hops()
    for u in users_t:
        obs=env_t.Observe_Yang(u,'NETWORK_LOAD'); act,_,_=net.act(torch.FloatTensor(obs).to(device),deterministic=True)
        last=u.sat_connected; env_t.step({u:act+1},'NETWORK')
        if u.sat_connected!=last: eh+=1
        if u.sat_connected and u.sat_connected in env_t.ho[u]:
            fd=env_t._get_feeder_sat(u,u.sat_connected)
            fb=env_t.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else 9999
            er+=min(env_t.ho[u][u.sat_connected].c_quality,fb); en+=1
        feeder=env_t._get_feeder_sat(u)
        if feeder and u.sat_connected:
            vis=[(s.ID-1,env_t.net.N2N_status[s.con_id-1][s.ID-1][feeder.ID].free_band) for s in u.sat_covered]
            vis.sort(key=lambda x:-x[1])
            cr=next((r for r,(sid,_) in enumerate(vis) if sid==u.sat_connected.ID-1),len(vis)-1)
            test_isl.append(cr/max(1,len(vis)-1) if len(vis)>1 else 0.0)
    test_rates.append(er/en if en else 0); test_ho.append(eh)
    env_t.Update_Env((ep+1)*30,'NETWORK_LOAD')
r_test=sum(test_rates)/20; l50=sum(test_rates[-10:])/10
isl_test=sum(test_isl)/len(test_isl) if test_isl else 0.5; ho_test=sum(test_ho)/20
print(f'[TEST ep{EP}] rate={r_test:.0f} last50={l50:.0f} ISLrank={isl_test:.3f} HO={ho_test:.0f}/ep',flush=True)
print(f'\nCOMPARE: MGCS rate={mgcs_r:.0f} HO={mgcs_ho:.0f} | PPO rate={r_test:.0f} HO={ho_test:.0f}',flush=True)
