import os, sys, copy, random as _random
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
import numpy as np; import torch
np.random.seed(0); _random.seed(42); torch.manual_seed(0)

from User import User; from Topology import Topology
from DRQNAgent import UserAgent, CenterAgent
from train_drqn import build_drqn_env

MODEL = "log/model/drqn_C_u200_tf4_v2_C_ep2550.pkl"
TAG, CONST, DEV = "tf4_mixed", "C", "cuda"
DENSITIES, EP, CYCLES = [50,100,150,200], 50, 5

print(f"Load {MODEL}", flush=True)
# 先用200用户初始化env，再创建agent
User.uid=0; Topology.index_con=0
env = build_drqn_env(200, CONST)
c_agent = CenterAgent(env, gamma=0.9, epsilon=0.05, batch=256, buffer=20000,
                       hidden_size=128, lr=0.0003, seq=6, device=DEV)
c_agent.evaluate_net = torch.load(MODEL, map_location=DEV, weights_only=False)
c_agent.target_net = copy.deepcopy(c_agent.evaluate_net)

for cycle in range(CYCLES):
    for d in DENSITIES:
        print(f"Cycle {cycle+1}/{CYCLES} dens={d}", flush=True)
        User.uid=0; Topology.index_con=0
        env = build_drqn_env(d, CONST); c_agent.env = env
        ua=[]
        for i in range(d):
            a=UserAgent(env.topo.user[i], env, c_agent, gamma=0.9, epsilon=0.05,
                        batch=256, buffer=2000, hidden_size=128, seq=6,
                        device=DEV, head_idx=min(i,199))
            a.use_transformer=True
            a.evaluate_net=copy.deepcopy(c_agent.evaluate_net).to(a.device)
            ua.append(a)
        c_agent.reset(ua)
        t=0; env.reset(t,"NETWORK_LOAD")
        for a in ua: a.reset(mode="train")
        for ep in range(EP):
            acts={}
            for a in ua:
                obs=a.observe("NETWORK_LOAD")
                acts[a.user]=a.step(obs,0.0)+1
            if ep==0: env.step(acts,"INITIAL")
            else: env.step(acts,"NETWORK")
            if len(c_agent.replayer)>=0.5*c_agent.replayer.capacity: c_agent.learn()
            if ep%10==0: c_agent.target_net=copy.deepcopy(c_agent.evaluate_net).to(DEV)
            t+=60; env.Update_Env(t,"NETWORK_LOAD")
        r=sum(a.user.allocate_band_to_gs for a in ua if a.user.sat_connected)/len(ua)
        print(f"  rate={r:.0f}", flush=True)
        ep_num=cycle*len(DENSITIES)*EP+(DENSITIES.index(d)+1)*EP
        torch.save(c_agent.evaluate_net,f"log/model/drqn_C_u200_{TAG}_ep{ep_num}.pkl")

torch.save(c_agent.evaluate_net,f"log/model/drqn_C_u200_{TAG}_final.pkl")
print("Done!", flush=True)
