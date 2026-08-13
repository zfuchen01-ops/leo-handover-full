import os, sys, copy, random as _random
os.environ.setdefault("LEO_CAHS_VARIANT","paper"); os.environ.setdefault("LEO_QUIET_LOGS","1")
import numpy as np; import torch
np.random.seed(0); _random.seed(42); torch.manual_seed(0)
from User import User; from Topology import Topology
from DRQNAgent import UserAgent, CenterAgent
from train_drqn import build_drqn_env
from DRQNAgent import train_episode

SRC = "log/model/drqn_C_u200_tf4_v2_C_ep2550.pkl"
TAG, CONST, DEV, DENSITY = "tf4_D50_freeze", "D", "cuda", 50

print(f"Load {SRC}", flush=True)
User.uid = 0; Topology.index_con = 0
env = build_drqn_env(DENSITY, CONST)
c = CenterAgent(env, gamma=0.9, epsilon=1.0, batch=256, buffer=20000,
                hidden_size=128, lr=0.0001, seq=6, device=DEV)
c.evaluate_net = torch.load(SRC, map_location=DEV, weights_only=False)
c.target_net = copy.deepcopy(c.evaluate_net)

# Freeze: 冻住 LSTM + Transformer + Embedding, 只微调 attention heads
frozen = 0
for name, p in c.evaluate_net.named_parameters():
    if any(x in name for x in ["lstm","transformer","agent_embed"]):
        p.requires_grad = False; frozen += 1
print(f"Frozen {frozen} params, only heads trainable", flush=True)

ua = []
for i in range(DENSITY):
    a = UserAgent(env.topo.user[i], env, c, gamma=0.9, epsilon=1.0,
                  batch=256, buffer=2000, hidden_size=128, seq=6, device=DEV, head_idx=i)
    a.use_transformer = True
    a.evaluate_net = copy.deepcopy(c.evaluate_net).to(a.device); ua.append(a)

import DRQNAgent
DRQNAgent.per_ep_log.close()
DRQNAgent.per_ep_log = open(f"./log/RL/DRQN_{TAG}_per_ep.csv", "w")
DRQNAgent.per_ep_log.write("episode,reward,rate_avg,hops_avg,ho_avg,q_spread,uniq_sat,beam_avg,rew_rate,rew_ho,loss,q_gap_avg,q_gap_noise,n_valid\n")

train_episode(env, ua, c, model=f"drqn_{CONST}_u{DENSITY}_{TAG}", mode="train",
              start_time=0, end_time=300*50, time_step=50, net_step=20,
              batch=256, patience=5, min_episodes=300)
print("Done!", flush=True)
