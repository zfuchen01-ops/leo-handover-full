
import os, sys
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "0")
os.environ.setdefault("LEO_C_BAND", "100")
os.environ.setdefault("LEO_HO_PENALTY", "0.1")
os.environ.setdefault("LEO_ORTH_LAMBDA", "0.1")
os.environ.setdefault("LEO_FEAT_PER_SAT", "3")
os.environ.setdefault("LEO_USE_TRANSFORMER", "1")
from math import pi as _pi
import random, numpy as np; np.random.seed(0); random.seed(42)
import torch; torch.manual_seed(0)
from Topology import Topology; from User import User
from Network import Network; from paper_model import PaperRebuildHandover
from Defination import SAT_HEIGHT, TYPE_2PI
from DRQNAgent import UserAgent, CenterAgent, train_episode, TOP_K, FEAT_PER_SAT

def reset_ids():
    Topology.index_con = 0; User.uid = 0

def build_env(n=200):
    reset_ids(); t = Topology(); o, s = 16, 16; p = 1
    fp = 2.0 * p * _pi / (o * s); ln = 54.0 / 180.0 * _pi; th = 2.0 * _pi / o
    t.Add_Constellation(o, s, SAT_HEIGHT, fp, ln, th, TYPE_2PI); t.Each_Satellite()
    for lon, lat, nm in [(0,0,"G1"),(60,0,"G2"),(120,0,"G3"),(180,0,"G4"),(-120,0,"G5"),(-60,0,"G6")]:
        t.Add_Gateway_Loc(lon/180.0*_pi, lat/180.0*_pi, 5, nm)
    users = [(random.uniform(-120,120)*_pi/180.0, random.uniform(-60,60)*_pi/180.0) for _ in range(n)]
    t.Add_User_From_Input(users)
    for u in t.user: u.assigned_gateway = t.gateway[random.randrange(len(t.gateway))]
    return PaperRebuildHandover(net=Network(t))

print("构建环境...", flush=True)
env = build_env(200)

c_agent = CenterAgent(env, gamma=0.9, epsilon=0.01, batch=256, buffer=20000,
                      hidden_size=128, lr=0.001, seq=6, device='cuda')
# 加载已训练模型
ckpt = torch.load("log/model/drqn_C_u200.pkl", map_location='cuda', weights_only=False)
c_agent.evaluate_net = ckpt.to('cuda')
c_agent.target_net = torch.load("log/model/drqn_C_u200.pkl", map_location='cuda', weights_only=False).to('cuda')
print("模型加载完成", flush=True)

u_agents = []
for i in range(200):
    agent = UserAgent(env.topo.user[i], env, c_agent, gamma=0.9, epsilon=0.05,
                      batch=256, buffer=2000, hidden_size=128, seq=6, device='cuda', head_idx=i)
    agent.evaluate_net = torch.load("log/model/drqn_C_u200.pkl", map_location='cuda', weights_only=False).to('cuda')
    u_agents.append(agent)

print(f"TF3续跑 100ep (ε=0.05)...", flush=True)
import DRQNAgent
DRQNAgent.per_ep_log = open('log/RL/DRQN_tf3_cont_per_ep.csv', 'w')
DRQNAgent.per_ep_log.write('episode,reward,rate_avg,hops_avg,ho_avg,q_spread,uniq_sat,beam_avg,rew_rate,rew_ho,loss,q_gap_avg,q_gap_noise,n_valid\n')

train_episode(env, u_agents, c_agent, model="tf3_cont", mode='train',
              start_time=0, end_time=5000, time_step=50, net_step=20,
              batch=256, patience=999, min_episodes=0, conv_threshold=0.005)
print("续跑完成!", flush=True)

