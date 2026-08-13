import os,sys,copy,random as _random
os.environ.setdefault("LEO_CAHS_VARIANT","paper");os.environ.setdefault("LEO_QUIET_LOGS","1")
import numpy as np;import torch
np.random.seed(0);_random.seed(42);torch.manual_seed(0)
from User import User;from Topology import Topology
from DRQNAgent import UserAgent, CenterAgent
from train_drqn import build_drqn_env

MODEL="log/model/drqn_C_u200_tf4_v2_C_ep2550.pkl"
DEV="cuda"
DENSITIES=[50,100,200,50,100,200,50,100,200,50,100,200]  # 循环4轮

print(f"Load {MODEL}",flush=True)
User.uid=0;Topology.index_con=0
env0=build_drqn_env(200,"C")
c=CenterAgent(env0,gamma=0.9,epsilon=0.1,batch=256,buffer=20000,hidden_size=128,lr=0.0003,seq=6,device=DEV)
c.evaluate_net=torch.load(MODEL,map_location=DEV,weights_only=False)
c.target_net=copy.deepcopy(c.evaluate_net)

for phase,d in enumerate(DENSITIES):
 print(f"Phase {phase+1}/{len(DENSITIES)} dens={d}",flush=True)
 User.uid=0;Topology.index_con=0
 env=build_drqn_env(d,"C");c.env=env
 ua=[]
 for i in range(d):
  a=UserAgent(env.topo.user[i],env,c,gamma=0.9,epsilon=0.1,batch=256,buffer=2000,hidden_size=128,seq=6,device=DEV,head_idx=i)
  a.use_transformer=True;a.evaluate_net=copy.deepcopy(c.evaluate_net).to(a.device);ua.append(a)
 c.reset(ua)  # 重要：不重建replayer，保留buffer
 t=0;env.reset(t,"NETWORK_LOAD")
 for a in ua:a.reset(mode="train");a.epsilon=0.1
 for ep in range(100):
  acts={}
  for a in ua:acts[a.user]=a.step(a.observe("NETWORK_LOAD"),0.0,None,None)+1
  if ep==0:env.step(acts,"INITIAL")
  else:env.step(acts,"NETWORK")
  if all(len(a.replayer.memory)>=a.sequence+1 for a in ua):c.learn()
  if ep%10==0:c.target_net=copy.deepcopy(c.evaluate_net).to(DEV)
  t+=60;env.Update_Env(t,"NETWORK_LOAD")
 rates=[sum(a.user.allocate_band.values()) for a in ua if a.user.sat_connected and hasattr(a.user,'allocate_band') and a.user.allocate_band]
 print(f"  rate={sum(rates)/len(ua):.0f}" if rates else "  n/a",flush=True)

torch.save(c.evaluate_net,f"log/model/drqn_C_u_mix_final.pkl")
print("Done!",flush=True)
