#!/usr/bin/env python3
import argparse, os, copy, random, sys
os.environ.setdefault('LEO_CAHS_VARIANT','paper'); os.environ.setdefault('LEO_QUIET_LOGS','1')
os.environ.setdefault('LEO_C_BAND','100'); os.environ.setdefault('LEO_HO_PENALTY','0.2')
os.environ.setdefault('LEO_ORTH_LAMBDA','0.1'); os.environ.setdefault('LEO_FEAT_PER_SAT','4')
os.environ.setdefault('LEO_USE_TRANSFORMER','0'); os.environ.setdefault('LEO_VARLEN','0')
import numpy as np, torch
from train_drqn import build_drqn_env
from Defination import SLOT_SECONDS
from Handover import RATE_UPPER
from DRQNAgent import CenterAgent, UserAgent, TOP_K

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--constellation', default='C'); p.add_argument('--users', type=int, default=200)
    p.add_argument('--slots', type=int, default=2000); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt', required=True); p.add_argument('--tail', type=int, default=100)
    p.add_argument('--device', default='auto'); p.add_argument('--tag', default='eval')
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if (args.device=='auto' and torch.cuda.is_available()) else args.device
    env = build_drqn_env(args.users, args.constellation, args.seed)
    c_agent = CenterAgent(env, gamma=0.9, epsilon=0.01, batch=256, buffer=20000, hidden_size=128, lr=1e-3, seq=6, device=device)
    u_agents = [UserAgent(u, env, c_agent, gamma=0.9, epsilon=0.01, batch=256, buffer=2000, hidden_size=128, seq=6, device=device, head_idx=i) for i,u in enumerate(env.topo.user)]
    net = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if isinstance(net, dict):
        sd = net.get('evaluate_net', net.get('target_net'))
        c_agent.evaluate_net.load_state_dict(sd)
    else:
        c_agent.evaluate_net = net
    c_agent.evaluate_net = c_agent.evaluate_net.to(device); c_agent.target_net = copy.deepcopy(c_agent.evaluate_net).to(device)
    for a in u_agents: a.evaluate_net = copy.deepcopy(c_agent.evaluate_net).to(device); a.lstm_h=None; a.lstm_c=None
    K = TOP_K; end_time = args.slots*SLOT_SECONDS; time=0; episode=0
    env.reset(time, 'NETWORK_LOAD'); c_agent.reset(u_agents); ob_re={}
    for a in u_agents: a.reset(mode='eval'); ob_re[a]=[a.observe('NETWORK_LOAD'),0.0]
    rates,hos=[],[]
    while time <= end_time:
        ep_rate=0.0; ep_ho=0.0; n_conn=0
        for a in u_agents:
            ob_re[a][0]=a.observe('NETWORK_LOAD'); action_idx=a.step(ob_re[a][0],0.0,None,None)
            sat_ids=getattr(a.user,'_topK_sat_ids',[0]*K); sat_id=sat_ids[action_idx] if action_idx<len(sat_ids) else sat_ids[0]
            env.step({a.user:sat_id}, 'INITIAL' if episode==0 else 'NETWORK'); u=a.user
            if u.sat_connected is not None and u.sat_connected in env.ho[u]:
                access=min(env.ho[u][u.sat_connected].c_quality, RATE_UPPER); fd=env._get_feeder_sat(u,u.sat_connected)
                isl_b=(env.net.N2N_status[u.sat_connected.con_id-1][u.sat_connected.ID-1][fd.ID-1].free_band if fd and u.sat_connected!=fd else RATE_UPPER)
                ep_rate+=min(access,isl_b,RATE_UPPER); ep_ho+=1.0 if u.sat_connected!=u.last_connected else 0.0; n_conn+=1
        if n_conn>0: rates.append(ep_rate/n_conn); hos.append(ep_ho/n_conn)
        env.Update_Env(time,'NETWORK_LOAD'); time+=SLOT_SECONDS; episode+=1
    env.close(); n=len(rates)
    if n==0: print(f'[{args.tag}] ERROR no connected'); sys.exit(1)
    tr=rates[-args.tail:]; th=hos[-args.tail:]
    print(f'[{args.tag}] tail{min(args.tail,n)} rate={np.mean(tr):.1f}+-{np.std(tr):.1f} ho={np.mean(th):.3f}+-{np.std(th):.3f} ep={n}', flush=True)
    with open(f'./log/RL/eval_{args.tag}.csv','w') as f:
        f.write('episode,rate,ho\n')
        for i,(r,h) in enumerate(zip(rates,hos)): f.write(f'{i},{r:.1f},{h:.3f}\n')
if __name__=='__main__': main()
