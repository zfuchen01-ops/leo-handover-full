"""测试256星LSA分配/释放 — 多用户长时间切换"""
import os, sys, copy
os.environ['LEO_QUIET_LOGS'] = '1'; os.environ['LEO_CAHS_VARIANT'] = 'paper'; os.environ['LEO_C_BAND'] = '500'
sys.path.insert(0, '.')
from train_drqn import build_drqn_env
from DRQNAgent import UserAgent, CenterAgent
import random; random.seed(42)

env = build_drqn_env(200, 'C')
c_agent = CenterAgent(env, gamma=0.9, epsilon=0.01, batch=256, buffer=20000, hidden_size=6, lr=0.001, seq=6, device='cpu')
u_agents = [UserAgent(u, env, c_agent, gamma=0.9, epsilon=0.01, batch=256, buffer=2000, hidden_size=6, seq=6, device='cpu') for u in env.topo.user]
env.reset(0, 'NETWORK_LOAD'); c_agent.reset(u_agents)
for a in u_agents: a.reset(mode='train')
ob_re = {a: [a.observe('NETWORK_LOAD'), 0.0] for a in u_agents}
actions = {}

def total_lsa():
    return sum(lsa.used_band for con in range(len(env.net.LSDB)) for src in range(len(env.net.LSDB[con])) for lsa in env.net.LSDB[con][src] if lsa.isEstablished)

# INITIAL
for agent in u_agents:
    ob_re[agent][0] = agent.observe('NETWORK_LOAD')
    h,c = agent.evaluate_net.init_hidden_state(256, False)
    actions[agent.user] = agent.step(ob_re[agent][0], ob_re[agent][1], h, c) + 1
env.step(actions, 'INITIAL'); actions.clear()
for agent in u_agents: ob_re[agent][1] = agent.get_reward()

prev_sat = {u.user_ID: u.sat_connected.ID if u.sat_connected else 0 for u in env.topo.user}

for ep in range(100):
    for agent in u_agents:
        ob_re[agent][0] = agent.observe('NETWORK_LOAD')
        h,c = agent.evaluate_net.init_hidden_state(256, False)
        actions[agent.user] = agent.step(ob_re[agent][0], ob_re[agent][1], h, c) + 1

    if len(u_agents[0].replayer.memory) >= 0.5 * u_agents[0].replayer.capacity:
        c_agent.learn()
        if ep % 20 == 0:
            c_agent.target_net = copy.deepcopy(c_agent.evaluate_net).to(c_agent.device)
        for agent in u_agents:
            agent.evaluate_net = copy.deepcopy(c_agent.evaluate_net).to(agent.device)

    env.net.enter_batch_mode(); env.step(actions, 'NETWORK'); env.net.exit_batch_mode()

    switches = sum(1 for u in env.topo.user if (u.sat_connected.ID if u.sat_connected else 0) != prev_sat.get(u.user_ID, 0))
    for u in env.topo.user: prev_sat[u.user_ID] = u.sat_connected.ID if u.sat_connected else 0

    actions.clear()
    for agent in u_agents: ob_re[agent][1] = agent.get_reward()
    env.Update_Env((ep+1)*50, 'NETWORK_LOAD')

    lsa = total_lsa()
    r = sum(a.get_reward() for a in u_agents)
    neg_bw = sum(1 for u in env.topo.user for d,bw in u.allocate_band.items() if bw < 0)
    neg_n2n = sum(1 for con in range(len(env.net.N2N_status)) for src in range(len(env.net.N2N_status[con])) for dst in range(len(env.net.N2N_status[con][src])) if env.net.N2N_status[con][src][dst].free_band < 0)

    alert = ''
    if neg_bw > 0: alert += ' NEG_BW!'
    if neg_n2n > 0: alert += ' NEG_N2N!'
    if ep % 10 == 9:
        print(f'ep {ep}: LSA={lsa:.0f} reward={r:.1f} switches={switches}{alert}', flush=True)

# disconnect all
lsa_before = total_lsa()
for u in list(env.topo.user):
    if u.sat_connected:
        env.net.User_Disconnect_Satellite(u, u.sat_connected)
        u.sat_connected.user_connected.discard(u)
        u.sat_connected = None
print(f'\nDisconnect ALL: {lsa_before:.0f} -> {total_lsa():.0f}', flush=True)
print('PASS' if total_lsa() < 5000 else f'FAIL: {total_lsa():.0f} remaining', flush=True)
