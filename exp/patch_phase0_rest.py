import pathlib
B=pathlib.Path('..')
def edit(rel,pairs):
    p=B/rel; s=p.read_text()
    for old,new,n in pairs:
        assert s.count(old)==n,(rel,old[:40],s.count(old)); s=s.replace(old,new)
    p.write_text(s); print('ok',rel)
edit('train_drqn.py',[('每窗口=400 ep','每窗口=100 ep',1),('args.patience*400','args.patience*100',1)])
edit('DRQNAgent.py',[
    ('每个窗口=400 episodes','每个窗口=100 episodes',1),
    ('patience, patience * 400)','patience, patience * 100)',1),
    ("                    log.logger.info('  -> new best reward!')",
     "                    log.logger.info('  -> new best reward!')\n                    c_agent.save_net('./log/model/%s_best.pkl' % model)",1),
    ('            else:\n                if average_reward > best_avg_reward:\n                    best_avg_reward = average_reward\n            temp_reward = 0',
     '            else:\n                if average_reward > best_avg_reward:\n                    best_avg_reward = average_reward\n                    c_agent.save_net(\'./log/model/%s_best.pkl\' % model)\n            temp_reward = 0',1),
])
print('REST DONE')
