import os, sys
sys.path.insert(0,"/root/leo_handover")
os.environ["LEO_CAHS_VARIANT"]="paper"
os.environ["LEO_QUIET_LOGS"]="1"
os.environ["LEO_C_BAND"]="100"
os.environ["LEO_FEAT_PER_SAT"]="4"
from train_drqn import build_drqn_env
env=build_drqn_env(200,"C")
env.reset(0,"NETWORK_LOAD")
vc=[len(env.ho[u]) for u in env.topo.user]
print(f"min={min(vc)} max={max(vc)} avg={sum(vc)/len(vc):.1f}")
from collections import Counter
c=Counter(vc)
for k in sorted(c): print(f"  {k:2d}: {c[k]:3d}")
