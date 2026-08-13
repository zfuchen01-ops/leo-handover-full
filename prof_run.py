#!/usr/bin/env python3
"""cProfile 定位剩余热点 (不改训练逻辑, 只套 profile)."""
import os, sys, time, random, cProfile, pstats, io
import numpy as np
import torch

random.seed(42)
np.random.seed(0)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tag = "prof"
slots = sys.argv[1] if len(sys.argv) > 1 else "30"

sys.argv = [
    "train_drqn.py",
    "--slots", slots,
    "--tag", tag,
    "--constellation", "C",
    "--users", "200",
    "--min-episodes", slots,
    "--patience", "5",
]
import train_drqn

pr = cProfile.Profile()
pr.enable()
train_drqn.main()
pr.disable()

s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('cumtime').print_stats(45)
print("===== PROFILE TOP 45 (cumtime) =====", flush=True)
print(s.getvalue(), flush=True)
print("===== PROFILE END =====", flush=True)
