#!/usr/bin/env python3
"""等价性验证 wrapper.

作用: 固定 python-random / numpy / torch 三种子 + cuDNN 确定性, 复用 train_drqn.main
的完整训练流程, 保证「同 tag 不同代码版本」可以逐位对比。

用法:
    python3 run_verify.py <tag> <slots>
      tag   : 决定输出 CSV / 模型文件名 (base1 / base2 / patch1 ...)
      slots : episode 数 (end_time = slots*50ms)
"""
import os
import sys
import time
import random

import numpy as np
import torch

# ── 强制种子 (必须在任何网络构建/随机消耗之前) ──
random.seed(42)
np.random.seed(0)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

# cuDNN 确定性: 消除 cuDNN LSTM 非确定性, 让 diff 的「0 差异」有明确含义
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tag = sys.argv[1] if len(sys.argv) > 1 else "verify"
slots = sys.argv[2] if len(sys.argv) > 2 else "60"

print(f"[VERIFY] seed: random=42 np=0 torch=1234 cudnn_deterministic=True", flush=True)
print(f"[VERIFY] tag={tag} slots={slots}", flush=True)

# 复用 train_drqn.main 的完整训练路径 (不 hack 训练逻辑, 保证验证的就是真实训练)
sys.argv = [
    "train_drqn.py",
    "--slots", str(slots),
    "--tag", tag,
    "--constellation", "C",
    "--users", "200",
    "--min-episodes", str(slots),
    "--patience", "5",
]
import train_drqn

t0 = time.time()
train_drqn.main()
t1 = time.time()
print(f"[VERIFY] DONE tag={tag} slots={slots} wall={t1 - t0:.1f}s", flush=True)
