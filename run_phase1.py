#!/usr/bin/env python3
"""Phase 1: 传统Baseline + 自动补传缺失文件.
运行: python3 run_phase1.py
"""
import paramiko, time, os
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 34172
USER = "root"
PASS = "f/oqAQ+57I15"
REMOTE = "/root/leo_handover"
LOCAL = Path(__file__).resolve().parent

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("连接GPU服务器...", flush=True)
c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
print("已连接\n", flush=True)

def run(cmd, timeout=300):
    print(f"  >>> {cmd[:120]}", flush=True)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if err.strip():
        for line in err.strip().split('\n')[-5:]:
            print(f"  [stderr] {line}", flush=True)
    return out, err

# Step 0: 补传缺失的依赖
print("=" * 50)
print("Step 0: 上传所有依赖")
print("=" * 50)
run("mkdir -p " + REMOTE + "/log/RL " + REMOTE + "/log/model " + REMOTE + "/log/handover " + REMOTE + "/log/network")

all_files = [f.name for f in LOCAL.glob("*.py")]
core_files = [
    "Constellation.py", "DataStructure.py", "Gateway.py", "Logger.py",
    "Orbit.py", "Position.py", "Satellite.py",
    "Topology.py", "User.py", "Network.py", "Handover.py", "Defination.py",
    "paper_model.py", "DRQNAgent.py", "train_drqn.py", "test_model.py",
    "baselines.py", "compare_isl.py",
]
missing = [f for f in core_files if f in all_files]
sftp = c.open_sftp()
for f in missing:
    local = LOCAL / f
    remote = f"{REMOTE}/{f}"
    sftp.put(str(local), remote)
    print(f"  ✓ {f}", flush=True)
sftp.close()
print(f"  共上传 {len(missing)} 个文件\n", flush=True)

# Step 1: GPU验证
print("=" * 50)
print("Step 1: GPU验证")
print("=" * 50)
out, err = run("python3 -c 'import torch; print(f\"CUDA:{torch.cuda.is_available()} GPU:{torch.cuda.get_device_name(0)} VRAM:{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB\")'")
print(f"  {out.strip()}", flush=True)

# Step 2: baselines.py
print("\n" + "=" * 50)
print("Step 2: 传统Baseline (MGCS/MRVT/Elev/Min-Load/Random)")
print("=" * 50)
t0 = time.time()
out, err = run(f"cd {REMOTE} && PYTHONUNBUFFERED=1 python3 -u baselines.py --users 200 --ep 20")
print(out)
if err.strip():
    print(f"\n[STDERR]:\n{err.strip()[-500:]}")
print(f"\n⏱ 耗时: {time.time()-t0:.0f}s", flush=True)

c.close()
print("\n✅ Phase 1 完成!")
