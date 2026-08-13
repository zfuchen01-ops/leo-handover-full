#!/usr/bin/env python3
"""远程GPU环境初始化: 升级PyTorch + 上传代码."""
import paramiko, os, sys, time
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 34172
USER = "root"
PASS = "f/oqAQ+57I15"
LOCAL_DIR = Path(__file__).resolve().parent

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"连接 {HOST}:{PORT}...", flush=True)
c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
print("已连接!", flush=True)

def run(cmd, timeout=120):
    print(f"  $ {cmd[:80]}...", flush=True)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if err.strip():
        print(f"  [err] {err.strip()[:200]}", flush=True)
    return out.strip(), err.strip()

# Step 1: 检查
print("\n=== 当前状态 ===", flush=True)
out, _ = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
print(f"  GPU: {out}", flush=True)
out, _ = run("python3 -c 'import torch; print(torch.__version__)' 2>&1")
print(f"  PyTorch: {out}", flush=True)

# Step 2: 升级PyTorch (RTX5090需要CUDA>=12.6)
print("\n=== 升级PyTorch for RTX 5090 (cu126) ===", flush=True)
out, _ = run(
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -q 2>&1",
    timeout=600
)
print("  完成!", flush=True)

# Step 3: 验证
out, _ = run(
    "python3 -c 'import torch; print(f\"CUDA:{torch.cuda.is_available()} GPU:{torch.cuda.get_device_name(0)} VRAM:{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB\")'"
)
print(f"\n=== 验证 ===\n  {out}", flush=True)

# Step 4: 创建远程工作目录
run("mkdir -p /root/leo_handover/log/RL /root/leo_handover/log/model /root/leo_handover/log/handover /root/leo_handover/log/network")

# Step 5: 上传代码
print("\n=== 上传代码 ===", flush=True)
sftp = c.open_sftp()
upload_files = [
    "Handover.py", "DRQNAgent.py", "train_drqn.py", "test_model.py", "baselines.py",
    "Topology.py", "User.py", "Network.py", "paper_model.py", "Defination.py",
    "compare_isl.py",
]
for f in upload_files:
    local = LOCAL_DIR / f
    if local.exists():
        remote = f"/root/leo_handover/{f}"
        sftp.put(str(local), remote)
        print(f"  ✓ {f}", flush=True)
    else:
        print(f"  ✗ {f} 不存在,跳过", flush=True)
sftp.close()

print(f"\n=== 完成! ===", flush=True)
print(f"远程目录: /root/leo_handover/", flush=True)
print(f"文件: {len(upload_files)} 个已上传", flush=True)
c.close()
