#!/usr/bin/env python3
"""连接新GPU, 检查环境 + 上传代码 + 装PyTorch."""
import paramiko, os, time

HOST = "connect.westb.seetacloud.com"
PORT = 30594
PASS = "mHwOnYbRqqfQ"
LOCAL = "/Users/cowboy/Downloads/LEO_handover_acceptance_20260608_014727(4)"
REMOTE = "/root/leo_handover"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASS, timeout=15)
print("已连接\n", flush=True)

def run(cmd, timeout=120):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(), stderr.read().decode()

# GPU check
out, err = run("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")
print(f"GPU: {out.strip()}", flush=True)

# PyTorch
out, err = run("python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1")
print(f"PyTorch: {out.strip()}", flush=True)

# CUDA version
out, _ = run("nvidia-smi | head -3 | tail -1")
print(f"CUDA: {out.strip().split('CUDA Version:')[-1].strip() if 'CUDA' in out else 'unknown'}", flush=True)

# 上传代码
run(f"rm -rf {REMOTE} && mkdir -p {REMOTE}/log/RL {REMOTE}/log/model {REMOTE}/log/handover {REMOTE}/log/network {REMOTE}/log/topology")
sftp = c.open_sftp()
n = 0
for f in os.listdir(LOCAL):
    if f.endswith('.py'):
        sftp.put(os.path.join(LOCAL, f), f"{REMOTE}/{f}")
        n += 1
sftp.close()
print(f"\n代码: {n} 文件已上传", flush=True)

# 如果PyTorch不行, 提示
if "False" in out or "Error" in err:
    print("\n⚠️ CUDA不可用, 需装PyTorch:", flush=True)
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall", flush=True)

c.close()
print("\nDone!", flush=True)
