#!/usr/bin/env python3
"""上传并运行 Transformer 实验: 3特征(无ISL) vs 4特征(有ISL)."""
import paramiko, time, sys

FEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # 默认3特征先跑
TAG = f"tf{FEAT}"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)
print(f"已连接, 上传代码...", flush=True)

# 上传修改后的文件
sftp = c.open_sftp()
for f in ["DRQNAgent.py", "train_drqn.py", "Handover.py", "CLAUDE.md"]:
    sftp.put(f"/Users/cowboy/Downloads/LEO_handover_acceptance_20260608_014727(4)/{f}", f"/root/leo_handover/{f}")
    print(f"  ✓ {f}", flush=True)
sftp.close()

# 创建目录
c.exec_command("cd /root/leo_handover && mkdir -p log/topology log/RL log/model log/handover log/network")

print(f"\n=== Transformer {FEAT}特征 (GPU) ===", flush=True)
stdin, stdout, stderr = c.exec_command(
    f"cd /root/leo_handover && "
    f"LEO_HO_PENALTY=0.1 LEO_FEAT_PER_SAT={FEAT} LEO_ORTH_LAMBDA=0.1 LEO_USE_TRANSFORMER=1 "
    f"PYTHONUNBUFFERED=1 python3 -u train_drqn.py --slots 200 --tag {TAG} "
    f"--min-episodes 0 --patience 999 --device cuda 2>&1", timeout=900)

stdout.channel.settimeout(0.5)
end = time.time() + 900
buf = ""
while time.time() < end:
    if stdout.channel.recv_ready():
        data = stdout.channel.recv(4096).decode(errors='replace')
        if data:
            buf += data
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                if line.strip():
                    print(line, flush=True)
    if stderr.channel.recv_ready():
        data = stderr.channel.recv(4096).decode(errors='replace')
        if data.strip():
            print(f"[e] {data.strip()[:300]}", flush=True)
    if stdout.channel.exit_status_ready():
        break
    time.sleep(0.1)
try:
    r = stdout.read().decode(errors='replace')
    if r.strip(): print(r, flush=True)
except: pass

print(f"\nDone! TAG={TAG}", flush=True)
c.close()
