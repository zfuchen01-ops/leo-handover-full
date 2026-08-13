#!/usr/bin/env python3
"""Phase 1+2: 上传修正版baseline + 跑baseline + 跑DRQN.
   python3 run_all.py [baseline|drqn|both]"""
import paramiko, time, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "both"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)
print("已连接\n", flush=True)

def run(cmd, timeout=900, prefix=""):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    stdout.channel.settimeout(0.5)
    end = time.time() + timeout
    buf = ""
    while time.time() < end:
        if stdout.channel.recv_ready():
            data = stdout.channel.recv(4096).decode(errors='replace')
            if data:
                buf += data
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    if line.strip():
                        print(f"{prefix}{line}", flush=True)
        if stdout.channel.exit_status_ready():
            break
        time.sleep(0.1)
    try:
        r = stdout.read().decode(errors='replace')
        if r.strip(): print(r, flush=True)
    except: pass

# 上传修正版
sftp = c.open_sftp()
for f in ["baselines.py", "CLAUDE.md"]:
    sftp.put(f"/Users/cowboy/Downloads/LEO_handover_acceptance_20260608_014727(4)/{f}", f"/root/leo_handover/{f}")
    print(f"  ✓ {f} uploaded", flush=True)
sftp.close()
c.exec_command("cd /root/leo_handover && mkdir -p log/topology log/RL log/model log/handover log/network")

if mode in ("baseline", "both"):
    print("\n=== Baseline 100ep (修正版) ===", flush=True)
    run("cd /root/leo_handover && PYTHONUNBUFFERED=1 python3 -u baselines.py --users 200 --ep 100 2>&1", timeout=900)

if mode in ("drqn", "both"):
    print("\n=== DRQN 200ep GPU ===", flush=True)
    run(
        "cd /root/leo_handover && "
        "LEO_HO_PENALTY=0.1 LEO_FEAT_PER_SAT=4 LEO_ORTH_LAMBDA=0.1 PYTHONUNBUFFERED=1 "
        "python3 -u train_drqn.py --slots 200 --tag baseline --min-episodes 0 --patience 999 "
        "--device cuda 2>&1", timeout=900)

print("\nDone!")
c.close()
