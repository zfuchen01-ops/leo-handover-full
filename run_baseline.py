#!/usr/bin/env python3
"""只跑传统Baseline: MGCS/MRVT/Elev/Min-Load/Random, 100ep."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)
print("已连接\n", flush=True)

def run(cmd, timeout=900):
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
                        print(line, flush=True)
        if stdout.channel.exit_status_ready():
            break
        time.sleep(0.1)
    try:
        r = stdout.read().decode(errors='replace')
        if r.strip(): print(r, flush=True)
    except: pass

print("=== 传统Baseline 100ep ===", flush=True)
run(
    "cd /root/leo_handover && mkdir -p log/topology log/RL log/model log/handover log/network && "
    "PYTHONUNBUFFERED=1 python3 -u baselines.py --users 200 --ep 100 2>&1",
    timeout=900)

print("\nDone!")
c.close()
