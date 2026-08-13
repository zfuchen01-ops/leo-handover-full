#!/usr/bin/env python3
"""DRQN 200ep GPU训练."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)
print("已连接, DRQN GPU训练 200ep...\n", flush=True)

stdin, stdout, stderr = c.exec_command(
    "cd /root/leo_handover && mkdir -p log/topology log/RL log/model log/handover log/network && "
    "LEO_HO_PENALTY=0.1 LEO_FEAT_PER_SAT=4 LEO_ORTH_LAMBDA=0.1 PYTHONUNBUFFERED=1 "
    "python3 -u train_drqn.py --slots 200 --tag baseline --min-episodes 0 --patience 999 "
    "--device cuda 2>&1", timeout=900)

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

print("\nDone!")
c.close()
