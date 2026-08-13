#!/usr/bin/env python3
"""上传修正后baselines.py并重跑100ep."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)
print("上传修正版baselines.py...", flush=True)

sftp = c.open_sftp()
sftp.put("/Users/cowboy/Downloads/LEO_handover_acceptance_20260608_014727(4)/baselines.py", "/root/leo_handover/baselines.py")
sftp.close()
print("已上传\n", flush=True)

stdin, stdout, stderr = c.exec_command(
    "cd /root/leo_handover && PYTHONUNBUFFERED=1 python3 -u baselines.py --users 200 --ep 100 2>&1", timeout=900)

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
    if stdout.channel.exit_status_ready():
        break
    time.sleep(0.1)
try:
    r = stdout.read().decode(errors='replace')
    if r.strip(): print(r, flush=True)
except: pass

print("\nDone!")
c.close()
