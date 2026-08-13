#!/usr/bin/env python3
"""GPU远程遥控器 — 跑一次挂着, 我用它发命令."""
import paramiko, time, sys

HOST = "connect.westd.seetacloud.com"
PORT = 34172

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"连接 {HOST}:{PORT}...", flush=True)
c.connect(HOST, port=PORT, username="root", password="f/oqAQ+57I15", timeout=15)
print("已连接! 等待命令...\n", flush=True)

# 从 stdin 读命令, 在远程执行
for line in sys.stdin:
    cmd = line.strip()
    if not cmd or cmd.startswith('#'):
        continue
    if cmd == 'exit' or cmd == 'quit':
        break

    print(f"\n>>> {cmd}", flush=True)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=3600)

    import select
    stdout.channel.settimeout(0.3)
    end = time.time() + 3600
    buf = ""
    while time.time() < end:
        if stdout.channel.recv_ready():
            data = stdout.channel.recv(4096).decode(errors='replace')
            if data:
                buf += data
                while '\n' in buf:
                    line_out, buf = buf.split('\n', 1)
                    print(line_out, flush=True)
        if stderr.channel.recv_ready():
            data = stderr.channel.recv(4096).decode(errors='replace')
            if data.strip():
                print(f"  [e] {data.strip()[:200]}", flush=True)
        if stdout.channel.exit_status_ready():
            break
        time.sleep(0.05)
    try:
        r = stdout.read().decode(errors='replace')
        if r.strip(): print(r, flush=True)
    except: pass

c.close()
print("断开", flush=True)
