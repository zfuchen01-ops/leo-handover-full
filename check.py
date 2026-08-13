#!/usr/bin/env python3
"""一键查GPU状态: DRQN在跑吗? 跑到哪了?"""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)

def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
    return stdout.read().decode(), stderr.read().decode()

# GPU
out, _ = run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
print(f"GPU: {out.strip()}")

# 进程
out, _ = run("ps aux | grep train_drqn | grep -v grep")
if out.strip():
    pid = out.strip().split()[1]
    cpu = out.strip().split()[2]
    print(f"DRQN: 运行中 PID={pid} CPU={cpu}%")
else:
    print("DRQN: 未运行")

# CSV
out, _ = run("tail -3 /root/leo_handover/log/RL/DRQN_baseline_per_ep.csv 2>/dev/null")
if out.strip():
    print(f"\n最新CSV:\n{out.strip()}")
else:
    print("CSV: 无数据")

# 日志
out, _ = run("tail -5 /tmp/drqn_baseline.log 2>/dev/null")
if out.strip():
    # 只取关键行
    for line in out.strip().split('\n'):
        if any(k in line for k in ['ep', '完成', 'TEST', '训练', '最终']):
            print(f"  {line}")

c.close()
