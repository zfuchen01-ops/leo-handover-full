#!/usr/bin/env python3
"""启动DRQN后台训练到GPU, 启动后即可关终端."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)

# 创建目录 + nohup 后台启动
cmd = (
    "cd /root/leo_handover && mkdir -p log/topology log/RL log/model log/handover log/network && "
    "nohup bash -c 'LEO_HO_PENALTY=0.1 LEO_FEAT_PER_SAT=4 LEO_ORTH_LAMBDA=0.1 PYTHONUNBUFFERED=1 "
    "python3 -u train_drqn.py --slots 200 --tag baseline --min-episodes 0 --patience 999 "
    "--device cuda' > /tmp/drqn_baseline.log 2>&1 &"
)
stdin, stdout, stderr = c.exec_command(cmd, timeout=10)
stdout.read(); stderr.read()

time.sleep(4)

# 确认进程
stdin, stdout, stderr = c.exec_command("ps aux | grep train_drqn | grep -v grep", timeout=10)
procs = stdout.read().decode().strip()
if procs:
    print("✅ 训练已启动:", procs.split('\n')[0][:120])
else:
    print("❌ 进程未找到, 检查日志...")
    stdin, stdout, stderr = c.exec_command("tail -20 /tmp/drqn_baseline.log", timeout=10)
    print(stdout.read().decode())
    c.close()
    exit(1)

# 初始日志
time.sleep(3)
stdin, stdout, stderr = c.exec_command("tail -15 /tmp/drqn_baseline.log", timeout=10)
print("\n=== 初始日志 ===")
print(stdout.read().decode())
print("\n查看进度: tail -f /tmp/drqn_baseline.log (在远程)")
print("查看CSV:   tail -5 /root/leo_handover/log/RL/DRQN_baseline_per_ep.csv")
print("\n关闭此窗口不影响训练 —— nohup后台运行中")
c.close()
