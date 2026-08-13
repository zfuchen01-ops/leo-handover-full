#!/usr/bin/env python3
"""Phase 2: MGCS 100ep + DRQN 200ep GPU — 不打断, 等跑完."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)
print("已连接\n", flush=True)

def run(cmd, timeout=900):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    stdout.channel.settimeout(0.3)
    stderr.channel.settimeout(0.3)
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
        if stderr.channel.recv_ready():
            data = stderr.channel.recv(4096).decode(errors='replace')
            if data.strip():
                print(f"[e] {data.strip()[:300]}", flush=True)
        if stdout.channel.exit_status_ready():
            break
        time.sleep(0.1)
    try:
        remain = stdout.read().decode(errors='replace')
        if remain.strip():
            print(remain, flush=True)
    except: pass

# Step 1: MGCS 100ep (脚本放对路径)
print("Step 1: MGCS 100ep...", flush=True)
run(
    "cd /root/leo_handover && mkdir -p log/topology log/RL log/model log/handover log/network && "
    "cat > mgcs_test.py << 'EOF'\n"
    "import os\n"
    "os.environ.setdefault('LEO_CAHS_VARIANT','paper')\n"
    "os.environ.setdefault('LEO_QUIET_LOGS','1')\n"
    "os.environ.setdefault('LEO_C_BAND','100')\n"
    "from baselines import build_env, run_baseline\n"
    "env=build_env(200)\n"
    "r,h,b,u=run_baseline(env,200,100,'MGCS')\n"
    "print(f'MGCS_100ep: rate={r:.0f} HO={h:.3f} beam={b:.3f} sat={u:.0f}')\n"
    "EOF\n"
    "PYTHONUNBUFFERED=1 python3 -u mgcs_test.py 2>&1",
    timeout=600)

# Step 2: DRQN 200ep GPU
print("\nStep 2: DRQN 200ep GPU — 别打断!", flush=True)
run(
    "cd /root/leo_handover && "
    "LEO_HO_PENALTY=0.1 LEO_FEAT_PER_SAT=4 LEO_ORTH_LAMBDA=0.1 PYTHONUNBUFFERED=1 "
    "python3 -u train_drqn.py --slots 200 --tag baseline --min-episodes 0 --patience 999 "
    "--device cuda 2>&1",
    timeout=900)

print("\nDone!")
c.close()
