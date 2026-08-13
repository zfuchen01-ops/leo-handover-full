#!/usr/bin/env python3
"""对比本地/远程环境关键参数, 诊断 rate 差异."""
import paramiko, subprocess, sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent

# === 本地 ===
print("=" * 50)
print("本地")
print("=" * 50)
result = subprocess.run([sys.executable, "-u", str(LOCAL / "diag.py")],
                        cwd=str(LOCAL), capture_output=True, text=True, timeout=60)
print(result.stdout)
if result.stderr.strip():
    print("STDERR:", result.stderr.strip()[:500])

# === 远程 ===
print("=" * 50)
print("远程 GPU")
print("=" * 50)
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)

sftp = c.open_sftp()
sftp.put(str(LOCAL / "diag.py"), "/root/leo_handover/diag.py")
sftp.close()

stdin, stdout, stderr = c.exec_command("cd /root/leo_handover && PYTHONUNBUFFERED=1 python3 -u diag.py", timeout=120)
print(stdout.read().decode())
err = stderr.read().decode()
if err.strip():
    print("STDERR:", err.strip()[:500])
c.close()
