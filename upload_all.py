#!/usr/bin/env python3
"""上传所有Python文件和依赖到远程GPU."""
import paramiko
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
REMOTE = "/root/leo_handover"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)

# 创建目录
stdin, stdout, stderr = c.exec_command(f"mkdir -p {REMOTE}/log/RL {REMOTE}/log/model {REMOTE}/log/handover {REMOTE}/log/network")
stdout.read(); stderr.read()

# 上传所有 .py 文件
sftp = c.open_sftp()
py_files = sorted(LOCAL.glob("*.py"))
n = 0
for f in py_files:
    remote_path = f"{REMOTE}/{f.name}"
    sftp.put(str(f), remote_path)
    n += 1
    print(f"  {f.name}", flush=True)

sftp.close()
c.close()
print(f"\n✅ {n} 个文件 → {REMOTE}/")
