#!/usr/bin/env python3
"""上传Transformer代码到GPU."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=34172, username="root", password="f/oqAQ+57I15", timeout=15)
sftp = c.open_sftp()
for f in ["DRQNAgent.py", "train_drqn.py", "Handover.py"]:
    sftp.put(f"/Users/cowboy/Downloads/LEO_handover_acceptance_20260608_014727(4)/{f}", f"/root/leo_handover/{f}")
    print(f"  ✓ {f}", flush=True)
sftp.close()
c.close()
print("上传完成!", flush=True)
