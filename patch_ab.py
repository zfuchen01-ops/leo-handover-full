#!/usr/bin/env python3
"""应用加速补丁 A+B, 带 assert 保证精确匹配 (不匹配则报错不写入).

补丁 A: 删除 UserAgent.reset 里的 target_net 死代码
        (UserAgent.step 从不读 self.target_net, 纯浪费 deepcopy)
补丁 B: train_episode 的 sync 改为共享同一份 deepcopy 快照
        (原来 200 个 agent 各自 deepcopy 同一份 c_agent.evaluate_net)
"""
import sys

path = 'DRQNAgent.py'
src = open(path, encoding='utf-8').read()

# ---- 补丁 A ----
old_a = """    def reset(self, mode=None):
        self.mode = mode
        self.target_net = copy.deepcopy(self.evaluate_net).to(self.device)
        self.state_fifo = collections.deque(maxlen=self.sequence)"""
new_a = """    def reset(self, mode=None):
        self.mode = mode
        self.state_fifo = collections.deque(maxlen=self.sequence)"""
assert old_a in src, "patch A: 目标字符串未找到 (可能已被改过)"
src = src.replace(old_a, new_a, 1)

# ---- 补丁 B ----
old_b = """                c_agent.target_net = copy.deepcopy(c_agent.evaluate_net).to(c_agent.device)
                for agent in u_agents:
                    agent.evaluate_net = copy.deepcopy(c_agent.evaluate_net).to(agent.device)
                    agent.lstm_h = None  # 换了权重,h/c也重置"""
new_b = """                c_agent.target_net = copy.deepcopy(c_agent.evaluate_net).to(c_agent.device)
                shared_net = copy.deepcopy(c_agent.evaluate_net).to(c_agent.device)
                for agent in u_agents:
                    agent.evaluate_net = shared_net  # 共享同一份快照, 不再逐个 deepcopy
                    agent.lstm_h = None  # 换了权重,h/c也重置"""
assert old_b in src, "patch B: 目标字符串未找到 (可能已被改过)"
src = src.replace(old_b, new_b, 1)

open(path, 'w', encoding='utf-8').write(src)
print("patch A+B applied ok")
