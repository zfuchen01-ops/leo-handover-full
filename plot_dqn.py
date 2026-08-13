#!/usr/bin/env python3
"""画DQN训练reward和loss图"""
import csv, sys, os
from collections import defaultdict

reward_file = sys.argv[1] if len(sys.argv) > 1 else 'log/RL/DQN_per_ep.csv'
loss_file = sys.argv[2] if len(sys.argv) > 2 else 'log/RL/DQN_loss.csv'

# Read reward
episodes, rewards = [], []
if os.path.exists(reward_file):
    with open(reward_file) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))

# Read loss
loss_eps, losses = [], []
if os.path.exists(loss_file):
    with open(loss_file) as f:
        for row in csv.DictReader(f):
            loss_eps.append(int(row['episode']))
            losses.append(float(row['loss']))

def smooth(data, window=200):
    if len(data) < window: return data
    import numpy as np
    return np.convolve(data, np.ones(window)/window, mode='valid')

def ascii_plot(data, label, width=60, episodes=None):
    if not data: return
    mn, mx = min(data), max(data)
    if mx == mn: mx = mn + 1
    print(f'\n{label} (range: {mn:.1f} - {mx:.1f})')
    print('─' * (width + 15))
    # Show trend with 400-ep windows
    window = min(400, len(data))
    for i in range(0, len(data), max(1, len(data)//15)):
        chunk = data[i:i+window]
        if len(chunk) < 10: continue
        avg = sum(chunk) / len(chunk)
        bar = '█' * max(1, int((avg - mn) / (mx - mn) * width))
        ep = episodes[i] if episodes else i
        print(f'ep {ep:>5}: {bar} {avg:.1f}')

    # Trend
    if len(data) >= 400:
        first = sum(data[:200])/200
        last = sum(data[-200:])/200
        change = (last - first) / (first + 1e-6) * 100
        direction = '↗ RISING' if change > 5 else '↘ FALLING' if change < -5 else '→ FLAT'
        print(f'\nFirst 200: {first:.1f}  Last 200: {last:.1f}  {direction} ({change:+.1f}%)')

# Plot
ascii_plot(rewards, 'Reward (per episode)', episodes=episodes)
ascii_plot(losses, 'Loss', episodes=loss_eps)

# Checkpoints
print('\nCheckpoint averages (400-ep windows):')
for i in range(0, len(rewards), 400):
    chunk = rewards[i:i+400]
    if len(chunk) >= 100:
        print(f'  ep {episodes[i] if episodes else i}-{episodes[min(i+399,len(episodes)-1)] if episodes else min(i+399,len(rewards)-1)}: avg={sum(chunk)/len(chunk):.1f}')
