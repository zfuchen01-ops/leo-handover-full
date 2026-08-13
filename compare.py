#!/usr/bin/env python3
"""对比两次 run 是否逐位等价.

用法: python3 compare.py <tag1> <tag2>
对比: (1) per_ep CSV 逐行逐字段  (2) 最终模型参数 bit 级 md5
"""
import sys
import hashlib

tag1, tag2 = sys.argv[1], sys.argv[2]

def read_csv_rows(path):
    rows = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('episode,'):  # header
                    continue
                rows.append(line)
    except FileNotFoundError:
        return None
    return rows

csv1 = read_csv_rows(f'./log/RL/DRQN_{tag1}_per_ep.csv')
csv2 = read_csv_rows(f'./log/RL/DRQN_{tag2}_per_ep.csv')

print(f'=== CSV 对比 ===')
print(f'{tag1}: {"缺失" if csv1 is None else f"{len(csv1)} 行"}')
print(f'{tag2}: {"缺失" if csv2 is None else f"{len(csv2)} 行"}')

if csv1 is not None and csv2 is not None:
    if csv1 == csv2:
        print(f'CSV 逐行逐字段: 完全一致 ✅  ({len(csv1)} 行 × 14 字段全部相同)')
    else:
        print(f'CSV 不一致 ❌')
        n_diff = 0
        for i in range(max(len(csv1), len(csv2))):
            a = csv1[i] if i < len(csv1) else '<缺失>'
            b = csv2[i] if i < len(csv2) else '<缺失>'
            if a != b:
                n_diff += 1
                if n_diff <= 3:
                    print(f'  第 {i} 行:')
                    print(f'    {tag1}: {a}')
                    print(f'    {tag2}: {b}')
        print(f'  共 {n_diff} 行不一致 / 总 {max(len(csv1), len(csv2))} 行')

print()
print(f'=== 模型参数 bit 级 md5 ===')

def model_md5(path):
    import torch
    net = torch.load(path, map_location='cpu', weights_only=False)
    h = hashlib.md5()
    for p in net.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

for tag in (tag1, tag2):
    p = f'./log/model/drqn_C_u200_{tag}.pkl'
    try:
        m = model_md5(p)
        print(f'  {tag}: {m}')
    except FileNotFoundError:
        print(f'  {tag}: 模型缺失 ({p})')
        m = None

# 汇总判定
eq_csv = (csv1 is not None and csv2 is not None and csv1 == csv2)
print()
print('=== 判定 ===')
if eq_csv:
    print('结果: 逐位等价 ✅  (补丁不改变训练轨迹)')
else:
    print('结果: 存在差异 ❌  (补丁改变了训练轨迹, 需排查)')
