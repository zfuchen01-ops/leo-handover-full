#!/usr/bin/env python3
"""补丁 G: 消除 N2N 全量重建中 bit 级无操作的钳制 + 外提属性读取 + 增量去冗余 max.

论证 (bit 级安全, 核心原则: 相同操作数/顺序 → 逐位相同; 只删"数学上恒等于自身"的钳制):

  1) 属性读取外提: lsa.total_band 原在 3 处重复读取, 外提为 tb 读 1 次。
     相同浮点值、相同运算顺序 → 逐位相同。

  2) 钳制去冗余 (关键):
     - load 是 free/total 的连乘, 每项 ∈ [0,1] → load ∈ [0,1],
       故 max(0.0, min(1.0, load)) ≡ load。
     - min_free = min(各 free), free=total-used>=0 (Allocate/Release 保证 used<=total),
       故 min_free>=0, max(0.0,min_free) ≡ min_free; min_total 同理 ≡ min_total。
     - 唯一例外是 else 故障分支把 load/min_free/min_total 置 0(int), 而原钳制
       max(0.0, min(1.0, 0)) 产出 0.0(float)。故故障分支同步改成 0.0 浮点字面量,
       使去钳制后类型/值与原来完全一致 (空路径 maxsize int 不变, 走原样)。
     - Python max/min 在相等时返回首参: max(0.0, 0.0)==0.0, 与直接 0.0 逐位相同。

  3) 增量分支 max(0.0, min(1.0, x)): x=load_rate/old_ratio*new_ratio,
     guard 保证 load_rate>0, old_ratio>0, new_ratio>=0 → x>=0 (不可能为 -0.0,
     因 0.0/total==0.0), 故 max(0.0,...) 恒等于自身, 仅删外层 max。
"""
import sys

path = 'Network.py'
src = open(path, encoding='utf-8').read()

# ---- R1: 外提 lsa.total_band 读取 (全量重建循环 valid 分支) ----
r1_old = (
    "        min_total, min_free, load = maxsize, maxsize, 1.0\n"
    "        for lsa in self._n2n_path[con-1][source-1][dest-1]:\n"
    "            if lsa is not None and lsa.total_band > 0:\n"
    "                free = lsa.total_band - lsa.used_band\n"
    "                load *= free / lsa.total_band\n"
    "                if lsa.total_band < min_total: min_total = lsa.total_band\n"
    "                if free < min_free: min_free = free\n"
)
r1_new = (
    "        min_total, min_free, load = maxsize, maxsize, 1.0\n"
    "        for lsa in self._n2n_path[con-1][source-1][dest-1]:\n"
    "            tb = lsa.total_band if lsa is not None else 0\n"
    "            if tb > 0:\n"
    "                free = tb - lsa.used_band\n"
    "                load *= free / tb\n"
    "                if tb < min_total: min_total = tb\n"
    "                if free < min_free: min_free = free\n"
)
assert r1_old in src, "G-R1 锚点未找到"
assert src.count(r1_old) == 1, "G-R1 锚点不唯一"
src = src.replace(r1_old, r1_new, 1)

# ---- R2: 故障分支 int 0 -> float 0.0 (配合去钳制保持类型一致) ----
r2_old = "                min_free = 0; min_total = 0; load = 0\n"
r2_new = "                min_free = 0.0; min_total = 0.0; load = 0.0\n"
assert src.count(r2_old) == 1, "G-R2 锚点不唯一"
src = src.replace(r2_old, r2_new, 1)

# ---- R3: 全量重建末尾三处无操作钳制去除 ----
r3_old = (
    "        n2n.load_rate = max(0.0, min(1.0, load))\n"
    "        n2n.free_band = max(0.0, min_free)\n"
    "        n2n.total_band = max(0.0, min_total)\n"
)
r3_new = (
    "        n2n.load_rate = load\n"
    "        n2n.free_band = min_free\n"
    "        n2n.total_band = min_total\n"
)
assert r3_old in src, "G-R3 锚点未找到"
assert src.count(r3_old) == 1, "G-R3 锚点不唯一"
src = src.replace(r3_old, r3_new, 1)

# ---- R4: 增量分支去冗余 max(0.0, ...) (两处: Update_N2N_Load_By_LSDB + When_LSA_Change) ----
r4_old = "max(0.0, min(1.0, n2n.load_rate / old_ratio * new_ratio))"
r4_new = "min(1.0, n2n.load_rate / old_ratio * new_ratio)"
assert src.count(r4_old) == 2, f"G-R4 期望 2 处, 实得 {src.count(r4_old)}"
src = src.replace(r4_old, r4_new)

open(path, 'w', encoding='utf-8').write(src)
print("patch G applied ok")
