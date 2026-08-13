#!/usr/bin/env python3
"""补丁 F: 外提 Update_N2N_Load_When_LSA_Change 的增量常量 (new_free/old_free/old_ratio/new_ratio).

论证 (bit 级安全):
  - 原 Update_N2N_Load_By_LSDB 增量分支里, new_free=total-used, old_free=new_free+delta,
    old_ratio=old_free/total, new_ratio=new_free/total 这四个量只依赖 changed_lsa 和 delta,
    对同一条链路变化影响到的 ~618 条 N2N 完全相同, 却在每条 N2N 内重复计算。
  - 外提后: 相同操作数、相同顺序, 只是从 N 次降为 1 次 → 结果逐位相同。
  - 内联增量分支 (case 1/2 直接调全量重建 Update_N2N_Load_By_LSDB(delta=0)):
    原代码 case1(new_free<free_band) 先 n2n.free_band=max(0,new_free) 再 fall-through 全量重建,
    但全量重建末尾无条件 n2n.free_band=max(0,min_free) 覆盖该中间值 → 跳过中间赋值逐位等价。
    case2(free_band>=old_free-1e-6) 同理直接全量重建。
    case3(else) 只更新 load_rate 后 return, 内联为继续循环, load_rate 计算表达式逐字不变。
  - 增量条件不满足时走 else 分支, 原样调 Update_N2N_Load_By_LSDB(delta, changed_lsa) 全量重建。
"""
import sys

path = 'Network.py'
src = open(path, encoding='utf-8').read()

old = (
    "    def Update_N2N_Load_When_LSA_Change(self, con, source, dest, delta=0, changed_lsa=None):\n"
    "        index = self.Lookup_LSA_Index(con, source, dest)\n"
    "        if index < 0: return\n"
    "        for i in range(len(self.LSA_N2N[con-1][source-1][index])):\n"
    "            n2n = self.LSA_N2N[con-1][source-1][index][i]\n"
    "            self.Update_N2N_Load_By_LSDB(con, n2n.source, n2n.destination, delta, changed_lsa)\n"
)
new = (
    "    def Update_N2N_Load_When_LSA_Change(self, con, source, dest, delta=0, changed_lsa=None):\n"
    "        index = self.Lookup_LSA_Index(con, source, dest)\n"
    "        if index < 0: return\n"
    "        lst = self.LSA_N2N[con-1][source-1][index]\n"
    "        if delta != 0 and changed_lsa is not None and changed_lsa.total_band > 0:\n"
    "            # 预计算该链路常量 (原代码在每条N2N内重复计算, 相同操作数顺序, bit级相同)\n"
    "            new_free = changed_lsa.total_band - changed_lsa.used_band\n"
    "            old_free = new_free + delta\n"
    "            old_ratio = old_free / changed_lsa.total_band\n"
    "            new_ratio = new_free / changed_lsa.total_band\n"
    "            for n2n in lst:\n"
    "                if old_ratio > 0 and n2n.load_rate > 0:\n"
    "                    n2n.load_rate = max(0.0, min(1.0, n2n.load_rate / old_ratio * new_ratio))\n"
    "                if new_free < n2n.free_band:\n"
    "                    self.Update_N2N_Load_By_LSDB(con, n2n.source, n2n.destination, 0, None)\n"
    "                elif n2n.free_band >= old_free - 1e-6:\n"
    "                    self.Update_N2N_Load_By_LSDB(con, n2n.source, n2n.destination, 0, None)\n"
    "        else:\n"
    "            for n2n in lst:\n"
    "                self.Update_N2N_Load_By_LSDB(con, n2n.source, n2n.destination, delta, changed_lsa)\n"
)
assert old in src, "F: Update_N2N_Load_When_LSA_Change 锚点未找到"
assert src.count(old) == 1, "F: 锚点不唯一"
src = src.replace(old, new, 1)

open(path, 'w', encoding='utf-8').write(src)
print("patch F applied ok")
