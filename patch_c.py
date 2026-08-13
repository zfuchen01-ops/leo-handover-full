#!/usr/bin/env python3
"""补丁 C: RVT 位置缓存 (Get_Satellite_We_Condition memoize).

论证 (bit 级安全):
  - Get_Satellite_We_Condition(time) 是纯函数, 只依赖 (sat 属性, time).
    同一 (sat, time) 输入 -> 逐位相同的输出.
  - 返回的 we_temp 是新对象, 调用方 Calc_Sphere_Elevation 只读 .lat/.lon/.radius.
  - current_time 是 int (time_step=50 累加), t0+dt 是 int, dict key 精确无浮点误差.
  - 同一 episode 内 200 个 agent 共享同一 current_time (Update_Env 在 for 循环后才推进),
    所以 200 用户重复计算同一 (sat, t0+dt) -> 缓存命中.

  profile 佐证: Get_Satellite_We_Condition 350万次调用全部来自 _compute_rvt,
  跨 200 用户冗余, 缓存后降到 ~1.6万次 (256星×61时刻), 减少 ~220 倍.
"""
import sys

path = 'Handover.py'
src = open(path, encoding='utf-8').read()

# ---- 补丁 C1: __init__ 挂载缓存 ----
old_c1 = """        self.source_ho_count = 0
        self._hops_cache = {}  # (sat_id, feeder_id) -> hops, 每集预计算
        self.destination_ho_count = 0"""
new_c1 = """        self.source_ho_count = 0
        self._hops_cache = {}  # (sat_id, feeder_id) -> hops, 每集预计算
        self._rvt_cache = {}   # (sat_id, time) -> Sphere_Position, 每时刻跨用户复用
        self._rvt_cache_time = None
        self.destination_ho_count = 0"""
assert old_c1 in src, "patch C1: __init__ 目标未找到"
src = src.replace(old_c1, new_c1, 1)

# ---- 补丁 C2: _compute_rvt 用缓存 ----
old_c2 = """    def _compute_rvt(self, user:User, sat, max_rvt=600, step=10):
        \"\"\"计算user-sat之间的剩余可见时间(秒). 10s步长, 最大600s.\"\"\"
        t0 = self.topo.current_time
        rvt = 0
        for dt in range(0, max_rvt + step, step):
            sat_pos = sat.Get_Satellite_We_Condition(t0 + dt)
            if Calc_Sphere_Elevation(sat_pos, user.we_pos) >= USER_ELEVATION:
                rvt = dt
            else:
                break
        return rvt"""
new_c2 = """    def _compute_rvt(self, user:User, sat, max_rvt=600, step=10):
        \"\"\"计算user-sat之间的剩余可见时间(秒). 10s步长, 最大600s.\"\"\"
        t0 = self.topo.current_time
        if t0 != self._rvt_cache_time:  # 新时刻: 清空位置缓存 (同ep 200用户共享)
            self._rvt_cache_time = t0
            self._rvt_cache.clear()
        rvt = 0
        for dt in range(0, max_rvt + step, step):
            key = (sat.ID, t0 + dt)
            sat_pos = self._rvt_cache.get(key)
            if sat_pos is None:  # 首次计算后缓存复用 (纯函数, bit级安全)
                sat_pos = sat.Get_Satellite_We_Condition(t0 + dt)
                self._rvt_cache[key] = sat_pos
            if Calc_Sphere_Elevation(sat_pos, user.we_pos) >= USER_ELEVATION:
                rvt = dt
            else:
                break
        return rvt"""
assert old_c2 in src, "patch C2: _compute_rvt 目标未找到"
src = src.replace(old_c2, new_c2, 1)

open(path, 'w', encoding='utf-8').write(src)
print("patch C applied ok")
