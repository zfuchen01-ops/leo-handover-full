import os
from math import pi

# 卫星轨道参数
ORBIT_NUM = 6           # 卫星网络默认轨道数量
SAT_PER_ORBIT = 9       # 卫星网络默认每轨道卫星数量
SAT_HEIGHT = 1.0e+6     # 卫星默认高度
LEAN = 85.0/180.0*pi    # 轨道倾角
FIRST_PHI = 0           # 相邻轨道之间第一号卫星的纬度差
THETA = pi/6.0          # 升交点赤经差，轨道与赤道交点的经度差值
CCW = True              # 逆时针旋转方向
CW = False              # 顺时针旋转方向
TYPE_PI = 1             # pi星座，有反向缝
TYPE_2PI = 2            # 2pi星座，无反向缝


# 终端参数
USER_NUM = 100                  # 卫星网络默认终端数量
SAT_BEAM = 64                   # 卫星默认波束数量
USER_HEIGHT = 0                 # 终端默认高度
USER_BEAM = 1                   # 终端默认可接入卫星数量
USER_ELEVATION = pi/180.0*float(os.environ.get("LEO_USER_ELEVATION_DEG", "5"))    # 终端默认仰角
USER_BAND = 500                 # 终端默认接入带宽(Mbps)

# 地球参数
EARTH_RADIUS = 6.370856e+6  # 地球半径，单位米
EARTH_ROTATE = 0.000072918  # 地球自转角速度，弧度制

# 计算所需常数
G_EARTH = 6.67408e-11   # 计算角速度所需
M_EARTH = 5.965e24      # 计算角速度所需

# 网络参数
THRESHOLD = pi/180.0*70 # 极轨断开
METRIC = 1              # 路由条目默认metric
BANDWIDTH = 2000       # 带宽10Gbps，单位Mbps
METRIC = 1              # 路径权值

#切换参数
HANDOVER_COST = float(os.environ.get("LEO_HANDOVER_COST", 0.1))

BANDWIDTH = float(os.environ.get("LEO_ISL_BANDWIDTH", BANDWIDTH))

# Reproduction variant:
# - paper: follow the paper/source formula as closely as possible.
# - calibrated: keep the curve-matching adjustments used during exploration.
CAHS_VARIANT = os.environ.get("LEO_CAHS_VARIANT", "paper").strip().lower()
if CAHS_VARIANT not in ("paper", "calibrated"):
    CAHS_VARIANT = "paper"

DIRECT_PATH_QUALITY = float(os.environ.get(
    "LEO_DIRECT_PATH_QUALITY",
    1.1 if CAHS_VARIANT == "calibrated" else 2.0,
))
# 时隙(仿真时间步)物理时长(秒): 论文/启发式基线(demand_trace/run_experiments)与
# DRQN 训练(train_drqn/DRQNAgent)必须同一口径, 统一为 30 秒(此前 DRQN 误用 50 秒
# 导致与论文基线不可比)。 需其它口径设环境变量 LEO_SLOT_SECONDS。
SLOT_SECONDS = int(os.environ.get("LEO_SLOT_SECONDS", "30"))
AGC_PREDICT_WINDOW = int(os.environ.get("LEO_AGC_PREDICT_WINDOW", 30))
LIGHT_LOAD_USER_THRESHOLD = int(os.environ.get(
    "LEO_LIGHT_LOAD_USER_THRESHOLD",
    450 if CAHS_VARIANT == "calibrated" else 0,
))

# 调试参数
TOPO_LOG_LEVEL = 4        # 是否输出log文件
NET_LOG_LEVEL = 6         #
HO_LOG_LEVEL = 2

# 地面站参数
GATEWAY_ELEVATION = pi/180.0 * 5     # 地面站最小仰角
FEEDLINK_CAPACITY = 10000            # 单天线馈电容量 (Mbps), 光纤回传无瓶颈, 与ISL对齐
ISL_CAPACITY = 2000                  # 星间链路容量 (Mbps), 博士论文表3-1 C_s=2.5Gbps
