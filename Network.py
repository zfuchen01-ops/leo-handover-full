from Defination import *
from Topology import Topology
from math import cos, fabs
from sys import float_info,maxsize
from DataStructure import Stack
from User import User
from Satellite import Satellite
from Position import Calc_Sphere_Distance
from Logger import Logger
import os

#输出文件列表
def _log_path(path):
    return os.devnull if os.environ.get("LEO_QUIET_LOGS", "0") == "1" else path


LCout = open(_log_path("./log/network/NET_Link_Change.log"), 'w')
LSDBout = open(_log_path("./log/network/NET_LSDB.log"), 'w')
SPTout = open(_log_path("./log/network/NET_SPT.log"), 'w')
RTout = open(_log_path("./log/network/NET_RT.log"), 'w')
N2Nout = open(_log_path("./log/network/NET_N2N.log"), 'w')
NSAout = open(_log_path("./log/network/NET_NSA.log"), 'w')
LNout = open(_log_path("./log/network/NET_LSA_N2N.log"), 'w')
NETout = open(_log_path("./log/network/NET_log.log"), 'w')


def Effective_ISL_Bandwidth():
    bandwidth_points = os.environ.get("LEO_ISL_BANDWIDTH_POINTS", "").strip()
    active_user_count = int(os.environ.get("LEO_ACTIVE_USER_COUNT", "0"))
    if bandwidth_points:
        points = []
        for item in bandwidth_points.split(","):
            item = item.strip()
            if not item:
                continue
            users, value = item.split(":", 1)
            points.append((int(users.strip()), float(value.strip())))
        points.sort()
        if active_user_count <= points[0][0]:
            return points[0][1]
        if active_user_count >= points[-1][0]:
            return points[-1][1]
        for (left_users, left_value), (right_users, right_value) in zip(points, points[1:]):
            if left_users <= active_user_count <= right_users:
                span = right_users - left_users
                if span <= 0:
                    return right_value
                ratio = (active_user_count - left_users) / span
                return left_value + ratio * (right_value - left_value)
        return points[-1][1]
    high_load_bandwidth = os.environ.get("LEO_ISL_BANDWIDTH_HIGH_LOAD")
    load_threshold = int(os.environ.get("LEO_ISL_BANDWIDTH_LOAD_THRESHOLD", "0"))
    if high_load_bandwidth is not None and load_threshold > 0 and active_user_count >= load_threshold:
        return float(high_load_bandwidth)
    return ISL_CAPACITY  # 博士论文 2.5Gbps

#Link state advertisement, OSPF中的链路状态通告，用于通告其他节点链路状态，此处代表LSDB中的一个条目（一条链路的数据结构）
class LSA:
    def __init__(self):
        self.isEstablished = False
        self.source_ID = 0
        self.source_interface = 0
        self.destinate_ID = 0
        self.destinate_interface = 0
        self.metric = 0.0
        self.total_band = 0.0
        self.used_band = 0.0
    
    def Config(self, flag, sid, sif, did, dif, metric, total_band=0, used_band=0):
        self.isEstablished = flag
        self.source_ID = sid
        self.source_interface = sif
        self.destinate_ID = did
        self.destinate_interface = dif
        self.metric = metric
        self.total_band = total_band
        #self.used_band = used_band

    def Establish(self, sid, sif, did, dif, metric, total_band=None, used_band=0):
        if total_band is None:
            total_band = Effective_ISL_Bandwidth()
        self.Config(True,sid,sif,did,dif,metric,total_band,used_band)

    def Interrupt(self):
        self.isEstablished = False
        self.total_band = 0.0
        self.used_band = 0.0

#最短路径
class ShortestPath:
    def __init__(self):
        self.isReached = False
        self.pre = 0
        self.destination = 0
        self.distance = 0.0

    def Config(self, flag, pre, destination, distance):
        self.isReached = flag
        self.pre = pre
        self.destination = destination
        self.distance = distance

#思考链表的可能性（已优化，采用map，底层是红黑树.），长度未知采用链表，长度固定采用数组
class NodeForwardingEntry:
    def __init__(self):
        self.isReached = False
        self.destination = 0
        self.nexthop = 0
        self.interface = 0
        self.cost = 0.0
    
    def Config(self, flag, destination, nexthop, interface, cost):
        self.isReached = flag
        self.destination = destination
        self.nexthop = nexthop
        self.interface = interface
        self.cost = cost

#仿照LSA，node state
class NSA:
    def __init__(self):
        self.total_band = 0
        self.used_band = 0

#卫星端到端流量情况
class N2N:
    def __init__(self, source=0, dest=0):
        self.source = source
        self.destination = dest
        self.total_band = 0
        self.free_band = 0
        self.used_band = 0
        self.load_rate = 0.0
        self.busy_node = 0

#网络统计信息
class Net_Statics:
    def __init__(self, cid):
        self.con_id = cid
        self.throughout_limit = 0    #吞吐量上限
        self.throughout = 0      #当前吞吐量
        self.N2N_throughout = 0 #端到端连接流量
        self.load_rate = 0  #网络负载率
        self.band_usage = 0 #当前吞吐量/吞吐量上限
        self.s_size = 0     #用于统计平均，当前为第几条统计数据
        self.average_usage = 0      #端到端连接流量/当前吞吐量

class Network:
    def __init__(self, topo):
        self.topo = Topology()
        self.topo = topo
        self.LSDB = []
        self.SPT = []
        self.forwarding_table = []
        self.node_status = []
        self.N2N_status = []    #卫星端到端网络状态
        self.LSA_N2N = []       #经过每条链路的端到端连接
        self.statics = []
        self._batch_n2n = False  # DRQN训练批量模式: 跳过per-user N2N更新
        self._lsa_lookup = {}   # (con,src,dst) → LSA 快速索引

    def enter_batch_mode(self):
        """进入批量模式: 跳过单用户N2N更新，最后统一刷新."""
        self._batch_n2n = True

    def reset_lsa_used_band(self):
        """每episode重置LSA, 论文每时隙独立决策, 不累积拥塞."""
        for con in range(len(self.LSDB)):
            for src in range(len(self.LSDB[con])):
                for lsa in self.LSDB[con][src]:
                    if lsa.isEstablished:
                        lsa.used_band = 0

    def init_isl_weibull(self, seed=42):
        """初始化每条ISL的Weibull参数 (对齐杨论文Table II)"""
        import random as _random; _random.seed(seed); import math
        modes = [(0.7,5000),(1.0,2000),(1.5,1000),(2.0,600)]
        for con in range(len(self.LSDB)):
            for src in range(len(self.LSDB[con])):
                for lsa in self.LSDB[con][src]:
                    if lsa.isEstablished and lsa.total_band > 0:
                        beta, eta = _random.choice(modes)
                        lsa.weibull_beta = beta
                        lsa.weibull_eta = eta
                        lsa.last_recovery_time = 0  # 上次恢复时刻
                        lsa.is_failed = False
                        if not hasattr(lsa, '_orig_total_band'):
                            lsa._orig_total_band = lsa.total_band

    def apply_isl_failures(self, current_time, delta_T=SLOT_SECONDS):
        """Weibull故障模型 (对齐杨): 断链和恢复都由同一Weibull公式驱动
        S(τ)=exp(-(τ/η)^β)
        正常链路: P(fail)=1-S(τ+∆T)/S(τ) = 在∆T内不断的条件概率
        故障链路: P(recover)=1-exp(-(τ_failed/η_r)^β)  恢复概率随时间增长
        其中η_r=η/4 (恢复比故障快4倍, 对齐杨: 链路恢复后τ重置)"""
        import random as _random; import math
        n_fail, n_recover = 0, 0
        for con in range(len(self.LSDB)):
            for src in range(len(self.LSDB[con])):
                for lsa in self.LSDB[con][src]:
                    if not hasattr(lsa, 'weibull_beta'):
                        continue
                    beta, eta = lsa.weibull_beta, lsa.weibull_eta
                    if not lsa.is_failed:
                        tau = max(0.1, current_time - lsa.last_recovery_time)
                        surv_now = math.exp(-(tau/eta)**beta)
                        surv_next = math.exp(-((tau+delta_T)/eta)**beta)
                        p_fail = 1.0 - surv_next/surv_now if surv_now>0 else 0
                        if _random.random() < p_fail:
                            lsa.total_band = 0; lsa.isEstablished = False
                            lsa.is_failed = True
                            lsa._fail_time = current_time; n_fail += 1
                    else:
                        tau_failed = current_time - getattr(lsa, '_fail_time', current_time)
                        eta_r = eta / 4  # 恢复速度是故障的4倍
                        p_recover = 1.0 - math.exp(-(max(1,tau_failed)/eta_r)**beta)
                        if _random.random() < p_recover:
                            lsa.total_band = lsa._orig_total_band
                            lsa.isEstablished = True
                            lsa.is_failed = False
                            lsa.last_recovery_time = current_time; n_recover += 1
        if n_fail > 0 or n_recover > 0:
            self.Update_N2N_Load_By_LSDB_All()

    def exit_batch_mode(self):
        """退出批量模式: 重置LSA+全量重建N2N.
        LSA跨时隙不可靠(SPT随拓扑变化),每episode重置保证一致性."""
        self._batch_n2n = False
        for con in range(len(self.LSDB)):
            for src in range(len(self.LSDB[con])):
                for lsa in self.LSDB[con][src]:
                    if lsa.isEstablished:
                        lsa.used_band = 0
        self.Update_N2N_Load_By_LSDB_All()

    
    def Initial_Network(self):
        con_num = len(self.topo.constellation)
        self.LSDB = [[] for _ in range(con_num)]
        self.SPT = [[] for _ in range(con_num)]
        self.forwarding_table = [[] for _ in range(con_num)]
        self.node_status = [[] for _ in range(con_num)]
        self.N2N_status = [[] for _ in range(con_num)]
        self.LSA_N2N = [[] for _ in range(con_num)]
        self._lsa_n2n_built = False  # LSA_N2N反向映射是否已构建 (SPT静态则跳过重复重建)
        self._n2n_path_built = False  # N2N路径缓存是否已构建 (SPT/LSA索引变则失效)       
        self.statics = [Net_Statics(i+1) for i in range(con_num)]
        for i in range(con_num):
            sat_num = self.topo.constellation[i].orbit_num*self.topo.constellation[i].sat_per_orbit
            self.LSDB[i] = [[] for _ in range(sat_num)]
            self.SPT[i] = [[] for _ in range(sat_num)]
            self.forwarding_table[i] = [[] for _ in range(sat_num)]
            self.node_status[i] = [NSA() for _ in range(sat_num)]
            self.N2N_status[i] = [[] for _ in range(sat_num)]
            self.LSA_N2N[i] = [[] for _ in range(sat_num)]
            for j in range(sat_num):
                self.LSDB[i][j] = [LSA() for _ in range(4)]
                self.SPT[i][j] = [ShortestPath() for _ in range(sat_num)]
                self.forwarding_table[i][j] = [NodeForwardingEntry() for _ in range(sat_num)]
                self.N2N_status[i][j] = [N2N(j+1,k+1) for k in range(sat_num)]
                self.LSA_N2N[i][j] = [[] for _ in range(4)]
            
    def Running_Network(self, start, end, step):
        for t in range(start,end,step):
            self.topo.Set_Time(t)
            self.topo.Update_Topology_Status()
            self.Update_LSDB()
            self.Dijkstra()
            self.Generate_Forwardingtable_By_Node()
            #self.Update_LSDB_By_N2N()
            #self.Update_N2N_Load_By_LSDB()
            #self.Link_LSDB_With_N2N()
            #self.Update_NSA_Band()

    #链路状态数据库更新，只更新链路对应关系
    def Update_LSDB(self):
        for i in range(len(self.topo.constellation)):
            temp_con = self.topo.constellation[i]
            for j in range(temp_con.orbit_num):
                temp_orbit = temp_con.orbit_sat[j]
                for k in range(temp_orbit.sat_num):
                    temp_sat = temp_orbit.sat_in_orbit[k]
                    self.LSDB[i][j*temp_con.sat_per_orbit+k][0].Config(True,temp_sat.ID,1,
                        temp_sat.ID+temp_orbit.sat_num-1 if temp_sat.ID%temp_orbit.sat_num==1 else temp_sat.ID-1,2,METRIC)
                    self.LSDB[i][j*temp_con.sat_per_orbit+k][1].Config(True,temp_sat.ID,2,
                        temp_sat.ID-temp_orbit.sat_num+1 if temp_sat.ID%temp_orbit.sat_num==0 else temp_sat.ID+1,1,METRIC)
                    if temp_con.orbit_num>1:
                        if fabs(temp_sat.we_pos.lat)<=THRESHOLD:
                            if temp_orbit.orbit_id==1:
                                if temp_con.type==TYPE_2PI:
                                    index = temp_con.sat_per_orbit*(temp_con.orbit_num-1)
                                    if fabs(temp_con.orbit_sat[temp_con.orbit_num-1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD:
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Config(True,temp_sat.ID, 
                                            3 if temp_sat.isNorth else 4, temp_sat.ID+index, 4 if temp_sat.isNorth else 3, METRIC)                               
                                if fabs(temp_con.orbit_sat[j+1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD:
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Config(True,temp_sat.ID, 
                                        4 if temp_sat.isNorth else 3, temp_sat.ID+temp_orbit.sat_num, 3 if temp_sat.isNorth else 4, METRIC)
                            elif temp_orbit.orbit_id<temp_con.orbit_num:
                                if fabs(temp_con.orbit_sat[j-1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD:
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Config(True,temp_sat.ID, 
                                        3 if temp_sat.isNorth else 4, temp_sat.ID-temp_orbit.sat_num, 4 if temp_sat.isNorth else 3, METRIC)
                                if fabs(temp_con.orbit_sat[j+1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD:
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Config(True,temp_sat.ID, 
                                        4 if temp_sat.isNorth else 3, temp_sat.ID+temp_orbit.sat_num, 3 if temp_sat.isNorth else 4, METRIC)
                            else:                             
                                if fabs(temp_con.orbit_sat[j-1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD:
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Config(True,temp_sat.ID, 
                                        3 if temp_sat.isNorth else 4, temp_sat.ID-temp_orbit.sat_num, 4 if temp_sat.isNorth else 3, METRIC)
                                if temp_con.type==TYPE_2PI:
                                    index = temp_con.sat_per_orbit*(temp_con.orbit_num-1)
                                    if fabs(temp_con.orbit_sat[0].sat_in_orbit[k].we_pos.lat)<=THRESHOLD:
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Config(True,temp_sat.ID, 
                                            4 if temp_sat.isNorth else 3, temp_sat.ID-index, 3 if temp_sat.isNorth else 4, METRIC)


    #链路状态数据库更新，包含网络带宽重分配、重路由功能
    def Update_LSDB_Netmode(self):
        LCout.write("Time:%d\n"%self.topo.current_time)
        #每个星座对应一个LSDB，存储顺序：同轨链路（1下，1上，2下，2上。。。），异轨链路（1左，1右，2左，2右。。。）
        for i in range(len(self.topo.constellation)):
            
            temp_con = self.topo.constellation[i]
            reroute = {}
            for num in range(temp_con.orbit_num*temp_con.sat_per_orbit):
                reroute[num+1] = set()
            for j in range(temp_con.orbit_num):
                temp_orbit = temp_con.orbit_sat[j]
                for k in range(temp_orbit.sat_num):
                    temp_sat = temp_orbit.sat_in_orbit[k]
                    #初始时刻构建同轨链路
                    if self.LSDB[i][j*temp_con.sat_per_orbit+k][0].isEstablished==False:
                        self.LSDB[i][j*temp_con.sat_per_orbit+k][0].Establish(temp_sat.ID,1, \
                            temp_sat.ID+temp_orbit.sat_num-1 if temp_sat.ID%temp_orbit.sat_num==1 else temp_sat.ID-1,2,METRIC)
                        LCout.write(" Links between Sat:%d and Sat:%d Establish\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][0].destinate_ID))
                    if self.LSDB[i][j*temp_con.sat_per_orbit+k][1].isEstablished==False:
                        self.LSDB[i][j*temp_con.sat_per_orbit+k][1].Establish(temp_sat.ID,2,
                            temp_sat.ID-temp_orbit.sat_num+1 if temp_sat.ID%temp_orbit.sat_num==0 else temp_sat.ID+1,1,METRIC)
                        LCout.write(" Links between Sat:%d and Sat:%d Establish\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][1].destinate_ID))
                    if temp_con.orbit_num>1:
                        if fabs(temp_sat.we_pos.lat)<=THRESHOLD:
                            if temp_orbit.orbit_id==1:
                                if temp_con.type==TYPE_2PI:
                                    index = temp_con.sat_per_orbit*(temp_con.orbit_num-1)
                                    if fabs(temp_con.orbit_sat[temp_con.orbit_num-1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD and \
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][2].isEstablished==False:
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Establish(temp_sat.ID, 3 if temp_sat.isNorth else 4, \
                                            temp_sat.ID+index, 4 if temp_sat.isNorth else 3, METRIC)
                                        LCout.write(" Links between Sat:%d and Sat:%d Establish\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][2].destinate_ID))
                                    elif fabs(temp_con.orbit_sat[temp_con.orbit_num-1].sat_in_orbit[k].we_pos.lat)>THRESHOLD and \
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][2].isEstablished:
                                        self.Extract_N2N_From_LSA(temp_con.ID,temp_sat.ID,temp_sat.ID+index,reroute)
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Interrupt()
                                        LCout.write(" Links between Sat:%d and Sat:%d Interrupt\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][2].destinate_ID))
                                if fabs(temp_con.orbit_sat[j+1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD and \
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].isEstablished==False:
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Establish(temp_sat.ID, 4 if temp_sat.isNorth else 3, 
                                        temp_sat.ID+temp_orbit.sat_num, 3 if temp_sat.isNorth else 4, METRIC)
                                    LCout.write(" Links between Sat:%d and Sat:%d Establish\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][3].destinate_ID))
                                elif fabs(temp_con.orbit_sat[j+1].sat_in_orbit[k].we_pos.lat)>THRESHOLD and \
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].isEstablished:
                                    self.Extract_N2N_From_LSA(temp_con.ID,temp_sat.ID,temp_sat.ID+temp_orbit.sat_num,reroute)
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Interrupt()
                                    LCout.write(" Links between Sat:%d and Sat:%d Interrupt\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][3].destinate_ID))
                            elif temp_orbit.orbit_id<temp_con.orbit_num:
                                if fabs(temp_con.orbit_sat[j-1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD and \
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].isEstablished==False:
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Establish(temp_sat.ID, 3 if temp_sat.isNorth else 4, 
                                        temp_sat.ID-temp_orbit.sat_num, 4 if temp_sat.isNorth else 3, METRIC)
                                    LCout.write(" Links between Sat:%d and Sat:%d Establish\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][2].destinate_ID))
                                elif fabs(temp_con.orbit_sat[j-1].sat_in_orbit[k].we_pos.lat)>THRESHOLD and \
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].isEstablished:
                                    self.Extract_N2N_From_LSA(temp_con.ID,temp_sat.ID,temp_sat.ID-temp_orbit.sat_num,reroute)
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Interrupt()
                                    LCout.write(" Links between Sat:%d and Sat:%d Interrupt\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][2].destinate_ID))
                                if fabs(temp_con.orbit_sat[j+1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD and \
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].isEstablished==False:
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Establish(temp_sat.ID, 4 if temp_sat.isNorth else 3, 
                                        temp_sat.ID+temp_orbit.sat_num, 3 if temp_sat.isNorth else 4, METRIC)   
                                    LCout.write(" Links between Sat:%d and Sat:%d Establish\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][3].destinate_ID))
                                elif fabs(temp_con.orbit_sat[j+1].sat_in_orbit[k].we_pos.lat)>THRESHOLD and \
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].isEstablished:
                                    self.Extract_N2N_From_LSA(temp_con.ID,temp_sat.ID,temp_sat.ID+temp_orbit.sat_num,reroute)
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Interrupt()
                                    LCout.write(" Links between Sat:%d and Sat:%d Interrupt\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][3].destinate_ID))
                            else:                             
                                if fabs(temp_con.orbit_sat[j-1].sat_in_orbit[k].we_pos.lat)<=THRESHOLD and \
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].isEstablished==False:
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Establish(temp_sat.ID, 3 if temp_sat.isNorth else 4, 
                                        temp_sat.ID-temp_orbit.sat_num, 4 if temp_sat.isNorth else 3, METRIC)
                                    LCout.write(" Links between Sat:%d and Sat:%d Establish\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][2].destinate_ID))
                                elif fabs(temp_con.orbit_sat[j-1].sat_in_orbit[k].we_pos.lat)>THRESHOLD and \
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].isEstablished:
                                    self.Extract_N2N_From_LSA(temp_con.ID,temp_sat.ID,temp_sat.ID-temp_orbit.sat_num,reroute)
                                    self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Interrupt()
                                    LCout.write(" Links between Sat:%d and Sat:%d Interrupt\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][2].destinate_ID))
                                if temp_con.type==TYPE_2PI:
                                    index = temp_con.sat_per_orbit*(temp_con.orbit_num-1)
                                    if fabs(temp_con.orbit_sat[0].sat_in_orbit[k].we_pos.lat)<=THRESHOLD and \
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][3].isEstablished==False:
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Establish(temp_sat.ID, 4 if temp_sat.isNorth else 3, 
                                            temp_sat.ID-index, 3 if temp_sat.isNorth else 4, METRIC)
                                        LCout.write(" Links between Sat:%d and Sat:%d Establish\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][3].destinate_ID))
                                    elif fabs(temp_con.orbit_sat[0].sat_in_orbit[k].we_pos.lat)>THRESHOLD and \
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][3].isEstablished:
                                        self.Extract_N2N_From_LSA(temp_con.ID,temp_sat.ID,temp_sat.ID-index,reroute)
                                        self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Interrupt()
                                        LCout.write(" Links between Sat:%d and Sat:%d Interrupt\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][3].destinate_ID))

                        else:
                            if self.LSDB[i][j*temp_con.sat_per_orbit+k][2].isEstablished:
                                self.Extract_N2N_From_LSA(temp_con.ID,temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][2].destinate_ID,reroute)
                                self.LSDB[i][j*temp_con.sat_per_orbit+k][2].Interrupt()
                                LCout.write(" Links between Sat:%d and Sat:%d Interrupt\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][2].destinate_ID))
                            if self.LSDB[i][j*temp_con.sat_per_orbit+k][3].isEstablished:
                                self.Extract_N2N_From_LSA(temp_con.ID,temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][3].destinate_ID,reroute)
                                self.LSDB[i][j*temp_con.sat_per_orbit+k][3].Interrupt()
                                LCout.write(" Links between Sat:%d and Sat:%d Interrupt\n"%(temp_sat.ID,self.LSDB[i][j*temp_con.sat_per_orbit+k][3].destinate_ID))
            self.Rerouting(temp_con.ID,reroute)
            reroute.clear()
        self._rebuild_lsa_cache()

    #从链路中提取受影响的N2N
    def Extract_N2N_From_LSA(self, con, source, dest, reroute):
        temp = N2N()
        index = self.Lookup_LSA_Index(con,source,dest)
        for i in range(len(self.LSA_N2N[con-1][source-1][index])):
            temp = self.LSA_N2N[con-1][source-1][index][i]
            reroute[temp.source].add(temp.destination)
    
    #对所有受影响的端到端连接进行重路由，首先释放资源，禁用路由表，重新计算受影响节点的最短路径树，依据N2N进行资源重新分配，然后计算路由表
    def Rerouting(self, con, reroute):
        for source in reroute:
            for dest in reroute[source]:
                if self.N2N_status[con-1][source-1][dest-1].used_band>0:
                    if self.Release_LSA_Band(con,source,dest,self.N2N_status[con-1][source-1][dest-1].used_band)==False:
                        print("Problems when releasing LSA band!!!")
                        exit(1)
                self.SPT[con-1][source-1][dest-1].isReached = False
                self.forwarding_table[con-1][source-1][dest-1].isReached = False

        for source in reroute:
            #计算受影响节点的最短路径，通过在原来的最短路径树中断开指定链路，为剩余的不可达的节点重新算路
            #按照N2N带宽进行排序，重新选路（目前算法会出现重路由之后拥塞问题）
            
            self.Dijkstra_Partially(con,source,reroute[source])
            for dest in reroute[source]:
                if self.N2N_status[con-1][source-1][dest-1].used_band>0:
                    if self.Allocate_LSA_Band(con,source,dest,self.N2N_status[con-1][source-1][dest-1].used_band)==False:
                        print("Problems when releasing LSA band!!!")
                        exit(1)
                self.Generate_Forwardingtable_By_Node(con, source, dest)

    #链路断开后，仅计算部分受影响的最短路径
    def Dijkstra_Partially(self, con, start, reroute):
        self._lsa_n2n_built = False  # 部分SPT将重建, 反向映射失效
        self._n2n_path_built = False
        length = len(self.LSDB[con-1])
        arcs = [float_info.max]*length
        visited = [False]*length
        arcs[start-1] = 0

        target = -1
        index = -1
        for i in range(length):
            min_arc = float_info.max
            #从未访问列表里找到距离最小的点，最开始是start
            for j in range(length):
                if visited[j]==False:
                    if arcs[j]<min_arc:
                        index = j+1
                        min_arc = arcs[j]
            visited[index-1]=True

            for j in range(4):
                if self.LSDB[con-1][index-1][j].isEstablished==False:
                    continue
                target = self.LSDB[con-1][index-1][j].destinate_ID
                if visited[target-1]==False:
                    if target in reroute and min_arc+self.LSDB[con-1][index-1][j].metric<arcs[target-1]:    #只为受影响节点对计算重路由
                        arcs[target-1] = min_arc+self.LSDB[con-1][index-1][j].metric
                        if self.SPT[con-1][start-1][target-1].isReached and self.SPT[con-1][start-1][target-1].distance>arcs[target-1]:
                            self.SPT[con-1][start-1][target-1].distance = arcs[target-1]
                            self.SPT[con-1][start-1][target-1].pre = index
                        elif self.SPT[con-1][start-1][target-1].isReached==False:
                            self.SPT[con-1][start-1][target-1].Config(True,index,target,arcs[target-1])

                    elif target not in reroute:
                        arcs[target-1] = self.SPT[con-1][start-1][target-1].distance

    #计算所有节点为根节点的最短路径树
    def Dijkstra_All(self):
        self._lsa_n2n_built = False  # SPT将重建, 反向映射失效
        self._n2n_path_built = False
        for i in range(len(self.LSDB)):
            for j in range(len(self.LSDB[i])):
                self.Dijkstra(i+1,j+1)

    #计算start为根节点的最短路径树 (BFS: O(V+E), 卫星图每节点4条边)
    def Dijkstra(self, con, start):
        length = len(self.LSDB[con-1])
        # BFS: 无权图, 所有边metric=1
        visited = [False] * length
        from collections import deque
        q = deque()
        visited[start-1] = True
        self.SPT[con-1][start-1][start-1].Config(True, start, start, 0)
        q.append(start)
        while q:
            node = q.popleft()
            for j in range(4):
                if not self.LSDB[con-1][node-1][j].isEstablished:
                    continue
                neighbor = self.LSDB[con-1][node-1][j].destinate_ID
                if not visited[neighbor-1]:
                    visited[neighbor-1] = True
                    self.SPT[con-1][start-1][neighbor-1].Config(True, node, neighbor, self.SPT[con-1][start-1][node-1].distance + 1)
                    q.append(neighbor)

    #以卫星节点编号代替目的网段进行寻址
    def Generate_Forwardingtable_By_AllNode(self):
        for i in range(len(self.LSDB)):
            for j in range(len(self.LSDB[i])):
                for k in range(len(self.LSDB[i])):
                    self.Generate_Forwardingtable_By_Node(i+1,j+1,k+1)

    #以卫星节点编号代替目的网段进行寻址
    def Generate_Forwardingtable_By_Node(self, con, source, destination):
        res = LSA()
        pre = -1
        dest = -1
        if source!=destination:
            if self.SPT[con-1][source-1][destination-1].isReached==False:
                self.forwarding_table[con-1][source-1][destination-1].isReached = False
            else:
                pre = self.SPT[con-1][source-1][destination-1].pre
                dest = self.SPT[con-1][source-1][destination-1].destination
                while pre!=source:
                    dest = self.SPT[con-1][source-1][pre-1].destination
                    pre = self.SPT[con-1][source-1][pre-1].pre
                res = self.Lookup_LSA(con,pre,dest)
                if res==None:
                    print("Problem in Generate_Forwardingtable_By_Node")
                    exit(1)
                self.forwarding_table[con-1][source-1][destination-1].Config(True,destination,dest,res.source_interface,
                    self.SPT[con-1][source-1][destination-1].distance)
        else:
            self.forwarding_table[con-1][source-1][destination-1].Config(True,source,source,0,0)

    def _rebuild_lsa_cache(self):
        """O(1)字典索引: (con,src,dst) → (index, LSA)."""
        self._lsa_n2n_built = False  # LSA索引变化, 反向映射失效
        self._n2n_path_built = False
        self._lsa_lookup.clear()
        for con in range(1, len(self.LSDB)+1):
            for src in range(1, len(self.LSDB[con-1])+1):
                for i, lsa in enumerate(self.LSDB[con-1][src-1]):
                    if lsa.isEstablished:
                        self._lsa_lookup[(con, src, lsa.destinate_ID)] = (i, lsa)

    #检索链路，返回指针 (O(1)字典)
    def Lookup_LSA(self, con, source, destination):
        key = (con, source, destination)
        if key in self._lsa_lookup:
            return self._lsa_lookup[key][1]
        return None

    #检索链路对应节点的几号端口，返回端口号 (O(1)字典)
    def Lookup_LSA_Index(self, con, source, destination):
        key = (con, source, destination)
        if key in self._lsa_lookup:
            return self._lsa_lookup[key][0]
        return -1

    #用户与卫星建立连接，下载和上传连接同时建立，如果目的地没有入网，那么发起寻呼
    # 论文模式: 有gateway时流量走gateway feeder卫星, 否则走配对用户卫星
    def User_Connect_Satellite(self, user:User, sat:Satellite, band):
        con = sat.con_id
        source = sat.ID
        # 确定ISL目的地: 选跳数最少的gateway feeder卫星
        gw = getattr(user, 'assigned_gateway', None)
        feeder_sat = None
        if gw is not None:
            best_h = 999
            for s in gw.connected_sat:
                if s is not None:
                    # 计算从当前卫星到feeder的跳数 (Dijkstra已跑,直接用SPT)
                    if s == sat:
                        h = 0
                    elif self.SPT[con-1][sat.ID-1][s.ID-1].isReached:
                        h = 999  # 简化, 选非零跳数最近的
                        # 遍历路径算跳数
                        curr = s.ID
                        h = 0
                        while curr != sat.ID and h < 100:
                            nxt = self.SPT[con-1][sat.ID-1][curr-1].pre
                            if nxt <= 0: break
                            curr = nxt; h += 1
                    else:
                        h = 999
                    if h < best_h:
                        best_h = h
                        feeder_sat = s
        if feeder_sat is None and gw is not None:
            feeder_sat = gw.connected_sat[0]  # fallback
        # 论文gateway场景: 无配对用户, 直接建立用户→gateway feeder的端到端连接
        if len(user.user_to_connect_to) == 0 and len(user.user_to_connect_by) == 0 and feeder_sat is not None:
            dest = feeder_sat.ID
            if self.User_Connect_Satellite_Band(con, source, dest, user, user, band):
                pass  # allocate_band[user] = band already set
        #建立上传连接
        for u in user.user_to_connect_to:
            #已经建立端到端连接
            if u in user.user_connecting_to:
                continue
            temp_user = u
            if feeder_sat is not None:
                dest = feeder_sat.ID  # 论文:用户→gateway
            else:
                if temp_user.sat_connected==None:
                    if self.Paging_User(temp_user)==False:
                        continue
                dest = temp_user.sat_connected.ID
            user.user_to_connect_to[u] = band
            u.user_to_connect_by[user] = band
            #如果没有可用的带宽，不建立端到端连接，但是终端与卫星的连接关系保持，不算切换失败
            if self.User_Connect_Satellite_Band(con,source,dest,user,u,user.user_to_connect_to[u])==False:
                continue
            user.user_connecting_to.add(temp_user)
            temp_user.user_connecting_by.add(user)
        #建立下载连接
        for u in user.user_to_connect_by:
            if u in user.user_connecting_by:
                continue
            temp_user = u
            if feeder_sat is not None:
                dest = feeder_sat.ID  # 论文:gateway→用户
            else:
                if temp_user.sat_connected==None:
                    if self.Paging_User(temp_user)==False:
                        continue
                dest = temp_user.sat_connected.ID
            if self.User_Connect_Satellite_Band(con,dest,source,u,user,user.user_to_connect_by[u])==False:
                continue
            user.user_connecting_by.add(temp_user)
            temp_user.user_connecting_to.add(user)
        return True if (len(user.user_connecting_to)>0 or len(user.user_connecting_by)>0 or len(user.allocate_band)>0) else False #只要有端到端连接或gateway分配, 就算接入成功

    #分配带宽
    # 尽力而为(FCFS): 先到先得, 抢不到=0
    def User_Connect_Satellite_Band(self, con, source, dest, s_user, d_user, band):
        if self.N2N_status[con-1][source-1][dest-1].free_band==0:
            s_user.allocate_band[d_user] = 0
            return True
        # 尽力而为: cap到可用带宽, 拿多少算多少
        band = min(band, self.N2N_status[con-1][source-1][dest-1].free_band) if source!=dest else band
        if self.Allocate_LSA_Band(con,source,dest,band):
            s_user.allocate_band[d_user] = band
            s_user.allocate_dest[d_user] = dest
            self.N2N_status[con-1][source-1][dest-1].free_band -= band
            self.N2N_status[con-1][source-1][dest-1].used_band += band
            if not self._batch_n2n:
                self.Update_When_N2N_Change(con, source, dest, delta=+band)
        return True  # 尽力而为: 不因LSA满而失败

    #分配链路带宽 (尽力而为: cap不rollback, 拿完为止)
    def Allocate_LSA_Band(self, con, source, dest, band):
        if source==dest or band==0: return True
        pre = self.SPT[con-1][source-1][dest-1].pre
        next_n = self.SPT[con-1][source-1][dest-1].destination
        while pre!=source or next_n!=source:
            res = self.Lookup_LSA(con,pre,next_n)
            actual = min(band, res.total_band - res.used_band)
            res.used_band += actual
            next_n = self.SPT[con-1][source-1][pre-1].destination
            pre = self.SPT[con-1][source-1][pre-1].pre
        return True

    #寻呼，与距离最近的卫星建立连接
    def Paging_User(self, user:User):
        min_dis = float_info.max
        target = None
        for sat in user.sat_covered:
            if sat.beam-len(sat.user_connected)>0:
                temp = Calc_Sphere_Distance(user.we_pos,sat.we_pos)
                if temp<min_dis:
                    min_dis = temp
                    target = sat
        if target==None:
            return False
        user.sat_connected=target
        target.user_connected.add(user)
    
    #用户与卫星断开连接，需要释放带宽，更新用户连接关系
    def User_Disconnect_Satellite(self, user:User, sat:Satellite):
        con = sat.con_id
        source = sat.ID
        # 确定ISL目的地: Connect时走的gateway feeder, Disconnect也必须一致
        gw = getattr(user, 'assigned_gateway', None)
        feeder_sat = gw.connected_sat[0] if (gw is not None and len(gw.connected_sat) > 0 and gw.connected_sat[0] is not None) else None
        # gateway模式: 直连feeder, connecting_to/by为空, 直接释放
        if len(user.user_connecting_to) == 0 and len(user.user_connecting_by) == 0:
            if len(user.allocate_band) > 0:
                for d_user, band in list(user.allocate_band.items()):
                    saved_dest = user.allocate_dest.get(d_user, source)
                    self.User_Disconnect_Satellite_Band(con, source, saved_dest, band)
                    user.allocate_band.pop(d_user, None)
                    user.allocate_dest.pop(d_user, None)
            return True

        # 配对模式: 清理用户级状态, 释放LSA带宽(用分配时记录的dest卫星)
        for u in list(user.user_connecting_to):
            saved_dest = user.allocate_dest.get(u, source)
            self.User_Disconnect_Satellite_Band(con, source, saved_dest, user.allocate_band.get(u, 0))
            u.user_connecting_by.discard(user)
            user.allocate_band.pop(u, None)
            user.allocate_dest.pop(u, None)
        user.user_connecting_to.clear()
        for u in list(user.user_connecting_by):
            saved_dest = u.allocate_dest.get(user, source)
            self.User_Disconnect_Satellite_Band(con, saved_dest, source, u.allocate_band.get(user, 0))
            u.user_connecting_to.discard(user)
            u.allocate_band.pop(user, None)
            u.allocate_dest.pop(user, None)
        user.user_connecting_by.clear()
        return True

    #用户与卫星断开连接，需要释放带宽
    def User_Disconnect_Satellite_Band(self, con, source, dest, band):
        if band+self.N2N_status[con-1][source-1][dest-1].free_band>self.N2N_status[con-1][source-1][dest-1].total_band:
            # print("Problems when releasing LSA band!!!")  # silenced: congestion is normal
            return False
        if self.Release_LSA_Band(con,source,dest,band):
            self.N2N_status[con-1][source-1][dest-1].free_band += band
            self.N2N_status[con-1][source-1][dest-1].used_band -= band
            if not self._batch_n2n:
                self.Update_When_N2N_Change(con, source, dest, delta=-band)
        return True  # 尽力而为: 不因LSA满而失败

    #释放N2N经过的所有链路带宽
    def Release_LSA_Band(self, con, source, dest, band):
        if source==dest or band==0:
            return True
        res = LSA()
        pre = self.SPT[con-1][source-1][dest-1].pre
        next_n = self.SPT[con-1][source-1][dest-1].destination
        while pre!=source or next_n!=source:
            res = self.Lookup_LSA(con,pre,next_n)
            if res == None:
                next_n = self.SPT[con-1][source-1][pre-1].destination
                pre = self.SPT[con-1][source-1][pre-1].pre
            elif res!=None and res.used_band-band<0:
                end = next_n
                pre = self.SPT[con-1][source-1][dest-1].pre
                next_n = self.SPT[con-1][source-1][dest-1].destination
                while next_n!=end:
                    res.used_band += band
                    next_n = self.SPT[con-1][source-1][pre-1].destination
                    pre = self.SPT[con-1][source-1][pre-1].pre
                return False
            else:
                res.used_band -= band
                next_n = self.SPT[con-1][source-1][pre-1].destination
                pre = self.SPT[con-1][source-1][pre-1].pre
        return True

    #更新节点的带宽信息
    def Update_NSA_Band_All(self):
        for i in range(len(self.LSDB)):
            for j in range(len(self.LSDB[i])):
                self.Update_NSA_Band(i+1,j+1)

    #更新节点的带宽信息
    def Update_NSA_Band(self, con, node):
        self.node_status[con-1][node-1].used_band = 0
        self.node_status[con-1][node-1].total_band = 0
        for i in range(len(self.LSDB[con-1][node-1])):
            if self.LSDB[con-1][node-1][i].isEstablished:
                self.node_status[con-1][node-1].used_band += self.LSDB[con-1][node-1][i].used_band
                self.node_status[con-1][node-1].total_band += self.LSDB[con-1][node-1][i].total_band

    #N2N变化导致其经过的链路带宽发生变化，需要调用相应函数进行更新
    def Update_When_N2N_Change(self, con, source, dest, delta=0):
        pre = self.SPT[con-1][source-1][dest-1].pre
        next_n = self.SPT[con-1][source-1][dest-1].destination
        while pre!=source or next_n!=source:
            lsa = self.Lookup_LSA(con, pre, next_n)
            self.Update_N2N_Load_When_LSA_Change(con, pre, next_n, delta, lsa)
            next_n = self.SPT[con-1][source-1][pre-1].destination
            pre = self.SPT[con-1][source-1][pre-1].pre

    #当某条链路上的带宽发生变化，需要遍历这条链路每一跳影响到的N2N
    def Update_N2N_Load_When_LSA_Change(self, con, source, dest, delta=0, changed_lsa=None):
        index = self.Lookup_LSA_Index(con, source, dest)
        if index < 0: return
        lst = self.LSA_N2N[con-1][source-1][index]
        if delta != 0 and changed_lsa is not None and changed_lsa.total_band > 0:
            # 预计算该链路常量 (原代码在每条N2N内重复计算, 相同操作数顺序, bit级相同)
            new_free = changed_lsa.total_band - changed_lsa.used_band
            old_free = new_free + delta
            old_ratio = old_free / changed_lsa.total_band
            new_ratio = new_free / changed_lsa.total_band
            for n2n in lst:
                if old_ratio > 0 and n2n.load_rate > 0:
                    n2n.load_rate = min(1.0, n2n.load_rate / old_ratio * new_ratio)
                if new_free < n2n.free_band:
                    self.Update_N2N_Load_By_LSDB(con, n2n.source, n2n.destination, 0, None)
                elif n2n.free_band >= old_free - 1e-6:
                    self.Update_N2N_Load_By_LSDB(con, n2n.source, n2n.destination, 0, None)
        else:
            for n2n in lst:
                self.Update_N2N_Load_By_LSDB(con, n2n.source, n2n.destination, delta, changed_lsa)

    #从LSDB中遍历，更新N2N的带宽及负载率 (增量: delta!=0时O(1), 仅瓶颈变更时全量)
    def _build_n2n_path_cache(self):
        """预计算每条 N2N 最短路径的 LSA 引用序列 (按原 walk 逆序), 消除每步 Lookup_LSA.
        bit 级安全: SPT 静态 (TYPE_2PI 只在 time==0 建一次), LSA 对象只原位改 used_band 不重建,
        缓存对象引用而非值, 运行时读 .used_band/.total_band 与原代码逐位相同."""
        self._n2n_path = [[[None for _ in range(len(self.SPT[con-1]))] for _ in range(len(self.SPT[con-1]))] for con in range(1, len(self.LSDB)+1)]
        for con in range(1, len(self.LSDB)+1):
            for source in range(1, len(self.SPT[con-1])+1):
                for dest in range(1, len(self.SPT[con-1][source-1])+1):
                    path = []
                    if source != dest:
                        pre = self.SPT[con-1][source-1][dest-1].pre
                        next_n = self.SPT[con-1][source-1][dest-1].destination
                        while pre != source or next_n != source:
                            path.append(self.Lookup_LSA(con, pre, next_n))
                            next_n = self.SPT[con-1][source-1][pre-1].destination
                            pre = self.SPT[con-1][source-1][pre-1].pre
                    self._n2n_path[con-1][source-1][dest-1] = path
        self._n2n_path_built = True

    def Update_N2N_Load_By_LSDB(self, con, source, dest, delta=0, changed_lsa=None):
        if source == dest:
            self.N2N_status[con-1][source-1][dest-1].load_rate = 1
            return
        n2n = self.N2N_status[con-1][source-1][dest-1]

        # 增量更新: 只有单条链路变化, O(1)更新load_rate
        if delta != 0 and changed_lsa is not None and changed_lsa.total_band > 0:
            new_free = changed_lsa.total_band - changed_lsa.used_band
            old_free = new_free + delta  # 反向推算旧值
            old_ratio = old_free / changed_lsa.total_band
            new_ratio = new_free / changed_lsa.total_band
            if old_ratio > 0 and n2n.load_rate > 0:
                n2n.load_rate = min(1.0, n2n.load_rate / old_ratio * new_ratio)
            # free_band: 新瓶颈或旧瓶颈移除才需要全量
            if new_free < n2n.free_band:
                n2n.free_band = max(0.0, new_free)
            elif n2n.free_band >= old_free - 1e-6:  # 旧瓶颈被移除
                pass  # fall through to full recompute below
            else:
                return  # load_rate已更新, free_band未变, 完成!

        # 全量重建 (初始化或瓶颈变更时)
        if not self._n2n_path_built:
            self._build_n2n_path_cache()
        min_total, min_free, load = maxsize, maxsize, 1.0
        for lsa in self._n2n_path[con-1][source-1][dest-1]:
            tb = lsa.total_band if lsa is not None else 0
            if tb > 0:
                free = tb - lsa.used_band
                load *= free / tb
                if tb < min_total: min_total = tb
                if free < min_free: min_free = free
            else:
                # 链路故障: 路径不通, free_band=0
                min_free = 0.0; min_total = 0.0; load = 0.0
        n2n.load_rate = load
        n2n.free_band = min_free
        n2n.total_band = min_total

    #从LSDB中遍历，更新N2N的带宽及负载率
    def Update_N2N_Load_By_LSDB_All(self):
        for i in range(len(self.LSDB)):
            for j in range(len(self.LSDB[i])):
                for k in range(len(self.LSDB[i])):
                    self.Update_N2N_Load_By_LSDB(i+1,j+1,k+1)
    
    def Update_N2N_Busy_Node(self):
        for i in range(len(self.N2N_status)):
            for j in range(len(self.N2N_status[i])):
                for k in range(len(self.N2N_status[i][j])):
                    self.N2N_status[i][j][k].busy_node = 0
        for i in range(len(self.LSA_N2N)):
            for j in range(len(self.LSA_N2N[i])):
                for k in range(len(self.LSA_N2N[i][j])):
                    for n in range(len(self.LSA_N2N[i][j][k])):
                        self.N2N_status[i][self.LSA_N2N[i][j][k][n].source-1][self.LSA_N2N[i][j][k][n].destination-1].busy_node += 1

                    

    #更新每条链路上经过的N2N
    def Link_LSDB_With_N2N_All(self):
        if self._lsa_n2n_built:
            return  # SPT/LSA索引未变, 反向映射已最新, 跳过重复重建 (bit级安全)
        for i in range(len(self.LSA_N2N)):
            for j in range(len(self.LSA_N2N[i])):
                for k in range(len(self.LSA_N2N[i][j])):
                    self.LSA_N2N[i][j][k].clear()
        for i in range(len(self.N2N_status)):
            for j in range(len(self.N2N_status[i])):
                for k in range(len(self.N2N_status[i][j])):
                    self.Link_LSDB_With_N2N(i+1,j+1,k+1)
        self._lsa_n2n_built = True

    #更新每条链路上经过的N2N
    def Link_LSDB_With_N2N(self, con, source, dest):
        pre = self.SPT[con-1][source-1][dest-1].pre
        next_n = self.SPT[con-1][source-1][dest-1].destination
        while pre!=source or next_n!=source:
            if pre<1:
                break
            res = self.Lookup_LSA_Index(con,pre,next_n)
            self.LSA_N2N[con-1][pre-1][res].append(self.N2N_status[con-1][source-1][dest-1])
            next_n = self.SPT[con-1][source-1][pre-1].destination
            pre = self.SPT[con-1][source-1][pre-1].pre

    #记录网络的统计信息
    def Record_Net_Statics(self, flag=False):
        for i in range(len(self.statics)):
            if self.statics[i].band_usage>0:
                total = self.statics[i].s_size*self.statics[i].average_usage
                self.statics[i].s_size += 1
                self.statics[i].average_usage = (total+self.statics[i].band_usage)/self.statics[i].s_size
            else:
                self.statics[i].s_size=0
                self.statics[i].average_usage=0
            NETout.write("%-12s%-12s%-12s%-12s%-12s%-12s%-12s%-16s\n"%("Con_ID","Capacity","Throughout","Load_Rate","N2N_Band", \
                "Band_Usage","s_size","average_usage"))
            NETout.write("%-12d%-12d%-12d%-12f%-12d%-12f%-12d%-16f\n"%(self.statics[i].con_id,self.statics[i].throughout_limit, \
                self.statics[i].throughout,self.statics[i].load_rate,self.statics[i].N2N_throughout, \
                self.statics[i].band_usage,self.statics[i].s_size,self.statics[i].average_usage))
        #NETout.flush()

    def Record_LSDB(self):
        if 5>NET_LOG_LEVEL:
            LSDBout.write("Time:%d\n"%self.topo.current_time)
        for i in range(len(self.LSDB)):
            total_band = 0
            used_band = 0
            if 5>NET_LOG_LEVEL:
                LSDBout.write("%20s%d\n"%('Constellation:',self.topo.constellation[i].ID))
                LSDBout.write("%-8s%-8s%-8s%-12s%-8s%-8s%-12s%-12s\n"%('Status','Source',"SIF","Destination","DIF","Metric","Total_Band","Used_Band"))
            for j in range(len(self.LSDB[i])):
                for k in range(len(self.LSDB[i][j])):
                    if 5>NET_LOG_LEVEL:
                        LSDBout.write("%-8d%-8d%-8d%-12d%-8d%-8d%-12d%-12d\n"%(self.LSDB[i][j][k].isEstablished,j+1,
                    self.LSDB[i][j][k].source_interface,self.LSDB[i][j][k].destinate_ID,self.LSDB[i][j][k].destinate_interface,
                    self.LSDB[i][j][k].metric,self.LSDB[i][j][k].total_band,self.LSDB[i][j][k].used_band))
                    if self.LSDB[i][j][k].isEstablished:
                        total_band += self.LSDB[i][j][k].total_band
                        used_band += self.LSDB[i][j][k].used_band
            self.statics[i].throughout_limit = total_band
            self.statics[i].throughout = used_band
            self.statics[i].load_rate = used_band/total_band
    
    def Record_SPT(self):
        if 5>NET_LOG_LEVEL:
            SPTout.write("Time:%d\n"%self.topo.current_time)
            for i in range(len(self.SPT)):
                SPTout.write("%20s%d\n"%('Constellation:',self.topo.constellation[i].ID))
                SPTout.write("%-8s%-8s%-12s%-8s%-12s%-8s\n"%('Status','Source',"Destination","Hops","Distance","Path"))
                for j in range(len(self.SPT[i])):
                    for k in range(len(self.SPT[i][j])):
                        SPTout.write("%-8d%-8d%-12d"%(self.SPT[i][j][k].isReached,j+1,k+1))
                        if j==k:
                            SPTout.write("\n")
                            continue
                        pre = self.SPT[i][j][k].pre
                        next_n = self.SPT[i][j][k].destination
                        s = Stack()
                        s.push(next_n)
                        while(pre!=j+1 or next_n!=j+1):
                            next_n = self.SPT[i][j][pre-1].destination
                            pre = self.SPT[i][j][pre-1].pre
                            s.push(next_n)
                        SPTout.write("%-8s%-12s"%(s.size()-1,self.SPT[i][j][k].distance))
                        while s.empty()==False:
                            SPTout.write("%d"%s.top())
                            if s.size()==1:
                                SPTout.write("\n")
                            else:
                                SPTout.write("->")
                            s.pop()


    def Record_Routingtable(self):
        if 3>NET_LOG_LEVEL:
            RTout.write("Time:%d\n"%self.topo.current_time)
            for i in range(len(self.forwarding_table)):
                RTout.write("%20s%d\n"%('Constellation:',self.topo.constellation[i].ID))
                RTout.write("%-8s%-8s%-12s%-8s%-12s%-8s\n"%('Status','Source',"Destination","NextHop","Interface","Cost"))
                for j in range(len(self.forwarding_table[i])):
                    for k in range(len(self.forwarding_table[i][j])):
                        RTout.write("%-8d%-8d%-12d%-8d%-12d%-8d\n"%(self.forwarding_table[i][j][k].isReached,j+1,
                        self.forwarding_table[i][j][k].destination,self.forwarding_table[i][j][k].nexthop,
                        self.forwarding_table[i][j][k].interface,self.forwarding_table[i][j][k].cost))
        
    def Record_N2N(self):
        if 5>NET_LOG_LEVEL:
            N2Nout.write("Time:%d\n"%self.topo.current_time)
        for i in range(len(self.N2N_status)):
            used_band = 0
            if 5>NET_LOG_LEVEL:
                N2Nout.write("%20s%d\n"%('Constellation:',self.topo.constellation[i].ID))
                N2Nout.write("%-8s%-12s%-12s%-12s%-12s%-12s\n"%("Source","Destination","Total_Band","Used_Band","Free_Band","Load_Rate"))
            for j in range(len(self.N2N_status[i])):
                for k in range(len(self.N2N_status[i][j])):
                    if 5>NET_LOG_LEVEL:
                        N2Nout.write("%-8d%-12d%-12d%-12d%-12d%-12f\n"%(j+1, self.N2N_status[i][j][k].destination, \
                        self.N2N_status[i][j][k].total_band,self.N2N_status[i][j][k].used_band,self.N2N_status[i][j][k].free_band, \
                            self.N2N_status[i][j][k].load_rate))
                    used_band += self.N2N_status[i][j][k].used_band;
            self.statics[i].N2N_throughout = used_band
            if self.statics[i].throughout>0:
                self.statics[i].band_usage = used_band/self.statics[i].throughout
            else: 
                self.statics[i].band_usage=0

    def Record_NSA(self):
        if 5>NET_LOG_LEVEL:
            NSAout.write("Time:%d\n"%self.topo.current_time)
            for i in range(len(self.node_status)):
                NSAout.write("%20s%d\n"%('Constellation:',self.topo.constellation[i].ID))
                NSAout.write("%-8s%-12s%-12s\n"%("Source","Total_Band","Used_Band"))
                for j in range(len(self.node_status[i])):
                    NSAout.write("%-8d%-12d%-12d\n"%(j+1,self.node_status[i][j].total_band,self.node_status[i][j].used_band))

    def Record_LSA_N2N(self):
        if 5>NET_LOG_LEVEL:
            LNout.write("Time:%d\n"%self.topo.current_time)
            for i in range(len(self.LSA_N2N)):
                LNout.write("%20s%d\n"%('Constellation:',self.topo.constellation[i].ID))
                LNout.write("%-8s%-12s%-8s%s\n"%("Source","Destination","Count","N2N"))
                for j in range(len(self.LSA_N2N[i])):
                    for k in range(len(self.LSA_N2N[i][j])):
                        LNout.write("%-8s%-12s%-8s"%(j+1,self.LSDB[i][j][k].destinate_ID,len(self.LSA_N2N[i][j][k])))
                        for n in range(len(self.LSA_N2N[i][j][k])):
                            LNout.write("(%d,%d) "%(self.LSA_N2N[i][j][k][n].source,self.LSA_N2N[i][j][k][n].destination))
                        LNout.write("\n")

    def Record(self):
        self.Record_LSDB()
        #self.Record_SPT()
        #self.Record_Routingtable()
        self.Record_N2N()
        #self.Record_NSA()
        #self.Record_LSA_N2N()
        self.Record_Net_Statics()
