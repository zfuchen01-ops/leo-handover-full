from Logger import Logger
from time import time
from Defination import *
from Network import Network
from Satellite import Satellite
from Topology import Topology
from Position import *
from User import User
from enum import Enum
from math import log2, pi as _pi
import sys
import os
import numpy as np
import logging
import random
import os

LIGHT_VEL = 3.0e+8
CARRIER = 2.0e+10
RAIN = -2.0
RICE = 1.0
EIRP = 10000.0
GS = 10000.0
GAUSE_NOISE = 3.98e-21
C_BAND = float(os.environ.get("LEO_C_BAND", USER_BAND))
INCLUDE_GSL_INTERFERENCE = os.environ.get("LEO_INCLUDE_GSL_INTERFERENCE", "1") == "1"
IPQ_MODE = os.environ.get("LEO_IPQ_MODE", "n2n_load").strip().lower()
IPQ_LOAD_THRESHOLD = int(os.environ.get("LEO_IPQ_LOAD_THRESHOLD", "0"))
PLDR_LIFETIME_ALPHA = float(os.environ.get("LEO_PLDR_LIFETIME_ALPHA", "1.0"))
PLDR_FREE_ALPHA = float(os.environ.get("LEO_PLDR_FREE_ALPHA", "1.0"))
HANDOVER_COST_HIGH_LOAD = os.environ.get("LEO_HANDOVER_COST_HIGH_LOAD")
HANDOVER_COST_HIGH_LOAD = float(HANDOVER_COST_HIGH_LOAD) if HANDOVER_COST_HIGH_LOAD is not None else None
HANDOVER_COST_LOAD_THRESHOLD = int(os.environ.get("LEO_HANDOVER_COST_LOAD_THRESHOLD", "0"))
DESTINATION_DECISION_MODE = os.environ.get("LEO_DESTINATION_DECISION_MODE", "SERVICE_TIME").strip().upper()
DELAY_HYSTERESIS = float(os.environ.get("LEO_DELAY_HYSTERESIS", "0.05"))
DESTINATION_MIN_SERVICE_TIME = float(os.environ.get("LEO_DESTINATION_MIN_SERVICE_TIME", "31.0"))
CHANNEL_QUALITY_MIN_SERVICE_TIME = float(os.environ.get("LEO_CHANNEL_QUALITY_MIN_SERVICE_TIME", "0.0"))
CHANNEL_QUALITY_HYSTERESIS_RATIO = float(os.environ.get("LEO_CHANNEL_QUALITY_HYSTERESIS_RATIO", "0.0"))
CHANNEL_QUALITY_DECISION_NOISE = os.environ.get("LEO_CHANNEL_QUALITY_DECISION_NOISE", "actual").strip().lower()
CHANNEL_QUALITY_MIN_HOLD_SLOTS = int(os.environ.get("LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS", "0"))
CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS = os.environ.get("LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS", "").strip()
MGCS_DELAY_WEIGHT = float(os.environ.get("LEO_MGCS_DELAY_WEIGHT", "0.0"))
MGCS_CAPACITY_TOLERANCE = float(os.environ.get("LEO_MGCS_CAPACITY_TOLERANCE", "0.0"))
RATE_UPPER = C_BAND*log2(1+EIRP*GS*pow(LIGHT_VEL/4.0/pi/SAT_HEIGHT/CARRIER,2.0)* \
    pow(10.0,0.3*SAT_HEIGHT*RAIN/SAT_HEIGHT)*RICE/(GAUSE_NOISE*C_BAND*1e+6))


def _log_path(path):
    return os.devnull if os.environ.get("LEO_QUIET_LOGS", "0") == "1" else path


hout = open(_log_path('./log/handover/Handover.log'),'w')
pout = open(_log_path('./log/handover/Hoparam.log'),'w')
hsout = Logger('./log/handover/Statics.log',level='debug',w_level='info')

dlout = open(_log_path('./log/handover/Delay.log'),'w')
htout = open(_log_path('./log/handover/Hotimes.log'),'w')
tpout = open(_log_path('./log/handover/Throughput.log'),'w')
tpout2 = open(_log_path('./log/handover/Allocate_band.log'),'w')
sout = open(_log_path('./log/handover/Handover_statics.log'),'a')

class HoParam:
    def __init__(self):
        #self.Q_value = 0.0
        self.rate_integral = 0
        self.s_time = 0
        self.c_quality = 0
        self.c_resource = 0
        #self.elevation = 0.0
        #self.distance = 0.0
        #self.available_band = 0.0
        self.delay = 0
        self.handover_factor = 0
        self.value = 0


def Is_GSL_Interference_Source(user: User) -> bool:
    return len(getattr(user, "user_to_connect_to", {})) > 0


def Apply_Path_Delay_Weight(base_value, delay_ms, weight, scale=1.0):
    return base_value + weight * delay_ms / scale


def Select_MGCS_Candidate(candidates, capacity_tolerance):
    if not candidates:
        return None
    max_quality = max(quality for _, quality, _ in candidates)
    threshold = max_quality * max(0.0, 1.0 - capacity_tolerance)
    eligible = [candidate for candidate in candidates if candidate[1] >= threshold]
    return min(eligible, key=lambda candidate: candidate[2])[0]


def delay_components_ms(
    source_gsl_distance,
    isl_distance,
    destination_gsl_distance,
    flow_count,
    legacy_hop_isl_distance=0.0,
):
    divisor = max(flow_count, 1) * 300000.0
    source_gsl_ms = source_gsl_distance / divisor
    isl_ms = isl_distance / divisor
    destination_gsl_ms = destination_gsl_distance / divisor
    return {
        "source_gsl_ms": source_gsl_ms,
        "isl_ms": isl_ms,
        "destination_gsl_ms": destination_gsl_ms,
        "total_ms": source_gsl_ms + isl_ms + destination_gsl_ms,
        "legacy_hop_isl_ms": legacy_hop_isl_distance / divisor,
        "legacy_hop_total_ms": (
            source_gsl_distance + legacy_hop_isl_distance + destination_gsl_distance
        ) / divisor,
    }


class Handover:
    def __init__(self, topo=None, net=None):
        self.block_count = 0
        self.ho_count = 0
        self.ho = {}
        self.ob = {}    #RL
        self.statics = [[],[],[],[]]    #[切换次数，平均时延，实际吞吐量，信道容量]
        self.weight_samples = []
        self.delay_component_samples = []
        self.source_ho_count = 0
        self._hops_cache = {}  # (sat_id, feeder_id) -> hops, 每集预计算
        self._rvt_cache = {}   # (sat_id, time) -> Sphere_Position, 每时刻跨用户复用
        self._rvt_cache_time = None
        self.destination_ho_count = 0
        if net!=None:
            self.net = net
            self.topo = net.topo
        elif topo!=None:
            self.topo = topo
        else:
            exit(1)
        for user in self.topo.user:
            self.ho[user] = {}
            self.ob[user] = []

    #覆盖关系变化，重新为每个终端初始化切换数据结构
    def Initial_Handover(self):
        for user in self.ho:
            self.ho[user].clear()
            for sat in user.sat_covered:
                self.ho[user][sat] = HoParam()
    
    #初始化星地连接关系(循环试验时添加)
    def Initial_GSlink(self):
        for user in self.topo.user:
            user.last_connected = None
            if user.sat_connected!=None:
                user.sat_connected.user_connected.clear()
                user.sat_connected=None
                user.user_connecting_by.clear()
                user.user_connecting_to.clear()

    #传统不包含网络的切换
    def Run_Tranditional_Handover(self, start, end, step, mode:str):
        assert(step>=1)
        assert(start>=0)
        assert(end>=start)
        self.block_count=0
        self.ho_count=0
        self.Initial_GSlink()
        self.Tranditional_Handover(start,'CHANNEL_QUALITY')
        for t in range(start+step,end,step):
            self.Tranditional_Handover(t,mode)

        hsout.logger.critical("Handover success count:%d"%self.ho_count)
        hsout.logger.critical("Handover fail count:%d"%self.block_count)

    def Tranditional_Handover(self, time, mode:str):
        assert(time>=0)
        logging.debug('time %d', time)
        self.topo.Update_Topology_Status(time)
        self.Initial_Handover()
        self.Update_Channel_Resource()
        self.Trig_Decision(mode,False)

    #切换会改变网络状态
    def Run_Network_Handover(self, start, end, step, mode:str):
        assert(step>=1)
        assert(start>=0)
        assert(end>=start)
        self.statics = [[],[],[],[]]
        self.weight_samples = []
        self.delay_component_samples = []
        self.source_ho_count = 0
        self.destination_ho_count = 0
        self.net.Initial_Network()
        self.Initial_GSlink()
        self.block_count=0
        self.ho_count=0
        self.Network_Handover(start,'NETWORK_INITIAL')
        if os.environ.get("LEO_RESET_HANDOVER_AFTER_INITIAL", "0") == "1":
            self.ho_count = 0
            self.source_ho_count = 0
            self.destination_ho_count = 0
        progress_every = int(os.environ.get("LEO_PROGRESS_EVERY", "0"))
        slot_times = list(range(start+step,end,step))
        total_slots = len(slot_times)
        for slot_index, t in enumerate(slot_times, start=1):
            if progress_every > 0 and slot_index % progress_every == 0:
                done = int(24 * slot_index / total_slots) if total_slots else 24
                bar = "#" * done + "." * (24 - done)
                percent = 100.0 * slot_index / total_slots if total_slots else 100.0
                print("[%s] %5.1f%% mode=%s slot=%d/%d time=%d ho=%d block=%d"%(
                    bar, percent, mode, slot_index, total_slots, t, self.ho_count, self.block_count), flush=True)
            self.Network_Handover(t, mode)
        htout.write("Start:%d End:%d Step:%d Time:%d\n"%(start,end,step,self.ho_count))
        self.Record_Statics()

    def Network_Handover(self, time, mode:str):
        assert(time>=0)
        logging.debug('time %d', time)
        self.topo.Update_Topology_Status(time)
        if self.topo.constellation[0].type == TYPE_PI:
            self.net.Update_LSDB_Netmode()
        if mode=='NETWORK_INITIAL':
            if self.topo.constellation[0].type == TYPE_2PI:
                self.net.Update_LSDB_Netmode()
              #仅适用于星间链路不断的星座
            self.net.Dijkstra_All()
            self.net.Generate_Forwardingtable_By_AllNode()
            self.net.Link_LSDB_With_N2N_All()
            self.net.Update_N2N_Load_By_LSDB_All()
            # self.net.Update_NSA_Band_All()
        self.Initial_Handover()
        self.Update_Channel_Resource()
        self.Update_Service_Time(self.topo.current_time)
        self.Update_Trans_Rate(self.topo.current_time)
        self.Update_Delay()
        if mode in ('RATE_INTEGRAL', 'UNION_MODE_1', 'UNION_MODE_2'):
            self.Update_Rate_Integral(self.topo.current_time)
        if mode in ('UNION_MODE_1', 'UNION_MODE_2'):
            self.Update_Handover_Factor()
        for user in self.ho:
            if (mode == 'NETWORK_INITIAL') or (user.user_ID <= len(self.topo.user)/2):
                self.Trig_Decision(mode,True,user)
            else:
                self.Trig_Decision(DESTINATION_DECISION_MODE,True,user)
        if self.topo.constellation[0].type != TYPE_2PI:
            self.net.Link_LSDB_With_N2N_All()
        if os.environ.get("LEO_QUIET_LOGS", "0") != "1":
            self.net.Record()
        self.Record_Delay()
        self.Record_Throughput()
        self.Record_Allocate()
        self.Record_Hotimes()

    def Get_Channel_Quality_Min_Hold_Slots(self):
        if not CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS:
            return CHANNEL_QUALITY_MIN_HOLD_SLOTS
        points = []
        for item in CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS.split(","):
            item = item.strip()
            if not item:
                continue
            users, value = item.split(":", 1)
            points.append((int(users.strip()), float(value.strip())))
        points.sort()
        user_count = len(self.topo.user)
        if user_count <= points[0][0]:
            return int(round(points[0][1]))
        if user_count >= points[-1][0]:
            return int(round(points[-1][1]))
        for (left_users, left_value), (right_users, right_value) in zip(points, points[1:]):
            if left_users <= user_count <= right_users:
                span = right_users - left_users
                ratio = 0.0 if span <= 0 else (user_count - left_users) / span
                return int(round(left_value + ratio * (right_value - left_value)))
        return int(round(points[-1][1]))
        

    # 批量切换决策
    # CHANNEL_QUALITY
    # DISTANCE
    # ELEVATION
    # RATE_INTEGRAL
    # SERVICE_TIME
    # CHANNEL_RESOURCE
    # NETWORK_LOAD
    # NETWORK_INITIAL
    # UNION_MODE_1     #网络拥塞+切换因子
    # UNION_MODE_2     #网络拥塞+信息率积分
    def Trig_Decision(self, mode:str, isNet:bool, user:User):
        if mode=='CHANNEL_QUALITY':
            hold_slots = self.Get_Channel_Quality_Min_Hold_Slots()
            if hold_slots > 0 and user.sat_connected in self.ho[user]:
                last_time = getattr(user, "_last_handover_time", None)
                if last_time is not None and self.topo.current_time - last_time < hold_slots * SLOT_SECONDS:
                    self.Trig_Handover(user.sat_connected,user,'NETWORK') if isNet==True else self.Trig_Handover(user.sat_connected,user,'OTHERS')
                    return
            if CHANNEL_QUALITY_MIN_SERVICE_TIME > 0 and user.sat_connected in self.ho[user]:
                if self.ho[user][user.sat_connected].s_time >= CHANNEL_QUALITY_MIN_SERVICE_TIME:
                    self.Trig_Handover(user.sat_connected,user,'NETWORK') if isNet==True else self.Trig_Handover(user.sat_connected,user,'OTHERS')
                    return
            max_quality = 0.0
            temp = None
            candidates = []
            for sat in self.ho[user]:
                if hasattr(self, "Calc_Channel_Quality_Decision"):
                    decision_quality = self.Calc_Channel_Quality_Decision(sat, user)
                else:
                    decision_quality = self.ho[user][sat].c_quality
                decision_quality = Apply_Path_Delay_Weight(
                    decision_quality,
                    self.ho[user][sat].delay,
                    MGCS_DELAY_WEIGHT,
                )
                if sat.beam-len(sat.user_connected)>0 or \
                    (user.sat_connected!=None and sat == user.sat_connected):
                    candidates.append((sat, decision_quality, self.ho[user][sat].delay))
                if decision_quality>max_quality and (sat.beam-len(sat.user_connected)>0 or \
                    (user.sat_connected!=None and sat == user.sat_connected)):
                    max_quality = decision_quality
                    temp = sat
            if MGCS_CAPACITY_TOLERANCE > 0:
                temp = Select_MGCS_Candidate(candidates, MGCS_CAPACITY_TOLERANCE)
            if CHANNEL_QUALITY_HYSTERESIS_RATIO > 0 and user.sat_connected in self.ho[user] and temp != user.sat_connected:
                current_quality = self.ho[user][user.sat_connected].c_quality
                if max_quality <= current_quality * (1.0 + CHANNEL_QUALITY_HYSTERESIS_RATIO):
                    temp = user.sat_connected
            self.Trig_Handover(temp,user,'NETWORK') if isNet==True else self.Trig_Handover(temp,user,'OTHERS')
        elif mode=='RATE_INTEGRAL':
            max_rate = 0.0
            temp = None
            for sat in self.ho[user]:
                if self.ho[user][sat].rate_integral>max_rate and (sat.beam-len(sat.user_connected)>0 or \
                    (user.sat_connected!=None and sat == user.sat_connected)):
                    max_rate = self.ho[user][sat].rate_integral
                    temp = sat
            self.Trig_Handover(temp,user,'NETWORK') if isNet==True else self.Trig_Handover(temp,user,'OTHERS')
        elif mode=='SERVICE_TIME':
            temp = None
            if self.ho[user][user.sat_connected].s_time >=31:
                temp = user.sat_connected
            else:
                max_time = 0.0
                for sat in self.ho[user]:
                    if self.ho[user][sat].s_time>max_time and (sat.beam-len(sat.user_connected)>0 or \
                        (user.sat_connected!=None and sat == user.sat_connected)):
                        max_time = self.ho[user][sat].s_time
                        temp = sat
            self.Trig_Handover(temp,user,'NETWORK') if isNet==True else self.Trig_Handover(temp,user,'OTHERS')
        elif mode == 'RANDOM':
            length = len(self.ho[user])
            index = random.randint(1,length)
            count = 1
            temp = None
            for sat in self.ho[user]:
                if count == index:
                    temp = sat
                    break
                count += 1
            self.Trig_Handover(temp,user,'NETWORK') if isNet==True else self.Trig_Handover(temp,user,'OTHERS')
        elif mode == 'DELAY':
            min_delay = 10000
            temp = None
            for sat in self.ho[user]:
                if self.ho[user][sat].delay<min_delay and (sat.beam-len(sat.user_connected)>0 or \
                    (user.sat_connected!=None and sat == user.sat_connected)):
                    min_delay = self.ho[user][sat].delay
                    temp = sat
            self.Trig_Handover(temp,user,'NETWORK') if isNet==True else self.Trig_Handover(temp,user,'OTHERS')
        elif mode == 'DELAY_HYSTERESIS':
            min_delay = 10000
            temp = None
            for sat in self.ho[user]:
                if self.ho[user][sat].delay<min_delay and (sat.beam-len(sat.user_connected)>0 or \
                    (user.sat_connected!=None and sat == user.sat_connected)):
                    min_delay = self.ho[user][sat].delay
                    temp = sat
            if user.sat_connected != None and temp != None and temp != user.sat_connected:
                current = self.ho[user][user.sat_connected]
                current_is_stable = current.s_time >= DESTINATION_MIN_SERVICE_TIME
                delay_gain = current.delay - min_delay
                hysteresis_margin = max(0.0, current.delay * DELAY_HYSTERESIS)
                if current_is_stable and delay_gain <= hysteresis_margin:
                    temp = user.sat_connected
            self.Trig_Handover(temp,user,'NETWORK') if isNet==True else self.Trig_Handover(temp,user,'OTHERS')
        elif mode=='NETWORK_LOAD':
            max_net_resource = -1.0
            temp = None
            for sat in self.ho[user]:
                self.ho[user][sat].available_band = self.Calc_Available_Band(sat,user)
                if self.ho[user][sat].available_band>max_net_resource and (sat.beam-len(sat.user_connected)>0 or \
                    (user.sat_connected!=None and sat == user.sat_connected)):
                    max_net_resource = self.ho[user][sat].available_band
                    temp = sat
            self.Trig_Handover(temp,user,'NETWORK')
        #第一个时隙的接入需要特殊处理
        elif mode=='NETWORK_INITIAL':
            max_time = 0.0
            temp = None
            for sat in self.ho[user]:
                if self.ho[user][sat].s_time>max_time and (sat.beam-len(sat.user_connected)>0 or \
                    (user.sat_connected!=None and sat == user.sat_connected)):
                    max_time = self.ho[user][sat].s_time
                    temp = sat
            self.Trig_Handover(temp,user,'INITIAL')
        elif mode=='UNION_MODE_1':
            max_value = -1
            temp = None
            mu = self.Effective_Handover_Cost()
            path_quality_sum = 0.0
            candidate_count = 0
            for sat in self.ho[user]:
                self.ho[user][sat].available_band = self.Calc_Available_Band(sat,user)
                path_quality_sum += self.ho[user][sat].available_band
                candidate_count += 1
            g_weight = (1.0 - mu) * path_quality_sum / candidate_count if candidate_count > 0 else 0.0
            g_weight = max(0.0, min(1.0 - mu, g_weight))
            light_load = max(0.0, (LIGHT_LOAD_USER_THRESHOLD - len(self.topo.user)) / LIGHT_LOAD_USER_THRESHOLD) \
                if LIGHT_LOAD_USER_THRESHOLD > 0 else 0.0
            g_weight = max(g_weight, (1.0 - mu) * light_load)
            f_weight = 1.0 - mu - g_weight
            self.weight_samples.append((f_weight, g_weight, mu))
            for sat in self.ho[user]:
                handover_cost = 0.0
                if user.sat_connected != None and user.sat_connected != sat:
                    handover_cost = -self.ho[user][sat].rate_integral
                self.ho[user][sat].value = f_weight * self.ho[user][sat].available_band + \
                    g_weight * self.ho[user][sat].rate_integral + mu * handover_cost
                if self.ho[user][sat].value>max_value and (sat.beam-len(sat.user_connected)>0 or \
                    (user.sat_connected!=None and sat == user.sat_connected)):
                    max_value = self.ho[user][sat].value
                    temp = sat
            self.Trig_Handover(temp,user,'NETWORK') if isNet==True else self.Trig_Handover(temp,user,'OTHERS')
        else:
            print("Wrong mode type!!!")

    #执行切换
    def Record_Handover_Event(self, user:User):
        self.ho_count += 1
        user._last_handover_time = self.topo.current_time
        if user.user_ID <= len(self.topo.user) / 2:
            self.source_ho_count += 1
        else:
            self.destination_ho_count += 1

    def Trig_Handover(self, s_target:Satellite, user:User, mode:str):
        #未找到合适切换卫星，断开星地链路
        user.last_connected = user.sat_connected
        if s_target==None:
            if user.sat_connected!=None:
                temp_target = user.sat_connected
                # 30s步长: 旧卫星可能飞出, 始终释放LSA防止泄漏
                self.net.User_Disconnect_Satellite(user,temp_target)  # 无ho检查, 直接释放
                if temp_target in self.ho[user]:
                    temp_target.user_connected.remove(user)
                user.sat_connected=None
            if 3>HO_LOG_LEVEL:
                hout.write("Time:%d User:%d located at %.4f %.4f blocked!!!\n" \
                %(self.topo.current_time,user.user_ID,user.we_pos.lon,user.we_pos.lat))
            self.block_count += 1
        else:
            #终端未接入卫星，建立连接
            band = self.ho[user][s_target].c_quality  # 不cap, Shannon天然上限
            if user.sat_connected==None:
                if (mode=='NETWORK' or mode=='INITIAL') and self.net.User_Connect_Satellite(user,s_target,band)==False:
                    if 3>HO_LOG_LEVEL:
                        hout.write("User:%d fails to connect to Sat:%d\n"%(user.user_ID,s_target.ID))
                    return
                user.sat_connected=s_target
                s_target.user_connected.add(user)
                if 3>HO_LOG_LEVEL:
                    hout.write("Time:%d User:%d located at %.4f %.4f handovers to Sat:%d located at %.4f %.4f\n" \
                    %(self.topo.current_time, user.user_ID, user.we_pos.lon, user.we_pos.lat, s_target.ID, s_target.we_pos.lon, s_target.we_pos.lat))
                self.Record_Handover_Event(user)
            #终端接入卫星，断开旧连接，建立新连接
            elif mode!='INITIAL':
                same_flag = (s_target == user.sat_connected)
                if same_flag:
                    return  # 同星切换:跳过disconnect/reconnect
                temp_target = user.sat_connected
                # 30s步长: 旧卫星可能飞出, 始终释放LSA防止泄漏
                self.net.User_Disconnect_Satellite(user,temp_target)  # 无ho检查, 直接释放
                if temp_target in self.ho[user]:
                    temp_target.user_connected.remove(user)
                user.sat_connected=None
                if (mode=='NETWORK' or mode=='INITIAL') and self.net.User_Connect_Satellite(user,s_target,band)==False:
                    if 3>HO_LOG_LEVEL:
                        hout.write("User:%d fails to connect to Sat:%d\n"%(user.user_ID,s_target.ID))
                    return
                user.sat_connected=s_target
                s_target.user_connected.add(user)
                if 3>HO_LOG_LEVEL:
                    hout.write("Time:%d User:%d located at %.4f %.4f handovers to Sat:%d located at %.4f %.4f\n" \
                    %(self.topo.current_time,user.user_ID,user.we_pos.lon,user.we_pos.lat,s_target.ID,s_target.we_pos.lon,s_target.we_pos.lat))
                if same_flag == False:
                    self.Record_Handover_Event(user)
            #如果是首次接入，但是由于寻呼该终端已经接入卫星，且接入卫星与当前决策的相同（如果不同在上一步if会进入），为该终端其他连接分配资源
            elif mode=='INITIAL':
                if self.net.User_Connect_Satellite(user,s_target,band)==False:
                    if 3>HO_LOG_LEVEL:
                        hout.write("User:%d fails to connect to Sat:%d\n"%(user.user_ID,s_target.ID))
                    return
                if 3>HO_LOG_LEVEL:
                    hout.write("Time:%d User:%d located at %.4f %.4f handovers to Sat:%d located at %.4f %.4f\n" \
                    %(self.topo.current_time,user.user_ID,user.we_pos.lon,user.we_pos.lat,s_target.ID,s_target.we_pos.lon,s_target.we_pos.lat))
                self.Record_Handover_Event(user)
    
    #批量更新剩余信道资源
    def Update_Channel_Resource(self):
        for user in self.ho:
            for sat in self.ho[user]:
                temp = self.Calc_Available_Channel(sat)
                self.ho[user][sat].c_resource = temp/SAT_BEAM

    #批量更新信道质量，信号功率/噪声功率
    def Update_Channel_Quality(self):
        quality_upper = PT*GS*GT*pow(LIGHT_VEL/4.0/pi/SAT_HEIGHT/CARRIER,2.0)* \
            pow(10.0,0.3*SAT_HEIGHT*RAIN/SAT_HEIGHT)*RICE/GAUSE_NOISE   #归一化
        for user in self.ho:
            for sat in self.ho[user]:
                temp = self.Calc_Signal_Power(sat,user)/self.Calc_Download_Noise(sat,user)
                self.ho[user][sat].c_quality = temp/quality_upper

    #批量更新信道容量采用香农公式
    def Update_Channel_Capacity(self):
        for user in self.ho:
            for sat in self.ho[user]:
                self.ho[user][sat].c_quality = F_UTILITY*C_BAND*log2(1+self.Calc_Signal_Power(s, u, time)/self.Calc_Download_Noise(s, u, time))

    #批量更新终端与卫星距离
    def Update_Distance(self):
        dis_upper = sqrt(pow(SAT_HEIGHT,2.0)+2.0*SAT_HEIGHT*EARTH_RADIUS+pow(sin(USER_ELEVATION)*EARTH_RADIUS,2.0)) \
            -sin(USER_ELEVATION)*EARTH_RADIUS
        for user in self.ho:
            for sat in self.ho[user]:
                temp = Calc_Sphere_Distance(sat.we_pos,user.we_pos)
                self.ho[user][sat].distance = temp/dis_upper
    
    #批量更新终端与卫星仰角
    def Update_Elevation(self):
        for user in self.ho:
            for sat in self.ho[user]:
                temp = Calc_Sphere_Elevation(sat.we_pos,user.we_pos)
                self.ho[user][sat].elevation = temp*2.0/pi
    
    def Update_Trans_Rate(self,time):
        for user in self.ho:
            for sat in self.ho[user]:
                self.ho[user][sat].c_quality = int(self.Calc_Trans_Rate(time, sat, user))

    #批量更新所有终端的决策变量（最大信息率积分）
    def Update_Rate_Integral(self,time):
        for user in self.ho:
            for sat in self.ho[user]:
                temp = self.Calc_Rate_Integral(time,sat,user)
                duration = min(self.ho[user][sat].s_time, AGC_PREDICT_WINDOW)
                if duration > 0:
                    self.ho[user][sat].rate_integral = temp / (RATE_UPPER * duration)
                else:
                    self.ho[user][sat].rate_integral = 0
    
    #批量更新对应时刻服务时长
    def Update_Service_Time(self, time):
        #先通过距离方式计算出夹角，然后计算出地球角速率，最后算出时间。
        for user in self.ho:
            for sat in self.ho[user]:
                temp = self.Calc_Service_Time(time,sat,user)
                self.ho[user][sat].s_time = int(temp)
    
    #RL
    def Update_Available_Band(self):
        for user in self.ho:
            for sat in self.ho[user]:
                self.ho[user][sat].available_band = self.Calc_Available_Band(sat,user)
                if user.sat_connected == sat:
                    self.ho[user][sat].connected = True

    def Update_Delay(self):
        for user in self.ho:
            for u in user.user_to_connect_to:
                for source in self.ho[user]:
                    res = Calc_Sphere_Distance(user.we_pos, source.we_pos)
                    min_distance = 10000
                    temp = None
                    for dest in u.sat_covered:
                        if self.net.SPT[0][source.ID-1][dest.ID-1].distance < min_distance:
                            min_distance = self.net.SPT[0][source.ID-1][dest.ID-1].distance
                            temp = dest
                    if temp != None:
                        res += self.Calc_Path_Distance(source, temp)
                        res += Calc_Sphere_Distance(u.we_pos, temp.we_pos)
                    self.ho[user][source].delay = res/300000


    def Update_Handover_Factor(self):
        for user in self.ho:
            for sat in self.ho[user]:
                if user.sat_connected:
                    if user.sat_connected != sat:
                        self.ho[user][sat].handover_factor = -1
                    else:
                        self.ho[user][sat].handover_factor = 0
                else:
                    self.ho[user][sat].handover_factor = -1

    def Calc_Available_Channel(self, s:Satellite):
        return s.beam-len(s.user_connected)
    
    #路径损耗*雨衰*小尺度衰落（莱斯）
    def Calc_Signal_Power(self, s:Satellite, u:User, time=-1.0):
        #time为默认值，计算当前时刻
        if time==-1.0:
            distance = Calc_Sphere_Distance(s.we_pos,u.we_pos)
        else:
            s_we = Sphere_Position()
            s_we = s.Get_Satellite_We_Condition(time)
            distance = Calc_Sphere_Distance(s_we,u.we_pos)
        G = pow(LIGHT_VEL/4.0/pi/distance/CARRIER,2.0)*pow(10.0,0.3*distance*RAIN/(s.orbit_radius-EARTH_RADIUS))*RICE
        PS = EIRP*GS*G
        return PS
    
    #下行噪声，受到覆盖的其他卫星的干扰，干扰考虑天线角度衰落
    def Calc_Download_Noise(self, s:Satellite, u:User, time=-1.0):
        res = GAUSE_NOISE*C_BAND*1e+6
        if INCLUDE_GSL_INTERFERENCE:
            for other_user in s.user_connected:
                if other_user != u and Is_GSL_Interference_Source(other_user):
                    res += self.Calc_Signal_Power(s, other_user, time)
        # for sat in u.sat_covered:
        #     if sat!=s:
        #         A = Rectangle_Position()
        #         B = Rectangle_Position()
        #         C = Rectangle_Position()
        #         Convert_Sphere_to_Rectangle(u.we_pos,A)
        #         Convert_Sphere_to_Rectangle(s.we_pos,B)
        #         Convert_Sphere_to_Rectangle(sat.we_pos,C)
        #         angle = Calc_Angle_Cos(A,B,C)
        #         if angle>0:
        #             res += self.Calc_Signal_Power(sat,u,time)*angle
        return res

    #计算一对用户与卫星节点之间的服务时长
    #算法：二分法趋近，首先计算最小仰角，然后计算指定时刻倾角，通过二分法调整下次计算时刻，趋近切换时刻
    def Calc_Service_Time(self, begin_t, s:Satellite, u:User):
        delta = pi/s.vel
        temp_time = begin_t+delta
        azimuth = abs(Calc_Elevation_Azimuth(s.we_pos.radius,u.we_pos.radius,USER_ELEVATION))
        s_we = Sphere_Position()
        s_we = s.Get_Satellite_We_Condition(temp_time)
        temp_azimuth = Calc_Sphere_Azimuth(s_we,u.we_pos)
        while abs(temp_azimuth-azimuth)>0.01:    #0.0001根据所需服务时长计算精度可调
            delta = delta/2
            if temp_azimuth<azimuth:
                temp_time += delta
            else:
                temp_time -= delta
            s_we = s.Get_Satellite_We_Condition(temp_time)
            temp_azimuth = Calc_Sphere_Azimuth(s_we,u.we_pos)
        return temp_time-begin_t

    #香农公式
    def Calc_Trans_Rate(self, time, s:Satellite, u:User):
        signal = self.Calc_Signal_Power(s, u, time)
        noise = self.Calc_Download_Noise(s, u, time)
        return C_BAND*log2(1+signal/noise)

    #采用一阶牛顿-柯特斯公式计算定积分，即为梯形法，修改interval调整精度
    #此外python中有现成的积分库integrate
    def Calc_Rate_Integral(self, begin_t, s:Satellite, u:User):
        t = 0
        interval = 16
        end_time = min(self.ho[u][s].s_time, AGC_PREDICT_WINDOW)
        if end_time <= 0:
            return 0
        begin_rate = self.ho[u][s].c_quality
        end_rate = 0
        t+=interval
        res = 0
        temp = 0
        while t<=end_time:
            end_rate = self.Calc_Trans_Rate(begin_t+t, s, u)
            temp = (begin_rate+end_rate)/2.0*interval
            res += temp
            begin_rate = end_rate
            t+=interval
        end_rate = self.Calc_Trans_Rate(begin_t+end_time, s, u)
        res += (end_time+interval-t)*(begin_rate+end_rate)/2.0
        return res

    #计算卫星可用星间带宽，如果目的与源节点处于同一颗卫星，该值最大。
    def Calc_Available_Band(self, s:Satellite, u:User):
        """计算从卫星s到用户u的目的地的ISL路径质量.
        论文模式: 目的地=gateway feeder卫星; 否则=配对用户卫星."""
        res = 0.0
        gw = getattr(u, 'assigned_gateway', None)
        feeder_sat = gw.connected_sat[0] if (gw is not None and len(gw.connected_sat) > 0 and gw.connected_sat[0] is not None) else None

        # 论文gateway场景: 无配对用户, 直连feeder
        if len(u.user_to_connect_to) == 0 and len(u.user_to_connect_by) == 0 and feeder_sat is not None:
            if s == feeder_sat:
                res = DIRECT_PATH_QUALITY
            elif self.net.SPT[s.con_id-1][s.ID-1][feeder_sat.ID-1].isReached:
                res = self.net.N2N_status[s.con_id-1][s.ID-1][feeder_sat.ID-1].load_rate
            else:
                res = 0.0
            return res

        for user in u.user_to_connect_to:
            if feeder_sat is not None:
                temp_sat = feeder_sat  # 论文: ISL路径到gateway
            elif user.sat_connected is not None:
                temp_sat = user.sat_connected
            else:
                temp_sat = None

            if temp_sat is not None:
                if s==temp_sat:
                    res += DIRECT_PATH_QUALITY
                elif self.Effective_IPQ_Mode() == "pldr_lifetime":
                    res += self.Calc_PLDR_Path_Quality(s, temp_sat)
                else:
                    res += self.net.N2N_status[s.con_id-1][s.ID-1][temp_sat.ID-1].load_rate
            #如果目的终端没有入网，则接入一颗负载最低的卫星，定义为4条ISL的负载率乘积
            else:
                temp = 1.0
                for i in range(4):
                    lsa = self.net.LSDB[s.con_id-1][s.ID-1][i]
                    if lsa.isEstablished and lsa.total_band > 0:
                        temp = temp * (lsa.total_band - lsa.used_band) / lsa.total_band
                res += temp
        return res

    def Effective_IPQ_Mode(self):
        if IPQ_LOAD_THRESHOLD > 0 and len(self.topo.user) < IPQ_LOAD_THRESHOLD:
            return "n2n_load"
        return IPQ_MODE

    def Effective_Handover_Cost(self):
        if HANDOVER_COST_HIGH_LOAD is not None and HANDOVER_COST_LOAD_THRESHOLD > 0 \
                and len(self.topo.user) >= HANDOVER_COST_LOAD_THRESHOLD:
            return HANDOVER_COST_HIGH_LOAD
        return HANDOVER_COST

    #计算源终端到目的终端的delay
    def Calc_PLDR_Link_Quality(self, source: Satellite, dest_id: int):
        link = self.net.Lookup_LSA(source.con_id, source.ID, dest_id)
        if link == None or link.total_band <= 0:
            return 0.0
        free_ratio = max(0.0, min(1.0, (link.total_band - link.used_band) / link.total_band))
        con = self.topo.constellation[source.con_id-1]
        same_orbit_neighbor = abs(source.ID - dest_id) in (1, con.sat_per_orbit - 1)
        lifetime = 1.0
        if not same_orbit_neighbor:
            dest = self.topo.satellite[dest_id-1]
            margin_source = max(0.0, (THRESHOLD - abs(source.we_pos.lat)) / THRESHOLD)
            margin_dest = max(0.0, (THRESHOLD - abs(dest.we_pos.lat)) / THRESHOLD)
            lifetime = min(1.0, margin_source, margin_dest)
        return pow(free_ratio, PLDR_FREE_ALPHA) * pow(lifetime, PLDR_LIFETIME_ALPHA)

    def Calc_PLDR_Path_Quality(self, source: Satellite, dest: Satellite):
        if source == dest:
            return DIRECT_PATH_QUALITY
        con = source.con_id
        if self.net.SPT[con-1][source.ID-1][dest.ID-1].isReached == False:
            return 0.0
        quality = 1.0
        current = dest.ID
        sat_count = self.topo.constellation[con-1].orbit_num * self.topo.constellation[con-1].sat_per_orbit
        guard = 0
        while current != source.ID and guard < sat_count:
            pre = self.net.SPT[con-1][source.ID-1][current-1].pre
            if pre <= 0:
                return 0.0
            quality *= self.Calc_PLDR_Link_Quality(self.topo.satellite[pre-1], current)
            current = pre
            guard += 1
        return quality

    def Calc_Path_Distance(self, source: Satellite, dest: Satellite):
        if source == dest:
            return 0.0
        con = source.con_id
        if self.net.SPT[con-1][source.ID-1][dest.ID-1].isReached == False:
            return 0.0
        distance = 0.0
        current = dest.ID
        sat_count = self.topo.constellation[con-1].orbit_num * self.topo.constellation[con-1].sat_per_orbit
        guard = 0
        while current != source.ID and guard < sat_count:
            pre = self.net.SPT[con-1][source.ID-1][current-1].pre
            if pre <= 0:
                return 0.0
            distance += Calc_Sphere_Distance(self.topo.satellite[pre-1].we_pos, self.topo.satellite[current-1].we_pos)
            current = pre
            guard += 1
        return distance

    def Calc_Path_Hops(self, source: Satellite, dest: Satellite):
        if source == dest:
            return 0
        con = source.con_id
        if self.net.SPT[con-1][source.ID-1][dest.ID-1].isReached == False:
            return 0
        current = dest.ID
        sat_count = self.topo.constellation[con-1].orbit_num * self.topo.constellation[con-1].sat_per_orbit
        hops = 0
        while current != source.ID and hops < sat_count:
            pre = self.net.SPT[con-1][source.ID-1][current-1].pre
            if pre <= 0:
                return 0
            current = pre
            hops += 1
        return hops

    def Calc_Delay(self):
        return self.Calc_Delay_Components()["total_ms"]

    def Calc_Delay_Components(self):
        source_gsl_distance = 0.0
        isl_distance = 0.0
        destination_gsl_distance = 0.0
        legacy_hop_isl_distance = 0.0
        length_per_hop = (
            self.topo.constellation[0].orbit_sat[0].orbit_radius
            * 2
            * pi
            / self.topo.constellation[0].sat_per_orbit
        )
        count = 0
        for u in self.topo.user:
            if u.user_ID > len(self.topo.user)/2:
                break
            gw = getattr(u, 'assigned_gateway', None)
            if gw is not None and gw.connected_sat[0] is not None:
                # 走 gateway 落地路径
                feeder_sat = gw.connected_sat[0]
                if u.sat_connected is not None:
                    if u.sat_connected != feeder_sat:
                        isl_distance += self.Calc_Path_Distance(u.sat_connected, feeder_sat)
                        legacy_hop_isl_distance += (
                            self.Calc_Path_Hops(u.sat_connected, feeder_sat) * length_per_hop
                        )
                    source_gsl_distance += Calc_Sphere_Distance(u.we_pos, u.sat_connected.we_pos)
                    destination_gsl_distance += Calc_Sphere_Distance(gw.we_pos, feeder_sat.we_pos)
                    count += 1
            else:
                # 原有逻辑: user-to-user path
                for user in u.user_to_connect_to:
                    if user.sat_connected!=None:
                        temp_sat = Satellite()
                        temp_sat = user.sat_connected
                        if u.sat_connected!=temp_sat:
                            isl_distance += self.Calc_Path_Distance(u.sat_connected, temp_sat)
                            legacy_hop_isl_distance += (
                                self.Calc_Path_Hops(u.sat_connected, temp_sat) * length_per_hop
                            )
                        source_gsl_distance += Calc_Sphere_Distance(u.we_pos,u.sat_connected.we_pos)
                        destination_gsl_distance += Calc_Sphere_Distance(user.we_pos,user.sat_connected.we_pos)
                        count = count + 1
        return delay_components_ms(
            source_gsl_distance,
            isl_distance,
            destination_gsl_distance,
            count,
            legacy_hop_isl_distance,
        )


    #输出切换参数到文本
    def Record_Hoparam(self, mode:str):
        if mode == 'NETWORK_INITIAL':
            return
        if 1>HO_LOG_LEVEL:
            pout.write("Time:%d\n"%self.topo.current_time)
            pout.write("%-8s%-16s\n"%("User","Handover_Parameter"))
            for user in self.ho:
                pout.write("%-8d"%user.user_ID)
                for sat in self.ho[user]:
                    pout.write("s%d:"%sat.ID)
                    
                    if mode=='CHANNEL_QUALITY':
                        pout.write("%.5f"%self.ho[user][sat].c_quality)
                    elif mode=='DISTANCE':
                        pout.write("%.5f"%self.ho[user][sat].distance)
                    elif mode=='ELEVATION':
                        pout.write("%.5f"%self.ho[user][sat].elevation)
                    elif mode=='RATE_INTEGRAL':
                        pout.write("%.5f"%self.ho[user][sat].rate_integral)
                    elif mode=='SERVICE_TIME':
                        pout.write("%.5f"%self.ho[user][sat].s_time)
                    elif mode=='CHANNEL_RESOURCE':
                        pout.write("%.5f"%self.ho[user][sat].c_resource)
                    elif mode=='NETWORK_LOAD':
                        pout.write("%.5f"%self.ho[user][sat].available_band)
                    elif mode=='UNION_MODE_1':
                        pass
                    elif mode=='UNION_MODE_2':
                        pass
                    else:
                        pass
                    pout.write(" ")
                pout.write("\n")
            # pout.flush()

    def Record_Hotimes(self):
        self.statics[0].append(self.ho_count)

    def Record_Delay(self):
        components = self.Calc_Delay_Components()
        self.delay_component_samples.append(components)
        delay = components["total_ms"]
        self.statics[1].append(delay)
        dlout.write("Time:%d delay: %.2f\n"%(self.topo.current_time, delay))

    def Record_Throughput(self):
        total = 0
        count = 0
        for user in self.topo.user:
            for u in user.user_to_connect_to:
                total += user.user_to_connect_to[u]
                count +=1
        self.statics[3].append(total/count)
        tpout.write("Time:%d Total:%d Average:%d\n"%(self.topo.current_time, total, total/count) )

    def Record_Allocate(self):
        total = 0
        count = 0
        for user in self.topo.user:
            for u in user.allocate_band:
                total += user.allocate_band[u]
                count +=1
        self.statics[2].append(total/count)
        tpout2.write("Time:%d Total:%d Average:%d\n"%(self.topo.current_time, total, total/count) )
        #tpout2.flush()
        
    def Record_Statics(self):
        sout.write("Sat:%d User:%d\n"%(self.topo.constellation[0].orbit_num*self.topo.constellation[0].sat_per_orbit,len(self.topo.user)))
        for i in range(len(self.statics)):
            for j in range(len(self.statics[i])):
                sout.write("%.6f "%self.statics[i][j])
            sout.write("\n")
        sout.write("\n")


    def reset(self,time,mode:str):
        self.net.Initial_Network()
        self.Initial_GSlink()
        self.topo.Each_Satellite()
        self.block_count=0
        self.ho_count=0
        self.Update_Env(time,mode)
    
    def Update_Env(self,time,mode:str):
        self.topo.Update_Topology_Status(time)
        if self.topo.constellation[0].type == TYPE_PI:
            self.net.Update_LSDB_Netmode()
        if time==0.0:
            if self.topo.constellation[0].type == TYPE_2PI:
                self.net.Update_LSDB_Netmode()
            self.net.Dijkstra_All()
            self.net.Generate_Forwardingtable_By_AllNode()
        # self.net.reset_lsa_used_band()  # 不重置: v1.1路线, 物理reward不care FCFS死锁
        self.net.Link_LSDB_With_N2N_All()
        self.net.Update_N2N_Load_By_LSDB_All()  # 初始化N2N端到端带宽
        self._precompute_hops()  # 预计算跳数矩阵
        self.net.Update_NSA_Band_All()
        self.net.Record()
        self.Initial_Handover()
        self.Update_Channel_Resource()
        self.Update_Trans_Rate(time)  # 填充c_quality (修复:原初始化流程遗漏)
        self.Update_Available_Band()
        self.Update_Rate_Integral(time)


    def step(self, actions, mode:str):
        for user in actions:
            s = self.topo.satellite[actions[user]-1]
            if s not in self.ho[user]: continue  # DQN可能选不可见卫星
            self.Trig_Handover(s, user, mode)

    def Get_Reward(self, user:User):
        """杨论文Eq.50对齐: r = Cavai/RATE_UPPER - ω₁∆access - ω₂∆backhaul
           ω₁=1.5, ω₂=2.0"""
        C_NORM = RATE_UPPER
        if user.sat_connected is not None and user.sat_connected in self.ho[user]:
            access = self.ho[user][user.sat_connected].c_quality
            feeder = self._get_feeder_sat(user, user.sat_connected)
            fb_val = 9999.0
            if feeder is not None and user.sat_connected != feeder:
                fb_val = self.net.N2N_status[user.sat_connected.con_id-1][user.sat_connected.ID-1][feeder.ID-1].free_band
            cavai = min(access, fb_val)
            rate_reward = cavai / C_NORM
            # 接入切换惩罚: ω₁ (环境变量LEO_HO_PENALTY, 默认1.5)
            ho_penalty = 0.0
            if user.last_connected is not None and user.sat_connected != user.last_connected:
                ho_penalty = float(os.environ.get('LEO_HO_PENALTY', '0.2'))
            # 回传路径变化惩罚: ω₂ (环境变量LEO_BH_PENALTY, 默认0.2)
            bh_penalty = 0.0
            last_feeder = getattr(user, '_last_feeder', None)
            last_hops = getattr(user, '_last_feeder_hops', 0)
            if feeder is not None:
                curr_hops = self._hops_cache.get((user.sat_connected.ID, feeder.ID),
                              max(1, self.Calc_Path_Hops(user.sat_connected, feeder)))
                if last_feeder is not None and (feeder.ID != last_feeder or curr_hops != last_hops):
                    bh_penalty = float(os.environ.get('LEO_BH_PENALTY', '0.2'))
                user._last_feeder = feeder.ID
                user._last_feeder_hops = curr_hops
            return rate_reward - ho_penalty - bh_penalty
        return 0.0
    
    def _get_feeder_sat(self, user:User, source_sat=None):
        """选择跳数最少的gateway feeder卫星."""
        gw = getattr(user, 'assigned_gateway', None)
        if gw is None:
            return None
        best = None
        best_hops = 999
        for sat in gw.connected_sat:
            if sat is None:
                continue
            if source_sat is not None:
                h = self._hops_cache.get((source_sat.ID, sat.ID), self.Calc_Path_Hops(source_sat, sat))
            else:
                h = 1
            if h < best_hops:
                best_hops = h
                best = sat
        return best

    def _precompute_hops(self):
        """每集预计算所有sat→feeder的跳数(O(1)查表替代SPT遍历)."""
        self._hops_cache.clear()
        feeders = []
        for gw in self.topo.gateway:
            for s in gw.connected_sat:
                if s is not None:
                    feeders.append(s)
        for con in range(len(self.topo.constellation)):
            sat_num = self.topo.constellation[con].orbit_num * self.topo.constellation[con].sat_per_orbit
            for src_id in range(1, sat_num + 1):
                src = self.topo.satellite[src_id - 1]
                for fs in feeders:
                    if src == fs:
                        self._hops_cache[(src_id, fs.ID)] = 1
                    elif self.net.SPT[con][src_id - 1][fs.ID - 1].isReached:
                        self._hops_cache[(src_id, fs.ID)] = self.Calc_Path_Hops(src, fs)

    def _compute_rvt(self, user:User, sat, max_rvt=600, step=10):
        """计算user-sat之间的剩余可见时间(秒). 10s步长, 最大600s."""
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
        return rvt

    #RL - Yang论文 Eq.46: 3特征/星 (3N), 对齐论文
    def Observe_Yang(self,user:User,mode):
        N = self.topo.total_sat
        RATE_CAP = 1000.0

        prev_conn = [0.0] * N        # [0] x^u: 连接状态
        cavai     = [-1.0] * N       # [1] Cavai/C_norm
        rvt       = [0.0] * N        # [2] RVT/600

        if user.last_connected is not None:
            prev_conn[user.last_connected.ID - 1] = 1.0

        feeder_sat = self._get_feeder_sat(user)

        for sat in self.ho[user]:
            idx = sat.ID - 1
            if feeder_sat is not None:
                fb = self.net.N2N_status[sat.con_id - 1][sat.ID - 1][feeder_sat.ID - 1].free_band
                val = min(self.ho[user][sat].c_quality, fb)
            else:
                val = self.ho[user][sat].c_quality
            cavai[idx] = max(0.001, min(1.0, val / RATE_CAP))
            rvt[idx] = min(1.0, self._compute_rvt(user, sat) / 600.0)

        return prev_conn + cavai + rvt

    #RL - 5N state: Yang式(prev_conn+Cavai+RVT) + beam_load + 回传可靠性
    def Observe(self,user:User,mode):
        """可见星×F特征: [elevation, RVT, c_quality, ISL_free_band].
           全部归一化到[0,1]. 按卫星ID排序(不泄露质量信息).
           变长模式(LEO_VARLEN=1): 全部可见星, 不足补0. sat_ids供动作映射."""
        FEAT = int(os.environ.get('LEO_FEAT_PER_SAT', '4'))
        VARLEN = os.environ.get('LEO_VARLEN', '0') == '1'
        K = 30 if VARLEN else 12  # 变长模式: 最多30颗可见星

        features = []  # [(elev, rvt, cq, [isl_fb], [is_cur], sat_id), ...]
        cur_id = user.sat_connected.ID if user.sat_connected is not None else -1

        for sat in self.ho[user]:
            elev_raw = Calc_Sphere_Elevation(sat.we_pos, user.we_pos)
            elev = elev_raw * 2.0 / _pi                 # [0, 1]
            rvt  = min(1.0, self._compute_rvt(user, sat) / 600.0)
            cq   = min(1.0, self.ho[user][sat].c_quality / RATE_UPPER)
            feat = [elev, rvt, cq]
            if FEAT >= 4:
                fd = self._get_feeder_sat(user, sat)
                if fd is not None and sat != fd:
                    fb = self.net.N2N_status[sat.con_id - 1][sat.ID - 1][fd.ID - 1].free_band
                    feat.append(min(1.0, fb / RATE_UPPER))
                else:
                    feat.append(1.0)  # 直连feeder, 无ISL瓶颈
            if FEAT >= 5:
                feat.append(1.0 if sat.ID == cur_id else 0.0)
            feat.append(sat.ID)
            features.append(tuple(feat))

        # 按卫星ID排序 (不泄露质量信息, 网络需要自己学)
        features.sort(key=lambda x: x[-1])
        # 变长模式: 截断到K
        if len(features) > K:
            features = features[:K]

        state = []
        sat_ids = []
        for i in range(K):
            if i < len(features):
                feat = features[i]
                state.extend(feat[:-1])  # 前FEAT个值
                sat_ids.append(feat[-1])  # 最后是sat_id
            else:
                state.extend([0.0] * FEAT)
                sat_ids.append(0)

        user._topK_sat_ids = sat_ids
        return state
    
    def close(self):
        hsout.logger.critical("Handover success count:%d"%self.ho_count)
        hsout.logger.critical("Handover fail count:%d\n"%self.block_count)
        self.net.Record_Net_Statics(True)
