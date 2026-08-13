import os
from math import log2, pi

import numpy as np

from Defination import (
    AGC_PREDICT_WINDOW,
    BANDWIDTH,
    EARTH_RADIUS,
    HANDOVER_COST,
    SAT_BEAM,
    SAT_HEIGHT,
)
from Handover import Apply_Path_Delay_Weight, Handover, Is_GSL_Interference_Source
import Handover as handover_module
from Position import (
    Calc_Sphere_Distance,
)


LIGHT_VEL = 3.0e8
CARRIER = float(os.environ.get("LEO_PAPER_CARRIER_HZ", 2.0e10))
C_BAND = float(os.environ.get("LEO_C_BAND", 500.0))
GSL_RATE_CAP = float(os.environ.get("LEO_GSL_RATE_CAP", "0"))
GSL_RATE_CAP_HIGH_LOAD = float(os.environ.get("LEO_GSL_RATE_CAP_HIGH_LOAD", "0"))
GSL_RATE_CAP_LOAD_THRESHOLD = int(os.environ.get("LEO_GSL_RATE_CAP_LOAD_THRESHOLD", "0"))
GSL_RATE_CAP_POINTS = os.environ.get("LEO_GSL_RATE_CAP_POINTS", "").strip()
SQ_FREE_ALPHA = float(os.environ.get("LEO_PAPER_SQ_FREE_ALPHA", "1.0"))
PT_GT_GS_DB = float(os.environ.get("LEO_PAPER_PT_GT_GS_DB", 80.0))
PT_GT_GS = 10.0 ** (PT_GT_GS_DB / 10.0)
NOISE_DBM_PER_HZ = float(os.environ.get("LEO_PAPER_NOISE_DBM_HZ", -173.0))
NOISE_W_PER_HZ = 10.0 ** ((NOISE_DBM_PER_HZ - 30.0) / 10.0)
ATMOSPHERIC_LOSS_DB = float(os.environ.get("LEO_PAPER_ATMOSPHERIC_LOSS_DB", 2.9))
RICE_FACTOR = float(os.environ.get("LEO_PAPER_RICE_FACTOR", 1.0))
DIRECT_PATH_QUALITY = float(os.environ.get("LEO_PAPER_DIRECT_PATH_QUALITY", 2.0))
UTILITY_H = float(os.environ.get("LEO_PAPER_HANDOVER_COST", HANDOVER_COST))
UTILITY_H_POINTS = os.environ.get("LEO_PAPER_HANDOVER_COST_POINTS", "").strip()
AGC_DELTA = int(os.environ.get("LEO_PAPER_AGC_DELTA", AGC_PREDICT_WINDOW))
INCLUDE_INTERFERENCE = os.environ.get("LEO_PAPER_INCLUDE_INTERFERENCE", "1") == "1"
INTERFERENCE_SCALE = float(os.environ.get("LEO_PAPER_INTERFERENCE_SCALE", "1.0"))
HANDOVER_CONTROL_MODE = os.environ.get("LEO_PAPER_HANDOVER_CONTROL_MODE", "agc").strip().lower()
UTILITY_HYSTERESIS = float(os.environ.get("LEO_PAPER_UTILITY_HYSTERESIS", "0.0"))
UTILITY_HYSTERESIS_POINTS = os.environ.get("LEO_PAPER_UTILITY_HYSTERESIS_POINTS", "").strip()
CAHS_DELAY_WEIGHT = float(os.environ.get("LEO_PAPER_CAHS_DELAY_WEIGHT", "0.0"))
CHANNEL_QUALITY_AVG_WINDOW = int(os.environ.get("LEO_CHANNEL_QUALITY_AVG_WINDOW", "30"))
CHANNEL_QUALITY_AVG_SAMPLES = max(2, int(os.environ.get("LEO_CHANNEL_QUALITY_AVG_SAMPLES", "3")))


def paper_signal_power(distance_m: float) -> float:
    if distance_m <= 0:
        return 0.0
    free_space_gain = (LIGHT_VEL / (4.0 * pi * distance_m * CARRIER)) ** 2.0
    atmosphere_gain = 10.0 ** (-ATMOSPHERIC_LOSS_DB / 10.0)
    return PT_GT_GS * free_space_gain * atmosphere_gain * RICE_FACTOR


def paper_capacity_upper() -> float:
    noise = NOISE_W_PER_HZ * C_BAND * 1.0e6
    signal = paper_signal_power(SAT_HEIGHT)
    return C_BAND * log2(1.0 + signal / noise)


def cap_gsl_rate(rate: float) -> float:
    if GSL_RATE_CAP > 0:
        return min(rate, GSL_RATE_CAP)
    return rate


def parse_load_points(raw: str):
    points = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        users, value = item.split(":", 1)
        points.append((int(users.strip()), float(value.strip())))
    return sorted(points)


def interpolated_load_value(user_count: int, raw_points: str) -> float:
    points = parse_load_points(raw_points)
    if not points:
        return 0.0
    if user_count <= points[0][0]:
        return points[0][1]
    if user_count >= points[-1][0]:
        return points[-1][1]
    for (left_users, left_value), (right_users, right_value) in zip(points, points[1:]):
        if left_users <= user_count <= right_users:
            span = right_users - left_users
            if span <= 0:
                return right_value
            ratio = (user_count - left_users) / span
            return left_value + ratio * (right_value - left_value)
    return points[-1][1]


def effective_gsl_rate_cap(user_count: int) -> float:
    if GSL_RATE_CAP_POINTS:
        return interpolated_load_value(user_count, GSL_RATE_CAP_POINTS)
    if GSL_RATE_CAP_HIGH_LOAD > 0 and GSL_RATE_CAP_LOAD_THRESHOLD > 0:
        if user_count >= GSL_RATE_CAP_LOAD_THRESHOLD:
            return GSL_RATE_CAP_HIGH_LOAD
    return GSL_RATE_CAP


def cap_gsl_rate_for_load(rate: float, user_count: int) -> float:
    cap = effective_gsl_rate_cap(user_count)
    if cap > 0:
        return min(rate, cap)
    return rate


def paper_link_sending_quality(free_ratio: float) -> float:
    ratio = max(0.0, min(1.0, free_ratio))
    return ratio ** SQ_FREE_ALPHA


RATE_UPPER = paper_capacity_upper()


def paper_utility_weights(ipq_values, agc_values, h_weight):
    remaining = max(0.0, 1.0 - h_weight)
    if not ipq_values:
        return remaining, 0.0
    g_weight = remaining * float(np.mean(ipq_values))
    g_weight = max(0.0, min(remaining, g_weight))
    return remaining - g_weight, g_weight


def effective_utility_h(user_count: int = 0) -> float:
    if UTILITY_H_POINTS:
        return interpolated_load_value(user_count, UTILITY_H_POINTS)
    return UTILITY_H


def effective_utility_hysteresis(user_count: int = 0) -> float:
    if UTILITY_HYSTERESIS_POINTS:
        return interpolated_load_value(user_count, UTILITY_HYSTERESIS_POINTS)
    return UTILITY_HYSTERESIS


def apply_utility_hysteresis(current_sat, target_sat, current_value, target_value, user_count: int = 0):
    hysteresis = effective_utility_hysteresis(user_count)
    if hysteresis <= 0.0:
        return target_sat
    if current_sat is None or target_sat is None or current_sat == target_sat:
        return target_sat
    if target_value <= current_value + hysteresis:
        return current_sat
    return target_sat


class PaperRebuildHandover(Handover):
    """Paper-first CAHS model used only by run_paper_rebuild.py."""

    def Calc_Signal_Power(self, s, u, time=-1.0):
        if time == -1.0:
            distance = Calc_Sphere_Distance(s.we_pos, u.we_pos)
        else:
            distance = Calc_Sphere_Distance(s.Get_Satellite_We_Condition(time), u.we_pos)
        return paper_signal_power(distance)

    def Calc_Download_Noise(self, s, u, time=-1.0):
        noise = NOISE_W_PER_HZ * C_BAND * 1.0e6
        if not INCLUDE_INTERFERENCE:
            return noise
        for other_user in s.user_connected:
            if other_user != u and Is_GSL_Interference_Source(other_user):
                noise += INTERFERENCE_SCALE * self.Calc_Signal_Power(s, other_user, time)
        return noise

    def Calc_Trans_Rate(self, time, s, u):
        signal = self.Calc_Signal_Power(s, u, time)
        noise = self.Calc_Download_Noise(s, u, time)
        return cap_gsl_rate_for_load(C_BAND * log2(1.0 + signal / noise), len(self.topo.user))

    def Calc_Channel_Quality_Decision(self, s, u):
        if handover_module.CHANNEL_QUALITY_DECISION_NOISE == "average":
            times = np.linspace(
                0.0,
                float(CHANNEL_QUALITY_AVG_WINDOW),
                CHANNEL_QUALITY_AVG_SAMPLES,
            )
            rates = [self.Calc_Trans_Rate(self.topo.current_time + t, s, u) for t in times]
            return float(np.mean(rates))
        if handover_module.CHANNEL_QUALITY_DECISION_NOISE != "thermal":
            return self.ho[u][s].c_quality
        signal = self.Calc_Signal_Power(s, u, self.topo.current_time)
        noise = NOISE_W_PER_HZ * C_BAND * 1.0e6
        return cap_gsl_rate_for_load(C_BAND * log2(1.0 + signal / noise), len(self.topo.user))

    def Update_Rate_Integral(self, time):
        for user in self.ho:
            for sat in self.ho[user]:
                duration = min(self.ho[user][sat].s_time, AGC_DELTA)
                if duration <= 0:
                    self.ho[user][sat].rate_integral = 0.0
                    continue
                integral = self.Calc_Rate_Integral(time, sat, user)
                self.ho[user][sat].rate_integral = integral / max(RATE_UPPER * duration, 1.0e-9)

    def Calc_Rate_Integral(self, begin_t, s, u):
        end_time = min(self.ho[u][s].s_time, AGC_DELTA)
        if end_time <= 0:
            return 0.0
        samples = max(2, int(os.environ.get("LEO_PAPER_AGC_SAMPLES", "5")))
        times = np.linspace(0.0, float(end_time), samples)
        rates = [self.Calc_Trans_Rate(begin_t + t, s, u) for t in times]
        return float(np.trapz(rates, times))

    def Calc_Path_Quality_Product(self, source, dest):
        if source == dest:
            return DIRECT_PATH_QUALITY
        con = source.con_id
        if not self.net.SPT[con - 1][source.ID - 1][dest.ID - 1].isReached:
            return 0.0
        quality = 1.0
        current = dest.ID
        sat_count = self.topo.constellation[con - 1].orbit_num * self.topo.constellation[con - 1].sat_per_orbit
        guard = 0
        while current != source.ID and guard < sat_count:
            pre = self.net.SPT[con - 1][source.ID - 1][current - 1].pre
            if pre <= 0:
                return 0.0
            link = self.net.Lookup_LSA(con, pre, current)
            if link is None or link.total_band <= 0:
                return 0.0
            free = max(0.0, link.total_band - link.used_band)
            quality *= paper_link_sending_quality(free / link.total_band)
            current = pre
            guard += 1
        return quality

    def Calc_Available_Band(self, s, u):
        total = 0.0
        count = 0

        # 论文gateway场景: 无配对用户, 直连feeder
        gw = getattr(u, 'assigned_gateway', None)
        feeder_sat = gw.connected_sat[0] if (gw is not None and len(gw.connected_sat) > 0 and gw.connected_sat[0] is not None) else None
        if len(u.user_to_connect_to) == 0 and len(u.user_to_connect_by) == 0 and feeder_sat is not None:
            if s == feeder_sat:
                return 0.0  # direct path quality is best
            elif self.net.SPT[s.con_id-1][s.ID-1][feeder_sat.ID-1].isReached:
                return self.net.N2N_status[s.con_id-1][s.ID-1][feeder_sat.ID-1].load_rate
            return 0.0

        for user in u.user_to_connect_to:
            if user.sat_connected is not None:
                total += self.Calc_Path_Quality_Product(s, user.sat_connected)
                count += 1
            else:
                adjacent_quality = 1.0
                adjacent_count = 0
                for link in self.net.LSDB[s.con_id - 1][s.ID - 1]:
                    if not link.isEstablished or link.total_band <= 0:
                        continue
                    adjacent_quality *= max(0.0, link.total_band - link.used_band) / link.total_band
                    adjacent_count += 1
                total += adjacent_quality if adjacent_count else 0.0
                count += 1
        return total / count if count else 0.0

    def Trig_Decision(self, mode, isNet, user):
        if mode != "UNION_MODE_1":
            return super().Trig_Decision(mode, isNet, user)

        max_value = -1.0e18
        target = None
        current_value = None
        h_weight = effective_utility_h(len(self.topo.user))
        ipq_values = []
        agc_values = []
        for sat in self.ho[user]:
            self.ho[user][sat].available_band = self.Calc_Available_Band(sat, user)
            ipq_values.append(self.ho[user][sat].available_band)
            agc_values.append(self.ho[user][sat].rate_integral)

        f_weight, g_weight = paper_utility_weights(ipq_values, agc_values, h_weight)
        self.weight_samples.append((f_weight, g_weight, h_weight))

        for sat in self.ho[user]:
            if user.sat_connected is not None and user.sat_connected != sat:
                handover_control = self.Calc_Handover_Control(sat, user)
            elif user.sat_connected is None:
                handover_control = self.Calc_Handover_Control(sat, user)
            else:
                handover_control = 0.0
            value = (
                f_weight * self.ho[user][sat].available_band
                + g_weight * self.ho[user][sat].rate_integral
                + h_weight * handover_control
            )
            value = Apply_Path_Delay_Weight(
                value,
                self.ho[user][sat].delay,
                CAHS_DELAY_WEIGHT,
                100.0,
            )
            self.ho[user][sat].value = value
            if user.sat_connected is not None and sat == user.sat_connected:
                current_value = value
            if value > max_value and (
                sat.beam - len(sat.user_connected) > 0
                or (user.sat_connected is not None and sat == user.sat_connected)
            ):
                max_value = value
                target = sat
        if current_value is not None:
            target = apply_utility_hysteresis(
                user.sat_connected,
                target,
                current_value,
                max_value,
                len(self.topo.user),
            )
        self.Trig_Handover(target, user, "NETWORK" if isNet else "OTHERS")

    def Calc_Handover_Control(self, sat, user):
        if HANDOVER_CONTROL_MODE == "none":
            return 0.0
        if HANDOVER_CONTROL_MODE == "constant":
            return -1.0
        if HANDOVER_CONTROL_MODE == "source":
            return -self.ho[user][sat].rate_integral
        return -self.ho[user][sat].rate_integral
