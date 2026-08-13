import argparse
import csv
import os
import random
import sys
from math import acos, cos, pi, sin
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("LEO_CAHS_VARIANT", "paper")
os.environ.setdefault("LEO_QUIET_LOGS", "1")
os.environ.setdefault("LEO_C_BAND", "500")
PAUSE_FILE = BASE_DIR / "EXPERIMENTS_PAUSED"
if PAUSE_FILE.exists() or os.environ.get("LEO_EXPERIMENTS_PAUSED", "0") == "1":
    print(f"experiments paused: remove {PAUSE_FILE} to run", file=sys.stderr)
    sys.exit(2)

from Defination import SAT_HEIGHT, TYPE_2PI
from Network import Network
from Position import Convert_Angle_to_PI
from Topology import Topology
from User import User
from paper_model import (
    PaperRebuildHandover,
    UTILITY_H,
    apply_utility_hysteresis,
    paper_utility_weights,
)


def reset_ids() -> None:
    User.uid = 0


def make_constellation(topo: Topology, constellation: str) -> None:
    if constellation == "A":
        orbit_num, sat_per_orbit = 8, 9
    elif constellation == "B":
        orbit_num, sat_per_orbit = 12, 12
    else:
        raise ValueError("constellation must be A or B")
    phase = 1
    first_phi = 2.0 * phase * pi / (orbit_num * sat_per_orbit)
    lean = 54.0 / 180.0 * pi
    theta = 2.0 * pi / orbit_num
    topo.Add_Constellation(orbit_num, sat_per_orbit, SAT_HEIGHT, first_phi, lean, theta, TYPE_2PI)
    topo.Each_Satellite()


def make_grid_locations(count: int) -> list[list[float]]:
    lat_start, lat_end = -60.0, 60.0
    lon_start, lon_end = -120.0, 120.0
    rows = min(10, max(1, int(round(count ** 0.5))))
    while rows > 1 and count % rows != 0:
        rows -= 1
    cols = count // rows
    lat_step = (lat_end - lat_start) / rows
    lon_step = (lon_end - lon_start) / cols
    return [
        [Convert_Angle_to_PI(lon_start + i * lon_step), Convert_Angle_to_PI(lat_start + j * lat_step)]
        for j in range(rows)
        for i in range(cols)
    ]


def make_random_locations(count: int) -> list[list[float]]:
    lon_min = float(os.environ.get("LEO_SOURCE_LON_MIN", "-180.0"))
    lon_max = float(os.environ.get("LEO_SOURCE_LON_MAX", "180.0"))
    lat_min = float(os.environ.get("LEO_SOURCE_LAT_MIN", "-60.0"))
    lat_max = float(os.environ.get("LEO_SOURCE_LAT_MAX", "60.0"))
    return [
        [
            Convert_Angle_to_PI(random.uniform(lon_min, lon_max)),
            Convert_Angle_to_PI(random.uniform(lat_min, lat_max)),
        ]
        for _ in range(count)
    ]


def make_user_locations(user_count: int, traffic_mode: str, gateway_count: int) -> list[list[float]]:
    if user_count % 2 != 0:
        raise ValueError("user_count must be even so source/destination pairs match")
    if traffic_mode == "ue_pair":
        return make_grid_locations(user_count)
    if traffic_mode != "ground_backbone":
        raise ValueError("traffic_mode must be ue_pair or ground_backbone")

    half = user_count // 2
    source_layout = os.environ.get("LEO_SOURCE_LAYOUT", "grid").strip().lower()
    sources = make_random_locations(half) if source_layout == "random" else make_grid_locations(half)
    gateway_count = max(1, gateway_count)
    gateway_layout = os.environ.get("LEO_GATEWAY_LAYOUT", "left_open").strip().lower()
    if gateway_layout == "endpoints" and gateway_count > 1:
        gateway_lons = [-120.0 + i * (240.0 / (gateway_count - 1)) for i in range(gateway_count)]
    else:
        gateway_lons = [-120.0 + i * (240.0 / gateway_count) for i in range(gateway_count)]
    gateway_lat = float(os.environ.get("LEO_GATEWAY_LAT", "0.0"))
    gateway_points = [
        [Convert_Angle_to_PI(lon), Convert_Angle_to_PI(gateway_lat)]
        for lon in gateway_lons
    ]
    gateway_assign = os.environ.get("LEO_GATEWAY_ASSIGN", "cycle").strip().lower()
    gateways = []
    for source in sources:
        if gateway_assign == "nearest":
            src_lon, src_lat = source

            def angular_distance(point: list[float]) -> float:
                gw_lon, gw_lat = point
                value = sin(src_lat) * sin(gw_lat) + cos(src_lat) * cos(gw_lat) * cos(src_lon - gw_lon)
                return acos(max(-1.0, min(1.0, value)))

            gateways.append(min(gateway_points, key=angular_distance))
        elif gateway_assign == "random":
            gateways.append(gateway_points[random.randrange(gateway_count)])
        else:
            gateways.append(gateway_points[len(gateways) % gateway_count])
    return sources + gateways


class DiagnosticPaperHandover(PaperRebuildHandover):
    def __init__(self, *args, sample_limit: int = 5000, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_limit = sample_limit
        self.decision_rows = []

    def Trig_Decision(self, mode, isNet, user):
        if mode != "UNION_MODE_1":
            return super().Trig_Decision(mode, isNet, user)

        candidates = []
        ipq_values = []
        agc_values = []
        for sat in self.ho[user]:
            self.ho[user][sat].available_band = self.Calc_Available_Band(sat, user)
            ipq = float(self.ho[user][sat].available_band)
            agc = float(self.ho[user][sat].rate_integral)
            ipq_values.append(ipq)
            agc_values.append(agc)
            candidates.append((sat, ipq, agc))

        f_weight, g_weight = paper_utility_weights(ipq_values, agc_values, UTILITY_H)
        self.weight_samples.append((f_weight, g_weight, UTILITY_H))

        scored = []
        for sat, ipq, agc in candidates:
            if user.sat_connected is not None and user.sat_connected != sat:
                handover_control = self.Calc_Handover_Control(sat, user)
            elif user.sat_connected is None:
                handover_control = self.Calc_Handover_Control(sat, user)
            else:
                handover_control = 0.0
            value = f_weight * ipq + g_weight * agc + UTILITY_H * handover_control
            self.ho[user][sat].value = value
            allowed = sat.beam - len(sat.user_connected) > 0 or (
                user.sat_connected is not None and sat == user.sat_connected
            )
            scored.append((value, sat, ipq, agc, handover_control, allowed))

        scored_allowed = [row for row in scored if row[5]]
        target_row = max(scored_allowed, key=lambda row: row[0]) if scored_allowed else None
        target = target_row[1] if target_row else None
        current_row = next(
            (row for row in scored if user.sat_connected is not None and row[1] == user.sat_connected),
            None,
        )
        if current_row is not None and target_row is not None:
            target = apply_utility_hysteresis(
                user.sat_connected,
                target,
                current_row[0],
                target_row[0],
                len(self.topo.user),
            )
        if len(self.decision_rows) < self.sample_limit and scored:
            sorted_by_value = sorted(scored, key=lambda row: row[0], reverse=True)
            selected_rank = next(
                (idx + 1 for idx, row in enumerate(sorted_by_value) if row[1] == target),
                None,
            )
            selected = next((row for row in scored if row[1] == target), None)
            self.decision_rows.append(
                {
                    "time": self.topo.current_time,
                    "user": user.user_ID,
                    "candidate_count": len(scored),
                    "selected_rank": selected_rank or 0,
                    "f_weight": f_weight,
                    "g_weight": g_weight,
                    "h_weight": UTILITY_H,
                    "mean_ipq": float(np.mean(ipq_values)) if ipq_values else 0.0,
                    "max_ipq": float(np.max(ipq_values)) if ipq_values else 0.0,
                    "min_ipq": float(np.min(ipq_values)) if ipq_values else 0.0,
                    "mean_agc": float(np.mean(agc_values)) if agc_values else 0.0,
                    "max_agc": float(np.max(agc_values)) if agc_values else 0.0,
                    "selected_ipq": selected[2] if selected else 0.0,
                    "selected_agc": selected[3] if selected else 0.0,
                    "selected_value": selected[0] if selected else 0.0,
                    "selected_handover_control": selected[4] if selected else 0.0,
                    "raw_best_value": target_row[0] if target_row else 0.0,
                    "current_value": current_row[0] if current_row else 0.0,
                    "kept_current": 1 if current_row is not None and target == user.sat_connected else 0,
                }
            )
        self.Trig_Handover(target, user, "NETWORK" if isNet else "OTHERS")


def build_env(
    user_count: int,
    constellation: str,
    traffic_mode: str,
    gateway_count: int,
    sample_limit: int,
) -> DiagnosticPaperHandover:
    reset_ids()
    topo = Topology()
    make_constellation(topo, constellation)
    topo.Add_User_From_Input(make_user_locations(user_count, traffic_mode, gateway_count))
    half = len(topo.user) // 2
    for i in range(half):
        topo.user[i].User_Connect_User(topo.user[i + half], "UPLOAD")
    return DiagnosticPaperHandover(net=Network(topo), sample_limit=sample_limit)


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "time",
        "user",
        "candidate_count",
        "selected_rank",
        "f_weight",
        "g_weight",
        "h_weight",
        "mean_ipq",
        "max_ipq",
        "min_ipq",
        "mean_agc",
        "max_agc",
        "selected_ipq",
        "selected_agc",
        "selected_value",
        "selected_handover_control",
        "raw_best_value",
        "current_value",
        "kept_current",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constellation", choices=["A", "B"], default="B")
    parser.add_argument("--users", type=int, default=600)
    parser.add_argument("--slots", type=int, default=20)
    parser.add_argument("--traffic-mode", choices=["ue_pair", "ground_backbone"], default="ground_backbone")
    parser.add_argument("--gateway-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("results") / "cahs_decision_diagnostics.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = build_env(args.users, args.constellation, args.traffic_mode, args.gateway_count, args.sample_limit)
    env.Run_Network_Handover(0, args.slots * 30, 30, "UNION_MODE_1")
    out_path = BASE_DIR / args.out
    write_rows(out_path, env.decision_rows)
    print(out_path)


if __name__ == "__main__":
    main()
