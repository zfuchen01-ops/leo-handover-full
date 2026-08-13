import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from math import acos, cos, pi, sin
from multiprocessing import get_context
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

from Defination import AGC_PREDICT_WINDOW, BANDWIDTH, HANDOVER_COST, SAT_HEIGHT, TYPE_2PI
from Network import Network
from Position import Convert_Angle_to_PI
from Topology import Topology
from User import User
from paper_model import (
    AGC_DELTA,
    ATMOSPHERIC_LOSS_DB,
    C_BAND,
    DIRECT_PATH_QUALITY,
    GSL_RATE_CAP,
    GSL_RATE_CAP_HIGH_LOAD,
    GSL_RATE_CAP_LOAD_THRESHOLD,
    GSL_RATE_CAP_POINTS,
    HANDOVER_CONTROL_MODE,
    INCLUDE_INTERFERENCE,
    INTERFERENCE_SCALE,
    NOISE_DBM_PER_HZ,
    PT_GT_GS_DB,
    RATE_UPPER,
    SQ_FREE_ALPHA,
    UTILITY_H,
    UTILITY_H_POINTS,
    UTILITY_HYSTERESIS,
    UTILITY_HYSTERESIS_POINTS,
    PaperRebuildHandover,
)


METHODS = {
    "RS": "RANDOM",
    "MSTS": "SERVICE_TIME",
    "MGCS": "CHANNEL_QUALITY",
    "CAHS": "UNION_MODE_1",
    "UNION_MODE_1": "UNION_MODE_1",
    "DELAY": "DELAY",
    "NETWORK_LOAD": "NETWORK_LOAD",
    "RATE_INTEGRAL": "RATE_INTEGRAL",
}


def ensure_dirs(base: Path) -> None:
    for name in [
        "log/handover",
        "log/network",
        "log/topology",
        "log/RL",
        "log/model",
        "log/hyper",
        "results",
    ]:
        (base / name).mkdir(parents=True, exist_ok=True)


def reset_ids() -> None:
    Topology.index_con = 0
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
    topo.Add_Constellation(
        orbit_num, sat_per_orbit, SAT_HEIGHT, first_phi, lean, theta, TYPE_2PI
    )
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
    if gateway_layout == "reference6":
        gateway_lons = [130.361634, 86.15528, 110.47, 108.963122, 116.397128, 75.944]
        gateway_lats = [46.809606, 41.76602, 19.938, 34.90892, 39.916527, 39.468]
        gateway_weights = [13, 17, 10, 9, 7, 4]
        gateway_count = len(gateway_lons)
    elif gateway_layout == "endpoints" and gateway_count > 1:
        gateway_lons = [-120.0 + i * (240.0 / (gateway_count - 1)) for i in range(gateway_count)]
        gateway_lats = None
        gateway_weights = None
    else:
        gateway_lons = [-120.0 + i * (240.0 / gateway_count) for i in range(gateway_count)]
        gateway_lats = None
        gateway_weights = None
    gateway_lat = float(os.environ.get("LEO_GATEWAY_LAT", "0.0"))
    gateway_points = [
        [
            Convert_Angle_to_PI(gateway_lons[i]),
            Convert_Angle_to_PI(gateway_lats[i] if gateway_lats else gateway_lat),
        ]
        for i in range(gateway_count)
    ]
    gateway_assign = os.environ.get("LEO_GATEWAY_ASSIGN", "cycle").strip().lower()
    weighted_points = []
    if gateway_weights:
        for point, weight in zip(gateway_points, gateway_weights):
            weighted_points.extend([point] * weight)
    gateways = []
    for i, source in enumerate(sources):
        if gateway_assign == "nearest":
            src_lon, src_lat = source
            def angular_distance(point: list[float]) -> float:
                gw_lon, gw_lat = point
                value = sin(src_lat) * sin(gw_lat) + cos(src_lat) * cos(gw_lat) * cos(src_lon - gw_lon)
                return acos(max(-1.0, min(1.0, value)))
            gateways.append(min(gateway_points, key=angular_distance))
        elif gateway_assign == "random":
            gateways.append(gateway_points[random.randrange(gateway_count)])
        elif gateway_assign == "antenna_weighted" and weighted_points:
            gateways.append(weighted_points[i % len(weighted_points)])
        elif gateway_assign == "antenna_random" and weighted_points:
            gateways.append(weighted_points[random.randrange(len(weighted_points))])
        else:
            gateways.append(gateway_points[i % gateway_count])
    return sources + gateways


def build_env(user_count: int, constellation: str, traffic_mode: str, gateway_count: int) -> PaperRebuildHandover:
    os.environ["LEO_ACTIVE_USER_COUNT"] = str(user_count)
    reset_ids()
    topo = Topology()
    make_constellation(topo, constellation)

    # 添加地面站 (博士论文: 6个, 对应6个回传目的地)
    from math import pi as _pi
    gw_coords = [
        (130.361634, 46.809606, "佳木斯"),
        (86.15528, 41.76602, "库尔勒"),
        (110.47, 19.938, "文昌"),
        (108.963122, 34.90892, "铜川"),
        (116.397128, 39.916527, "雄安"),
        (75.944, 39.468, "喀什"),
    ]
    for lon_deg, lat_deg, name in gw_coords:
        topo.Add_Gateway_Loc(
            lon_deg / 180.0 * _pi,
            lat_deg / 180.0 * _pi,
            antenna_Num=1,
            name_str=name,
        )

    # 添加用户 (全部为源终端, 不需要fake gateway)
    all_users = make_user_locations(user_count, traffic_mode, gateway_count)
    # 如果 ground_backbone 模式且 ground station 未被禁用, 只取source部分
    if traffic_mode == "ground_backbone":
        half = user_count // 2
        source_locations = all_users[:half]  # 只取源终端
    else:
        source_locations = all_users
    topo.Add_User_From_Input(source_locations)

    # 每个用户分配一个地面站 (轮询)
    for i, user in enumerate(topo.user):
        user.assigned_gateway = topo.gateway[i % len(topo.gateway)]

    # 用户两两配对(保持源-目的配对结构, 但实际路由走gateway)
    half_u = len(topo.user) // 2
    for i in range(half_u):
        topo.user[i].User_Connect_User(topo.user[i + half_u], "UPLOAD")

    return PaperRebuildHandover(net=Network(topo))


def summarize(env: PaperRebuildHandover, user_count: int, method: str) -> dict[str, float | int | str]:
    handovers = np.asarray(env.statics[0], dtype=float)
    delay = np.asarray(env.statics[1], dtype=float)
    allocated = np.asarray(env.statics[2], dtype=float)
    requested = np.asarray(env.statics[3], dtype=float)
    flow_count = float(sum(len(user.user_to_connect_to) for user in env.topo.user))
    weights = (
        np.asarray(env.weight_samples, dtype=float)
        if getattr(env, "weight_samples", None)
        else np.empty((0, 3))
    )
    delay_components = getattr(env, "delay_component_samples", [])
    slots = len(handovers)
    source_handover_frequency = (
        float(env.source_ho_count / ((user_count / 2) * slots)) if slots and user_count else 0.0
    )
    destination_handover_frequency = (
        float(env.destination_ho_count / ((user_count / 2) * slots)) if slots and user_count else 0.0
    )
    handover_scope = os.environ.get("LEO_HANDOVER_SCOPE", "all").strip().lower()
    if handover_scope == "source":
        handover_frequency = source_handover_frequency
    elif handover_scope == "destination":
        handover_frequency = destination_handover_frequency
    else:
        handover_frequency = float(handovers[-1] / (user_count * slots)) if slots else 0.0
    return {
        "method": method,
        "users": user_count,
        "slots": len(delay),
        "handover_frequency": handover_frequency,
        "source_handover_frequency": source_handover_frequency,
        "destination_handover_frequency": destination_handover_frequency,
        "avg_delay_ms": float(np.mean(delay)) if len(delay) else 0.0,
        "avg_source_gsl_delay_ms": float(np.mean([row["source_gsl_ms"] for row in delay_components])) if delay_components else 0.0,
        "avg_isl_delay_ms": float(np.mean([row["isl_ms"] for row in delay_components])) if delay_components else 0.0,
        "avg_destination_gsl_delay_ms": float(np.mean([row["destination_gsl_ms"] for row in delay_components])) if delay_components else 0.0,
        "avg_legacy_hop_isl_delay_ms": float(np.mean([row["legacy_hop_isl_ms"] for row in delay_components])) if delay_components else 0.0,
        "avg_legacy_hop_total_delay_ms": float(np.mean([row["legacy_hop_total_ms"] for row in delay_components])) if delay_components else 0.0,
        "avg_allocated_mhz": float(np.mean(allocated) * flow_count) if len(allocated) else 0.0,
        "avg_requested_mhz": float(np.mean(requested) * flow_count) if len(requested) else 0.0,
        "last_allocated_mhz": float(allocated[-1] * flow_count) if len(allocated) else 0.0,
        "last_requested_mhz": float(requested[-1] * flow_count) if len(requested) else 0.0,
        "avg_allocated_bps": float(np.mean(allocated) * flow_count * 1.0e6) if len(allocated) else 0.0,
        "avg_requested_bps": float(np.mean(requested) * flow_count * 1.0e6) if len(requested) else 0.0,
        "last_allocated_bps": float(allocated[-1] * flow_count * 1.0e6) if len(allocated) else 0.0,
        "last_requested_bps": float(requested[-1] * flow_count * 1.0e6) if len(requested) else 0.0,
        "avg_f_weight": float(np.mean(weights[:, 0])) if len(weights) else 0.0,
        "avg_g_weight": float(np.mean(weights[:, 1])) if len(weights) else 0.0,
        "avg_h_weight": float(np.mean(weights[:, 2])) if len(weights) else 0.0,
        "weight_samples": int(len(weights)),
    }


def run_one(job: tuple[int, str, str, int, int, str, int]) -> dict[str, float | int | str]:
    users, constellation, method, slots, seed, traffic_mode, gateway_count = job
    random.seed(seed)
    np.random.seed(seed)
    env = build_env(users, constellation, traffic_mode, gateway_count)
    env.Run_Network_Handover(0, slots * 30, 30, METHODS[method])
    return summarize(env, users, method)


def write_csv(rows: list[dict[str, float | int | str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "users",
        "slots",
        "handover_frequency",
        "source_handover_frequency",
        "destination_handover_frequency",
        "avg_delay_ms",
        "avg_source_gsl_delay_ms",
        "avg_isl_delay_ms",
        "avg_destination_gsl_delay_ms",
        "avg_legacy_hop_isl_delay_ms",
        "avg_legacy_hop_total_delay_ms",
        "avg_allocated_mhz",
        "avg_requested_mhz",
        "last_allocated_mhz",
        "last_requested_mhz",
        "avg_allocated_bps",
        "avg_requested_bps",
        "last_allocated_bps",
        "last_requested_bps",
        "avg_f_weight",
        "avg_g_weight",
        "avg_h_weight",
        "weight_samples",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(args: argparse.Namespace, out: Path, rows: list[dict[str, float | int | str]]) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "output_csv": str(out),
        "model": "paper_rebuild",
        "constellation": args.constellation,
        "users": args.users,
        "methods": args.methods,
        "slots": args.slots,
        "seed": args.seed,
        "jobs": args.jobs,
        "traffic_mode": args.traffic_mode,
        "gateway_count": args.gateway_count,
        "paper_parameters": {
            "C_BAND": C_BAND,
            "PT_GT_GS_DB": PT_GT_GS_DB,
            "NOISE_DBM_PER_HZ": NOISE_DBM_PER_HZ,
            "ATMOSPHERIC_LOSS_DB": ATMOSPHERIC_LOSS_DB,
            "AGC_DELTA": AGC_DELTA,
            "UTILITY_H": UTILITY_H,
            "UTILITY_H_POINTS": UTILITY_H_POINTS,
            "INCLUDE_INTERFERENCE": INCLUDE_INTERFERENCE,
            "INTERFERENCE_SCALE": INTERFERENCE_SCALE,
            "CHANNEL_QUALITY_DECISION_NOISE": os.environ.get("LEO_CHANNEL_QUALITY_DECISION_NOISE", "actual"),
            "CHANNEL_QUALITY_AVG_WINDOW": os.environ.get("LEO_CHANNEL_QUALITY_AVG_WINDOW", "30"),
            "CHANNEL_QUALITY_AVG_SAMPLES": os.environ.get("LEO_CHANNEL_QUALITY_AVG_SAMPLES", "3"),
            "CHANNEL_QUALITY_MIN_HOLD_SLOTS": os.environ.get("LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS", "0"),
            "CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS": os.environ.get("LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS", ""),
            "MGCS_DELAY_WEIGHT": os.environ.get("LEO_MGCS_DELAY_WEIGHT", "0.0"),
            "MGCS_CAPACITY_TOLERANCE": os.environ.get("LEO_MGCS_CAPACITY_TOLERANCE", "0.0"),
            "USER_ELEVATION_DEG": os.environ.get("LEO_USER_ELEVATION_DEG", "5"),
            "RESET_HANDOVER_AFTER_INITIAL": os.environ.get("LEO_RESET_HANDOVER_AFTER_INITIAL", "0"),
            "RATE_UPPER": RATE_UPPER,
            "GSL_RATE_CAP": GSL_RATE_CAP,
            "GSL_RATE_CAP_HIGH_LOAD": GSL_RATE_CAP_HIGH_LOAD,
            "GSL_RATE_CAP_LOAD_THRESHOLD": GSL_RATE_CAP_LOAD_THRESHOLD,
            "GSL_RATE_CAP_POINTS": GSL_RATE_CAP_POINTS,
            "ISL_BANDWIDTH_POINTS": os.environ.get("LEO_ISL_BANDWIDTH_POINTS", ""),
            "ISL_BANDWIDTH_HIGH_LOAD": os.environ.get("LEO_ISL_BANDWIDTH_HIGH_LOAD", ""),
            "ISL_BANDWIDTH_LOAD_THRESHOLD": os.environ.get("LEO_ISL_BANDWIDTH_LOAD_THRESHOLD", ""),
            "SQ_FREE_ALPHA": SQ_FREE_ALPHA,
            "HANDOVER_CONTROL_MODE": HANDOVER_CONTROL_MODE,
            "UTILITY_HYSTERESIS": UTILITY_HYSTERESIS,
            "UTILITY_HYSTERESIS_POINTS": UTILITY_HYSTERESIS_POINTS,
            "CAHS_DELAY_WEIGHT": os.environ.get("LEO_PAPER_CAHS_DELAY_WEIGHT", "0.0"),
        },
        "unpublished_assumptions": {
            "traffic_mode": args.traffic_mode,
            "gateway_count": args.gateway_count,
            "DIRECT_PATH_QUALITY": DIRECT_PATH_QUALITY,
            "ISL_BANDWIDTH": BANDWIDTH,
            "source_AGC_PREDICT_WINDOW": AGC_PREDICT_WINDOW,
            "source_HANDOVER_COST": HANDOVER_COST,
        },
        "notes": [
            "Independent paper-rebuild path; run_experiments.py and final source-adjusted outputs are not modified.",
            "Gateway layout is an explicit assumption because the paper does not publish exact gateway positions.",
            "Baseline simulations remain CPU-bound; CUDA does not materially accelerate these numerical methods.",
        ],
        "row_count": len(rows),
    }
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def progress_bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else width
    return "#" * filled + "." * (width - filled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constellation", choices=["A", "B"], default="B")
    parser.add_argument("--users", nargs="+", type=int, default=[100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["RS", "MSTS", "MGCS", "CAHS"])
    parser.add_argument("--slots", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results") / "paper_rebuild_summary.csv")
    parser.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--traffic-mode", choices=["ue_pair", "ground_backbone"], default="ground_backbone")
    parser.add_argument("--gateway-count", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    ensure_dirs(BASE_DIR)
    if args.progress_every > 0:
        os.environ["LEO_PROGRESS_EVERY"] = str(args.progress_every)

    jobs = [
        (users, args.constellation, method, args.slots, args.seed, args.traffic_mode, args.gateway_count)
        for users in args.users
        for method in args.methods
    ]
    rows = []
    total_jobs = len(jobs)
    out_path = BASE_DIR / args.out
    if args.jobs <= 1:
        for job_index, job in enumerate(jobs, start=1):
            users, constellation, method, *_ = job
            print(f"running paper_rebuild constellation={constellation} users={users} method={method}", flush=True)
            rows.append(run_one(job))
            rows.sort(key=lambda r: (int(r["users"]), str(r["method"])))
            write_csv(rows, out_path)
            print(f"[{progress_bar(job_index, total_jobs)}] {job_index}/{total_jobs} jobs complete", flush=True)
    else:
        with get_context("spawn").Pool(processes=args.jobs) as pool:
            for job_index, row in enumerate(pool.imap_unordered(run_one, jobs), start=1):
                print(
                    f"[{progress_bar(job_index, total_jobs)}] {job_index}/{total_jobs} done "
                    f"constellation={args.constellation} users={row['users']} method={row['method']}",
                    flush=True,
                )
                rows.append(row)
                rows.sort(key=lambda r: (int(r["users"]), str(r["method"])))
                write_csv(rows, out_path)
    write_metadata(args, out_path, rows)
    print(out_path)
    if args.notify:
        subprocess.run(
            [
                sys.executable,
                str(BASE_DIR / "notify_qq.py"),
                f"paper-rebuild LEO reproduction complete: constellation={args.constellation}, output={out_path}",
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
