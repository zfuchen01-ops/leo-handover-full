import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from multiprocessing import get_context

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
early_parser = argparse.ArgumentParser(add_help=False)
early_parser.add_argument("--variant", choices=["paper", "calibrated"], default=os.environ.get("LEO_CAHS_VARIANT", "paper"))
early_args, _ = early_parser.parse_known_args()
os.environ["LEO_CAHS_VARIANT"] = early_args.variant
os.environ.setdefault("LEO_QUIET_LOGS", "1")
PAUSE_FILE = BASE_DIR / "EXPERIMENTS_PAUSED"
if PAUSE_FILE.exists() or os.environ.get("LEO_EXPERIMENTS_PAUSED", "0") == "1":
    print(f"experiments paused: remove {PAUSE_FILE} to run", file=sys.stderr)
    sys.exit(2)
for log_dir in [
    "log/handover",
    "log/network",
    "log/topology",
    "log/RL",
    "log/model",
    "log/hyper",
]:
    (BASE_DIR / log_dir).mkdir(parents=True, exist_ok=True)

from Defination import (
    SLOT_SECONDS,
    AGC_PREDICT_WINDOW,
    BANDWIDTH,
    CAHS_VARIANT,
    DIRECT_PATH_QUALITY,
    HANDOVER_COST,
    LIGHT_LOAD_USER_THRESHOLD,
    SAT_HEIGHT,
    TYPE_2PI,
)
from Handover import Handover
from Handover import C_BAND
from Network import Network
from Position import Convert_Angle_to_PI
from Topology import Topology
from User import User
from math import acos, cos, pi, sin


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


def ensure_log_dirs(base: Path) -> None:
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
    loc = []
    for j in range(rows):
        for i in range(cols):
            loc.append(
                [
                    Convert_Angle_to_PI(lon_start + i * lon_step),
                    Convert_Angle_to_PI(lat_start + j * lat_step),
                ]
            )
    return loc


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
        gateway_lons = [
            -120.0 + i * (240.0 / (gateway_count - 1))
            for i in range(gateway_count)
        ]
        gateway_lats = None
        gateway_weights = None
    else:
        gateway_lons = [
            -120.0 + i * (240.0 / gateway_count)
            for i in range(gateway_count)
        ]
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


def build_env(user_count: int, constellation: str, traffic_mode: str, gateway_count: int) -> Handover:
    reset_ids()
    topo = Topology()
    make_constellation(topo, constellation)
    topo.Add_User_From_Input(make_user_locations(user_count, traffic_mode, gateway_count))
    half = len(topo.user) // 2
    for i in range(half):
        topo.user[i].User_Connect_User(topo.user[i + half], "UPLOAD")
    net = Network(topo)
    return Handover(net=net)


def summarize(env: Handover, user_count: int, method: str) -> dict[str, float | int | str]:
    statics = env.statics
    handovers = np.asarray(statics[0], dtype=float)
    delay = np.asarray(statics[1], dtype=float)
    allocated = np.asarray(statics[2], dtype=float)
    requested = np.asarray(statics[3], dtype=float)
    flow_count = float(sum(len(user.user_to_connect_to) for user in env.topo.user))
    weights = np.asarray(env.weight_samples, dtype=float) if getattr(env, "weight_samples", None) else np.empty((0, 3))
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
        "avg_allocated_mhz": float(np.mean(allocated) * flow_count) if len(allocated) else 0.0,
        "avg_requested_mhz": float(np.mean(requested) * flow_count) if len(requested) else 0.0,
        "last_allocated_mhz": float(allocated[-1] * flow_count) if len(allocated) else 0.0,
        "last_requested_mhz": float(requested[-1] * flow_count) if len(requested) else 0.0,
        "avg_f_weight": float(np.mean(weights[:, 0])) if len(weights) else 0.0,
        "avg_g_weight": float(np.mean(weights[:, 1])) if len(weights) else 0.0,
        "avg_h_weight": float(np.mean(weights[:, 2])) if len(weights) else 0.0,
        "weight_samples": int(len(weights)),
    }


def run_one(user_count: int, constellation: str, method: str, slots: int, seed: int, traffic_mode: str, gateway_count: int) -> dict[str, float | int | str]:
    random.seed(seed)
    np.random.seed(seed)
    env = build_env(user_count, constellation, traffic_mode, gateway_count)
    env.Run_Network_Handover(0, slots * SLOT_SECONDS, SLOT_SECONDS, METHODS[method])
    return summarize(env, user_count, method)


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
        "avg_allocated_mhz",
        "avg_requested_mhz",
        "last_allocated_mhz",
        "last_requested_mhz",
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
        "constellation": args.constellation,
        "users": args.users,
        "methods": args.methods,
        "slots": args.slots,
        "seed": args.seed,
        "jobs": args.jobs,
        "traffic_mode": args.traffic_mode,
        "gateway_count": args.gateway_count,
        "variant": CAHS_VARIANT,
        "calibrated_parameters": {
            "CAHS_VARIANT": CAHS_VARIANT,
            "HANDOVER_COST": HANDOVER_COST,
            "DIRECT_PATH_QUALITY": DIRECT_PATH_QUALITY,
            "AGC_PREDICT_WINDOW": AGC_PREDICT_WINDOW,
            "LIGHT_LOAD_USER_THRESHOLD": LIGHT_LOAD_USER_THRESHOLD,
            "C_BAND": C_BAND,
            "BANDWIDTH": BANDWIDTH,
        },
        "notes": [
            "Baseline simulations are CPU-bound; CUDA is prepared for DQN/DRQN only.",
            "CAHS rows include average f/g/h utility weights for traceability.",
        ],
        "row_count": len(rows),
    }
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def progress_bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else width
    return "#" * filled + "." * (width - filled)


def run_job(job: tuple[int, str, str, int, int, str, int]) -> dict[str, float | int | str]:
    users, constellation, method, slots, seed, traffic_mode, gateway_count = job
    return run_one(users, constellation, method, slots, seed, traffic_mode, gateway_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constellation", choices=["A", "B"], default="B")
    parser.add_argument("--users", nargs="+", type=int, default=[100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["RS", "MSTS", "MGCS", "CAHS"])
    parser.add_argument("--slots", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results") / "summary.csv")
    parser.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--variant", choices=["paper", "calibrated"], default=early_args.variant)
    parser.add_argument("--traffic-mode", choices=["ue_pair", "ground_backbone"], default="ue_pair")
    parser.add_argument("--gateway-count", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=0, help="Print per-slot progress every N slots for single-process runs.")
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    base = BASE_DIR
    os.chdir(base)
    ensure_log_dirs(base)
    if args.progress_every > 0:
        os.environ["LEO_PROGRESS_EVERY"] = str(args.progress_every)

    jobs = [
        (users, args.constellation, method, args.slots, args.seed, args.traffic_mode, args.gateway_count)
        for users in args.users
        for method in args.methods
    ]
    rows = []
    total_jobs = len(jobs)
    if args.jobs <= 1:
        for job_index, (users, constellation, method, slots, seed, traffic_mode, gateway_count) in enumerate(jobs, start=1):
            print(f"running constellation={constellation} users={users} method={method}", flush=True)
            rows.append(run_one(users, constellation, method, slots, seed, traffic_mode, gateway_count))
            rows.sort(key=lambda r: (int(r["users"]), str(r["method"])))
            write_csv(rows, base / args.out)
            print(f"[{progress_bar(job_index, total_jobs)}] {job_index}/{total_jobs} jobs complete", flush=True)
    else:
        with get_context("spawn").Pool(processes=args.jobs) as pool:
            for job_index, row in enumerate(pool.imap_unordered(run_job, jobs), start=1):
                print(
                    f"[{progress_bar(job_index, total_jobs)}] {job_index}/{total_jobs} done "
                    f"constellation={args.constellation} users={row['users']} method={row['method']}",
                    flush=True,
                )
                rows.append(row)
                rows.sort(key=lambda r: (int(r["users"]), str(r["method"])))
                write_csv(rows, base / args.out)
    out_path = base / args.out
    write_metadata(args, out_path, rows)
    print(out_path)
    if args.notify:
        subprocess.run(
            [
                sys.executable,
                str(base / "notify_qq.py"),
                f"LEO复现实验完成: constellation={args.constellation}, output={out_path}",
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
