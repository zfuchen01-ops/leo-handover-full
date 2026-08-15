import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("LEO_QUIET_LOGS", "1")

from Handover import Handover
from Defination import SLOT_SECONDS
from Position import Calc_Sphere_Distance, Calc_Sphere_Elevation
from run_experiments import METHODS, build_env, ensure_log_dirs


def read_targets(path: Path, figure: str, method: str) -> dict[int, float]:
    if not path.exists():
        return {}
    targets = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("figure") != figure:
                continue
            if row.get("method") != method:
                continue
            if row.get("metric") != "throughput":
                continue
            targets[int(row["users"])] = float(row["value"])
    return targets


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def path_stats(env: Handover, source_sat, dest_sat) -> tuple[int, float, float]:
    if source_sat is None or dest_sat is None or source_sat == dest_sat:
        return 0, 1.0, 0.0
    con = source_sat.con_id
    spt = env.net.SPT[con - 1][source_sat.ID - 1][dest_sat.ID - 1]
    if not spt.isReached:
        return 0, 0.0, 0.0
    current = dest_sat.ID
    sat_count = env.topo.constellation[con - 1].orbit_num * env.topo.constellation[con - 1].sat_per_orbit
    hops = 0
    min_free_ratio = 1.0
    used_ratios = []
    while current != source_sat.ID and hops < sat_count:
        pre = env.net.SPT[con - 1][source_sat.ID - 1][current - 1].pre
        if pre <= 0:
            return hops, 0.0, percentile(used_ratios, 50)
        link = env.net.Lookup_LSA(con, pre, current)
        if link is None or link.total_band <= 0:
            return hops, 0.0, percentile(used_ratios, 50)
        free_ratio = max(0.0, min(1.0, (link.total_band - link.used_band) / link.total_band))
        min_free_ratio = min(min_free_ratio, free_ratio)
        used_ratios.append(1.0 - free_ratio)
        current = pre
        hops += 1
    return hops, min_free_ratio, percentile(used_ratios, 50)


def collect_slot(
    env: Handover,
    user_count: int,
    slot_index: int,
    flow_writer: csv.DictWriter | None,
    include_path_stats: bool = False,
) -> dict[str, float]:
    half = len(env.topo.user) // 2
    requested = []
    allocated = []
    source_rates = []
    source_distances = []
    source_elevations = []
    dest_distances = []
    dest_elevations = []
    path_hops = []
    path_min_free = []
    path_median_used = []
    zero_alloc = 0
    congested = 0
    same_sat = 0
    connected = 0
    beam_free = []

    for source_user in env.topo.user[:half]:
        source_sat = source_user.sat_connected
        if source_sat is not None:
            connected += 1
            beam_free.append(float(source_sat.beam - len(source_sat.user_connected)))
            source_distances.append(Calc_Sphere_Distance(source_user.we_pos, source_sat.we_pos))
            source_elevations.append(Calc_Sphere_Elevation(source_sat.we_pos, source_user.we_pos))
            source_rates.append(float(env.ho[source_user][source_sat].c_quality))
        for dest_user in source_user.user_to_connect_to:
            dest_sat = dest_user.sat_connected
            req = float(source_user.user_to_connect_to.get(dest_user, 0.0))
            alloc = float(source_user.allocate_band.get(dest_user, 0.0))
            requested.append(req)
            allocated.append(alloc)
            if alloc <= 0:
                zero_alloc += 1
            if req > 0 and alloc < req * 0.999:
                congested += 1
            if source_sat is not None and dest_sat is not None and source_sat == dest_sat:
                same_sat += 1
            if dest_sat is not None:
                dest_distances.append(Calc_Sphere_Distance(dest_user.we_pos, dest_sat.we_pos))
                dest_elevations.append(Calc_Sphere_Elevation(dest_sat.we_pos, dest_user.we_pos))
            if include_path_stats:
                hops, min_free, median_used = path_stats(env, source_sat, dest_sat)
            else:
                hops, min_free, median_used = 0, 0.0, 0.0
            path_hops.append(float(hops))
            path_min_free.append(float(min_free))
            path_median_used.append(float(median_used))
            if flow_writer is not None:
                flow_writer.writerow(
                    {
                        "slot": slot_index,
                        "time": env.topo.current_time,
                        "users": user_count,
                        "source_user": source_user.user_ID,
                        "dest_user": dest_user.user_ID,
                        "source_sat": source_sat.ID if source_sat is not None else "",
                        "dest_sat": dest_sat.ID if dest_sat is not None else "",
                        "requested_mhz": req,
                        "allocated_mhz": alloc,
                        "source_gsl_rate_mhz": source_rates[-1] if source_sat is not None else 0.0,
                        "source_gsl_distance_km": source_distances[-1] if source_sat is not None else 0.0,
                        "dest_gsl_distance_km": dest_distances[-1] if dest_sat is not None else 0.0,
                        "path_hops": hops,
                        "path_min_free_ratio": min_free,
                        "path_median_used_ratio": median_used,
                    }
                )

    flow_count = len(requested)
    req_sum = float(sum(requested))
    alloc_sum = float(sum(allocated))
    return {
        "users": user_count,
        "slot": slot_index,
        "time": env.topo.current_time,
        "flow_count": flow_count,
        "connected_sources": connected,
        "requested_total_mhz": req_sum,
        "allocated_total_mhz": alloc_sum,
        "request_mean_mhz": req_sum / flow_count if flow_count else 0.0,
        "allocated_mean_mhz": alloc_sum / flow_count if flow_count else 0.0,
        "allocation_fill_ratio": alloc_sum / req_sum if req_sum else 0.0,
        "zero_alloc_share": zero_alloc / flow_count if flow_count else 0.0,
        "congested_share": congested / flow_count if flow_count else 0.0,
        "same_sat_share": same_sat / flow_count if flow_count else 0.0,
        "source_rate_p10": percentile(source_rates, 10),
        "source_rate_p50": percentile(source_rates, 50),
        "source_rate_p90": percentile(source_rates, 90),
        "source_distance_p50_km": percentile(source_distances, 50),
        "source_elevation_p50_rad": percentile(source_elevations, 50),
        "dest_distance_p50_km": percentile(dest_distances, 50),
        "dest_elevation_p50_rad": percentile(dest_elevations, 50),
        "path_hops_mean": float(np.mean(path_hops)) if path_hops else 0.0,
        "path_min_free_p10": percentile(path_min_free, 10),
        "path_min_free_p50": percentile(path_min_free, 50),
        "path_median_used_p50": percentile(path_median_used, 50),
        "beam_free_p10": percentile(beam_free, 10),
        "beam_free_p50": percentile(beam_free, 50),
        "handover_total": env.ho_count,
        "source_handover_total": env.source_ho_count,
        "destination_handover_total": env.destination_ho_count,
        "block_count": env.block_count,
    }


def write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, float | int]], target: float | None) -> dict[str, float | int | str]:
    users = int(rows[0]["users"])
    flow_count = int(rows[0]["flow_count"])
    slots = len(rows)
    req = [float(r["requested_total_mhz"]) for r in rows]
    alloc = [float(r["allocated_total_mhz"]) for r in rows]
    result = {
        "users": users,
        "slots": slots,
        "flow_count": flow_count,
        "avg_requested_total_mhz": float(np.mean(req)),
        "avg_allocated_total_mhz": float(np.mean(alloc)),
        "last_requested_total_mhz": req[-1],
        "last_allocated_total_mhz": alloc[-1],
        "avg_allocation_fill_ratio": float(np.mean([float(r["allocation_fill_ratio"]) for r in rows])),
        "avg_congested_share": float(np.mean([float(r["congested_share"]) for r in rows])),
        "avg_zero_alloc_share": float(np.mean([float(r["zero_alloc_share"]) for r in rows])),
        "avg_same_sat_share": float(np.mean([float(r["same_sat_share"]) for r in rows])),
        "source_rate_p50_mean": float(np.mean([float(r["source_rate_p50"]) for r in rows])),
        "source_rate_p10_mean": float(np.mean([float(r["source_rate_p10"]) for r in rows])),
        "source_distance_p50_km_mean": float(np.mean([float(r["source_distance_p50_km"]) for r in rows])),
        "source_elevation_p50_rad_mean": float(np.mean([float(r["source_elevation_p50_rad"]) for r in rows])),
        "path_hops_mean": float(np.mean([float(r["path_hops_mean"]) for r in rows])),
        "path_min_free_p10_mean": float(np.mean([float(r["path_min_free_p10"]) for r in rows])),
        "beam_free_p10_mean": float(np.mean([float(r["beam_free_p10"]) for r in rows])),
        "handover_frequency": float(rows[-1]["handover_total"]) / (users * slots) if users and slots else 0.0,
        "source_handover_frequency": float(rows[-1]["source_handover_total"]) / ((users / 2) * slots) if users and slots else 0.0,
        "destination_handover_frequency": float(rows[-1]["destination_handover_total"]) / ((users / 2) * slots) if users and slots else 0.0,
        "block_count": int(rows[-1]["block_count"]),
    }
    if target is not None:
        result["paper_throughput_target_mhz"] = target
        result["requested_vs_target_ratio"] = result["avg_requested_total_mhz"] / target if target else 0.0
        result["allocated_vs_target_ratio"] = result["avg_allocated_total_mhz"] / target if target else 0.0
        result["requested_below_target"] = "yes" if result["avg_requested_total_mhz"] < target else "no"
    return result


def run_user_trace(job: tuple[argparse.Namespace, int]) -> tuple[list[dict[str, float | int]], dict[str, float | int | str]]:
    args, users = job
    random.seed(args.seed)
    np.random.seed(args.seed)
    ensure_log_dirs(BASE_DIR)
    targets = read_targets(args.targets, f"fig4_{args.constellation}", args.method)
    env = build_env(users, args.constellation, args.traffic_mode, args.gateway_count)
    env.net.Initial_Network()
    env.Initial_GSlink()
    rows_for_user = []
    env.Network_Handover(0, METHODS[args.method])
    rows_for_user.append(collect_slot(env, users, 1, None, args.path_stats))
    for slot_index, t in enumerate(range(SLOT_SECONDS, args.slots * SLOT_SECONDS, SLOT_SECONDS), start=2):
        env.Network_Handover(t, METHODS[args.method])
        row = collect_slot(env, users, slot_index, None, args.path_stats)
        rows_for_user.append(row)
        if args.progress_every and slot_index % args.progress_every == 0:
            print(
                f"trace users={users} slot={slot_index}/{args.slots} "
                f"req={row['requested_total_mhz']:.1f} alloc={row['allocated_total_mhz']:.1f}",
                flush=True,
            )
    return rows_for_user, aggregate(rows_for_user, targets.get(users))


def run_trace(args: argparse.Namespace) -> tuple[list[dict[str, float | int]], list[dict[str, float | int | str]]]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    ensure_log_dirs(BASE_DIR)
    slot_rows = []
    summary_rows = []
    flow_writer = None
    flow_file = None
    if args.flow_out:
        flow_path = args.flow_out if args.flow_out.is_absolute() else BASE_DIR / args.flow_out
        flow_path.parent.mkdir(parents=True, exist_ok=True)
        flow_file = flow_path.open("w", newline="", encoding="utf-8")
        flow_writer = csv.DictWriter(
            flow_file,
            fieldnames=[
                "slot",
                "time",
                "users",
                "source_user",
                "dest_user",
                "source_sat",
                "dest_sat",
                "requested_mhz",
                "allocated_mhz",
                "source_gsl_rate_mhz",
                "source_gsl_distance_km",
                "dest_gsl_distance_km",
                "path_hops",
                "path_min_free_ratio",
                "path_median_used_ratio",
            ],
        )
        flow_writer.writeheader()

    try:
        if args.jobs > 1 and flow_writer is None:
            jobs = [(args, users) for users in args.users]
            with get_context("spawn").Pool(processes=min(args.jobs, len(jobs))) as pool:
                for job_index, (rows_for_user, summary) in enumerate(pool.imap_unordered(run_user_trace, jobs), start=1):
                    slot_rows.extend(rows_for_user)
                    summary_rows.append(summary)
                    print(
                        f"[{job_index}/{len(args.users)}] users={summary['users']} "
                        f"avg_req={summary['avg_requested_total_mhz']:.1f} "
                        f"avg_alloc={summary['avg_allocated_total_mhz']:.1f} "
                        f"req/target={summary.get('requested_vs_target_ratio', 0):.3f}",
                        flush=True,
                    )
        else:
            targets = read_targets(args.targets, f"fig4_{args.constellation}", args.method)
            for job_index, users in enumerate(args.users, start=1):
                random.seed(args.seed)
                np.random.seed(args.seed)
                env = build_env(users, args.constellation, args.traffic_mode, args.gateway_count)
                env.net.Initial_Network()
                env.Initial_GSlink()
                rows_for_user = []
                env.Network_Handover(0, METHODS[args.method])
                row = collect_slot(
                    env,
                    users,
                    1,
                    flow_writer if args.flow_every and 1 % args.flow_every == 0 else None,
                    args.path_stats,
                )
                rows_for_user.append(row)
                slot_rows.append(row)
                for slot_index, t in enumerate(range(SLOT_SECONDS, args.slots * SLOT_SECONDS, SLOT_SECONDS), start=2):
                    env.Network_Handover(t, METHODS[args.method])
                    row = collect_slot(
                        env,
                        users,
                        slot_index,
                        flow_writer if args.flow_every and slot_index % args.flow_every == 0 else None,
                        args.path_stats,
                    )
                    rows_for_user.append(row)
                    slot_rows.append(row)
                    if args.progress_every and slot_index % args.progress_every == 0:
                        print(
                            f"trace users={users} slot={slot_index}/{args.slots} "
                            f"req={row['requested_total_mhz']:.1f} alloc={row['allocated_total_mhz']:.1f}",
                            flush=True,
                        )
                summary = aggregate(rows_for_user, targets.get(users))
                summary_rows.append(summary)
                print(
                    f"[{job_index}/{len(args.users)}] users={users} "
                    f"avg_req={summary['avg_requested_total_mhz']:.1f} "
                    f"avg_alloc={summary['avg_allocated_total_mhz']:.1f} "
                    f"req/target={summary.get('requested_vs_target_ratio', 0):.3f}",
                    flush=True,
                )
    finally:
        if flow_file is not None:
            flow_file.close()
    slot_rows.sort(key=lambda row: (int(row["users"]), int(row["slot"])))
    summary_rows.sort(key=lambda row: int(row["users"]))
    return slot_rows, summary_rows


def write_meta(path: Path, args: argparse.Namespace, summary_rows: list[dict[str, float | int | str]]) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "env": {
            key: os.environ.get(key)
            for key in sorted(os.environ)
            if key.startswith("LEO_")
        },
        "users": args.users,
        "slots": args.slots,
        "method": args.method,
        "traffic_mode": args.traffic_mode,
        "gateway_count": args.gateway_count,
        "row_count": len(summary_rows),
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constellation", choices=["A", "B"], default="B")
    parser.add_argument("--users", nargs="+", type=int, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), default="CAHS")
    parser.add_argument("--slots", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--traffic-mode", choices=["ue_pair", "ground_backbone"], default="ground_backbone")
    parser.add_argument("--gateway-count", type=int, default=2)
    parser.add_argument("--targets", type=Path, default=BASE_DIR / "results" / "paper_targets_fig4_B.csv")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "results" / "demand_trace_summary.csv")
    parser.add_argument("--slot-out", type=Path, default=BASE_DIR / "results" / "demand_trace_slots.csv")
    parser.add_argument("--flow-out", type=Path)
    parser.add_argument("--flow-every", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--path-stats", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    out = args.out if args.out.is_absolute() else BASE_DIR / args.out
    slot_out = args.slot_out if args.slot_out.is_absolute() else BASE_DIR / args.slot_out
    slot_rows, summary_rows = run_trace(args)
    write_rows(slot_out, slot_rows)
    write_rows(out, summary_rows)
    write_meta(out, args, summary_rows)
    print(out)
    print(slot_out)


if __name__ == "__main__":
    main()
