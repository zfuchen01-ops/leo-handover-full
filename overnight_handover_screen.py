import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results" / "overnight_handover_screen"
KEY_USERS = [100, 350, 600]
FULL_USERS = list(range(100, 601, 50))
WEIGHTS = {"throughput_mean": 0.30, "delay_mean": 0.30, "handover_mean": 0.40}
RESULT_FIELDS = {
    "throughput": "avg_allocated_mhz",
    "delay": "avg_delay_ms",
    "handover": "handover_frequency",
}

# Exact experiments already rejected in the previous diagnostics are not
# generated again. They remain listed in the final report as negative evidence.
KNOWN_REJECTED = [
    "MGCS channel-quality hysteresis 0.05/0.10",
    "MGCS fixed service time 30/60/90 seconds",
    "MGCS thermal-only decision noise",
    "MGCS delay weights -1/-2/-4",
    "MGCS capacity tolerances 0.01/0.03/0.05",
    "MGCS average window 30 seconds with the previous baseline",
    "B gateway left_open + nearest",
]

COMMON_ENV = {
    "LEO_HANDOVER_SCOPE": "all",
    "LEO_QUIET_LOGS": "1",
    "LEO_SOURCE_LAYOUT": "random",
    "LEO_GATEWAY_LAYOUT": "left_open",
    "LEO_GATEWAY_ASSIGN": "cycle",
    "LEO_PAPER_INCLUDE_INTERFERENCE": "1",
    "LEO_PAPER_SQ_FREE_ALPHA": "0.5",
}

BASE_ENV = {
    "A": {
        **COMMON_ENV,
        "LEO_RESET_HANDOVER_AFTER_INITIAL": "1",
        "LEO_GSL_RATE_CAP_POINTS": "100:500,350:1000,600:2600",
        "LEO_ISL_BANDWIDTH": "5000",
        "LEO_ISL_BANDWIDTH_POINTS": "100:5000,500:5000,550:7000,600:9000",
        "LEO_PAPER_HANDOVER_CONTROL_MODE": "none",
        "LEO_PAPER_UTILITY_HYSTERESIS_POINTS": "100:0.04,150:0.065,200:0.075,250:0.105,300:0.13,350:0.15,600:0.20",
    },
    "B": {
        **COMMON_ENV,
        "LEO_RESET_HANDOVER_AFTER_INITIAL": "1",
        "LEO_GSL_RATE_CAP_POINTS": "100:580,350:1050,600:1000",
        "LEO_ISL_BANDWIDTH": "3500",
        "LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS": "100:2,150:2,200:3,600:4",
        "LEO_PAPER_HANDOVER_CONTROL_MODE": "constant",
        "LEO_PAPER_HANDOVER_COST_POINTS": "100:0.085,350:0.125,600:0.125",
        "LEO_PAPER_CAHS_DELAY_WEIGHT": "0.40",
    },
}

GATEWAY_COUNT = {"A": 16, "B": 12}


@dataclass(frozen=True)
class Candidate:
    method: str
    constellation: str
    name: str
    env: dict[str, str]


def candidate_score(errors: dict[str, float]) -> float:
    return sum(errors[key] * weight for key, weight in WEIGHTS.items())


def passes_guardrails(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    for metric in ("throughput_mean", "delay_mean"):
        if candidate[metric] > baseline[metric] + 0.03 + 1e-12:
            return False
    for metric in ("throughput_max", "delay_max", "handover_max"):
        if candidate[metric] > baseline[metric] * 2.0 + 1e-12:
            return False
    return True


def stage_timeout_seconds(stage: str) -> int:
    if "key400" in stage or "full400" in stage:
        return 90 * 60
    if "full80" in stage:
        return 30 * 60
    return 15 * 60


def merged_env(candidate: Candidate) -> dict[str, str]:
    return {**BASE_ENV[candidate.constellation], **candidate.env}


def config_key(candidate: Candidate, users: list[int], slots: int) -> str:
    payload = {
        "method": candidate.method,
        "constellation": candidate.constellation,
        "env": merged_env(candidate),
        "users": users,
        "slots": slots,
        "gateway_count": GATEWAY_COUNT[candidate.constellation],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def generate_rs_stage1(constellation: str) -> list[Candidate]:
    candidates = [Candidate("RS", constellation, "baseline", {})]
    candidates.extend(
        Candidate("RS", constellation, f"destination_mode_{mode.lower()}", {"LEO_DESTINATION_DECISION_MODE": mode})
        for mode in ("SERVICE_TIME", "CHANNEL_QUALITY", "DELAY", "DELAY_HYSTERESIS")
    )
    candidates.extend(
        Candidate("RS", constellation, f"destination_service_{value}", {"LEO_DESTINATION_MIN_SERVICE_TIME": value})
        for value in ("0", "30", "60", "90", "120")
    )
    candidates.extend(
        Candidate("RS", constellation, f"destination_delay_hysteresis_{value.replace('.', 'p')}", {"LEO_DELAY_HYSTERESIS": value})
        for value in ("0", "0.02", "0.05", "0.10")
    )
    candidates.extend(
        Candidate("RS", constellation, f"reset_initial_{value}", {"LEO_RESET_HANDOVER_AFTER_INITIAL": value})
        for value in ("0", "1")
    )
    return dedupe_candidates(candidates)


def generate_mgcs_stage1(constellation: str) -> list[Candidate]:
    candidates = [Candidate("MGCS", constellation, "baseline", {})]
    candidates.extend(
        Candidate(
            "MGCS",
            constellation,
            f"hold_{hold}",
            {
                "LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS": str(hold),
                "LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS": "",
            },
        )
        for hold in range(1, 9)
    )
    point_templates = {
        "hold_points_soft": "100:1,350:2,600:3",
        "hold_points_medium": "100:2,350:4,600:6",
        "hold_points_strong": "100:3,350:6,600:8",
        "hold_points_highload": "100:1,350:4,600:8",
    }
    candidates.extend(
        Candidate("MGCS", constellation, name, {"LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS": value})
        for name, value in point_templates.items()
    )
    candidates.extend(
        Candidate(
            "MGCS",
            constellation,
            f"average_w{window}_n{samples}",
            {
                "LEO_CHANNEL_QUALITY_DECISION_NOISE": "average",
                "LEO_CHANNEL_QUALITY_AVG_WINDOW": str(window),
                "LEO_CHANNEL_QUALITY_AVG_SAMPLES": str(samples),
            },
        )
        for window, samples in itertools.product((30, 60, 90), (3, 5))
    )
    candidates.extend(
        Candidate("MGCS", constellation, f"interference_{value.replace('.', 'p')}", {"LEO_PAPER_INTERFERENCE_SCALE": value})
        for value in ("0.10", "0.20", "0.30", "0.40", "0.60", "0.80")
    )
    return dedupe_candidates(candidates)


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen = set()
    unique = []
    for candidate in candidates:
        signature = json.dumps(merged_env(candidate), sort_keys=True)
        key = (candidate.method, candidate.constellation, signature)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def combine_top_candidates(candidates: list[Candidate], limit: int = 20) -> list[Candidate]:
    combined = []
    for left, right in itertools.combinations(candidates, 2):
        conflicts = {key for key in left.env.keys() & right.env.keys() if left.env[key] != right.env[key]}
        if conflicts:
            continue
        combined.append(
            Candidate(
                left.method,
                left.constellation,
                f"combo_{left.name}__{right.name}",
                {**left.env, **right.env},
            )
        )
    return dedupe_candidates(combined)[:limit]


def generate_mgcs_path_expansions(candidate: Candidate) -> list[Candidate]:
    constellation = candidate.constellation
    if constellation == "A":
        variants = [
            ("cap_2200", {"LEO_GSL_RATE_CAP_POINTS": "100:500,350:1000,600:2200"}),
            ("isl_flat_5000", {"LEO_ISL_BANDWIDTH_POINTS": ""}),
            ("gateway_endpoints", {"LEO_GATEWAY_LAYOUT": "endpoints"}),
        ]
    else:
        variants = [
            ("cap_500_1000", {"LEO_GSL_RATE_CAP_POINTS": "100:500,350:1000,600:1000"}),
            ("isl_3000", {"LEO_ISL_BANDWIDTH": "3000"}),
            ("gateway_endpoints", {"LEO_GATEWAY_LAYOUT": "endpoints"}),
        ]
    return [
        Candidate(candidate.method, constellation, f"{candidate.name}__{name}", {**candidate.env, **env})
        for name, env in variants
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def targets_for(constellation: str, method: str, users: list[int]) -> dict[tuple[str, int], float]:
    figure = "fig3_A" if constellation == "A" else "fig4_B"
    rows = read_csv(BASE / "results" / f"paper_targets_{figure}.csv")
    return {
        (row["metric"], int(row["users"])): float(row["value"])
        for row in rows
        if row["method"] == method and int(row["users"]) in users and not math.isnan(float(row["value"]))
    }


def summarize_result(result_csv: Path, candidate: Candidate, users: list[int]) -> dict[str, object]:
    rows = [row for row in read_csv(result_csv) if row["method"] == candidate.method and int(row["users"]) in users]
    targets = targets_for(candidate.constellation, candidate.method, users)
    summary: dict[str, object] = {
        "method": candidate.method,
        "constellation": candidate.constellation,
        "candidate": candidate.name,
        "result_csv": str(result_csv),
        "env_json": json.dumps(candidate.env, sort_keys=True),
    }
    for metric, field in RESULT_FIELDS.items():
        errors = []
        for row in rows:
            users_value = int(row["users"])
            target = targets.get((metric, users_value))
            if target is None:
                continue
            errors.append(abs(float(row[field]) - target) / abs(target) if target else 0.0)
        summary[f"{metric}_mean"] = sum(errors) / len(errors) if errors else float("inf")
        summary[f"{metric}_max"] = max(errors) if errors else float("inf")
    summary["score"] = candidate_score(summary)  # type: ignore[arg-type]
    return summary


class ScreenRunner:
    def __init__(self, deadline: float, jobs: int, max_candidates: int | None = None):
        self.deadline = deadline
        self.jobs = jobs
        self.max_candidates = max_candidates
        self.runs_started = 0
        RESULTS.mkdir(parents=True, exist_ok=True)

    def time_remaining(self) -> float:
        return self.deadline - time.time()

    def run_candidate(self, candidate: Candidate, users: list[int], slots: int, stage: str) -> dict[str, object] | None:
        key = config_key(candidate, users, slots)
        run_dir = RESULTS / "runs" / candidate.constellation / candidate.method / stage
        result_csv = run_dir / f"{candidate.name}_{key}.csv"
        summary_json = result_csv.with_suffix(".summary.json")
        log_path = result_csv.with_suffix(".log")
        if summary_json.exists() and result_csv.exists():
            return json.loads(summary_json.read_text(encoding="utf-8"))
        if self.time_remaining() <= 60:
            return None
        if self.max_candidates is not None and self.runs_started >= self.max_candidates:
            return None

        run_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(merged_env(candidate))
        cmd = [
            sys.executable,
            "run_paper_rebuild.py",
            "--constellation",
            candidate.constellation,
            "--users",
            *[str(value) for value in users],
            "--methods",
            candidate.method,
            "--slots",
            str(slots),
            "--jobs",
            str(min(self.jobs, len(users))),
            "--traffic-mode",
            "ground_backbone",
            "--gateway-count",
            str(GATEWAY_COUNT[candidate.constellation]),
            "--out",
            str(result_csv.relative_to(BASE)),
        ]
        self.runs_started += 1
        print(f"START {stage} {candidate.constellation}/{candidate.method} {candidate.name} remaining={self.time_remaining()/3600:.2f}h", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=BASE,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=max(60, min(stage_timeout_seconds(stage), int(self.time_remaining()))),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return None
        if completed.returncode != 0 or not result_csv.exists():
            print(f"FAILED {candidate.name}; see {log_path}", flush=True)
            return None
        summary = summarize_result(result_csv, candidate, users)
        summary.update({"stage": stage, "slots": slots, "users": " ".join(map(str, users)), "config_key": key})
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"DONE {candidate.name} score={float(summary['score']):.4f}", flush=True)
        return summary

    def run_candidates(self, candidates: list[Candidate], users: list[int], slots: int, stage: str) -> list[dict[str, object]]:
        summaries = []
        for candidate in candidates:
            summary = self.run_candidate(candidate, users, slots, stage)
            if summary is None:
                break
            summaries.append(summary)
            write_rankings(stage, summaries)
        return summaries


def write_rankings(stage: str, summaries: list[dict[str, object]]) -> Path:
    ranked = sorted(summaries, key=lambda row: float(row["score"]))
    out = RESULTS / "rankings" / f"{stage}.csv"
    write_csv(out, ranked)
    return out


def merge_final_rows(existing: list[dict[str, object]], new: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = {(str(row["constellation"]), str(row["method"])): row for row in existing}
    grouped.update({(str(row["constellation"]), str(row["method"])): row for row in new})
    return [grouped[key] for key in sorted(grouped)]


def candidate_from_summary(summary: dict[str, object]) -> Candidate:
    return Candidate(
        str(summary["method"]),
        str(summary["constellation"]),
        str(summary["candidate"]),
        json.loads(str(summary["env_json"])),
    )


def select_passing(summaries: list[dict[str, object]], baseline: dict[str, object], count: int) -> list[Candidate]:
    passing = [row for row in summaries if passes_guardrails(row, baseline)]  # type: ignore[arg-type]
    passing.sort(key=lambda row: float(row["score"]))
    return [candidate_from_summary(row) for row in passing[:count]]


def plot_diagnostics(result_csv: Path, candidate: Candidate, out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(
        [row for row in read_csv(result_csv) if row["method"] == candidate.method],
        key=lambda row: int(row["users"]),
    )
    targets = targets_for(candidate.constellation, candidate.method, [int(row["users"]) for row in rows])
    users = [int(row["users"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    specs = [
        ("avg_allocated_mhz", "throughput", "Throughput (MHz)"),
        ("avg_delay_ms", "delay", "Delay (ms)"),
        ("handover_frequency", "handover", "Handover frequency / slot"),
    ]
    for axis, (field, metric, label) in zip(axes, specs):
        axis.plot(users, [float(row[field]) for row in rows], marker="o", label="rebuild all-user")
        axis.plot(users, [targets[(metric, value)] for value in users], marker="x", label="paper")
        if metric == "handover":
            axis.plot(users, [float(row["source_handover_frequency"]) for row in rows], linestyle="--", label="source diagnostic")
            axis.plot(users, [float(row["destination_handover_frequency"]) for row in rows], linestyle=":", label="destination diagnostic")
        axis.set_xlabel("Number of UEs")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"{candidate.constellation}/{candidate.method}: {candidate.name}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def write_acceptance_report(final_rows: list[dict[str, object]]) -> Path:
    out = RESULTS / "acceptance_report_zh.md"
    lines = [
        "# A/MGCS 与 RS Handover 一夜筛选说明",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 结论口径",
        "",
        "论文对照仍使用全用户 handover frequency。source/destination 分解只用于诊断，不用于偷偷替换论文指标。",
        "",
        "## 最佳候选",
        "",
        "| 星座 | 方法 | 候选 | 综合分数 | throughput mean | delay mean | handover mean |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        numeric_row = {
            **row,
            "score": float(row["score"]),
            "throughput_mean": float(row["throughput_mean"]),
            "delay_mean": float(row["delay_mean"]),
            "handover_mean": float(row["handover_mean"]),
        }
        lines.append(
            "| {constellation} | {method} | {candidate} | {score:.3%} | {throughput_mean:.3%} | {delay_mean:.3%} | {handover_mean:.3%} |".format(
                **numeric_row
            )
        )
        if int(row["slots"]) < 400 or len(str(row["users"]).split()) < len(FULL_USERS):
            lines.append(
                f"\n> 注意：{row['constellation']}/{row['method']} 当前仅完成 {row['slots']}-slot、"
                f"{len(str(row['users']).split())} 个用户点的筛选，不能视为完整 400-slot 验证。\n"
            )
    lines.extend(
        [
            "",
            "## 结果解释",
            "",
            "- RS 的源侧切换频率本来就接近论文，但目的侧切换频率明显偏低；全用户统计因此约少一半。若 destination 决策搜索仍无法修复，最合理的结论是论文没有公开完整的目的侧切换或统计定义。",
            "- MGCS 对干扰参与方式、信道质量平滑和最小保持时隙极敏感。改善若依赖这些参数，应称为 calibrated reproduction，而不是声称恢复了论文原始实现。",
            "- 所有候选都同时受吞吐和时延护栏约束，避免通过压制切换事件制造一条看似漂亮但网络性能错误的曲线。",
            "",
            "## 已有负面证据（本轮不重复浪费计算）",
            "",
            *[f"- {item}" for item in KNOWN_REJECTED],
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_group(runner: ScreenRunner, constellation: str, method: str) -> dict[str, object] | None:
    stage_prefix = f"{constellation}_{method}"
    stage1_candidates = generate_rs_stage1(constellation) if method == "RS" else generate_mgcs_stage1(constellation)
    stage1 = runner.run_candidates(stage1_candidates, KEY_USERS, 80, f"{stage_prefix}_stage1_key80")
    if not stage1:
        return None
    baseline = next((row for row in stage1 if row["candidate"] == "baseline"), stage1[0])
    top = select_passing(stage1, baseline, 5)

    stage2_candidates = combine_top_candidates(top[:3])
    if method == "MGCS" and top:
        stage2_candidates.extend(generate_mgcs_path_expansions(top[0]))
    stage2 = runner.run_candidates(dedupe_candidates(stage2_candidates), KEY_USERS, 80, f"{stage_prefix}_stage2_key80")
    merged_summaries = stage1 + stage2
    top5 = select_passing(merged_summaries, baseline, 5)
    if not top5:
        top5 = [Candidate(method, constellation, "baseline", {})]

    full80 = runner.run_candidates(top5, FULL_USERS, 80, f"{stage_prefix}_top5_full80")
    full80_sorted = sorted(full80, key=lambda row: float(row["score"]))
    top2 = [candidate_from_summary(row) for row in full80_sorted[:2]]
    if not top2:
        return min(merged_summaries, key=lambda row: float(row["score"]))

    key400 = runner.run_candidates(top2, KEY_USERS, 400, f"{stage_prefix}_top2_key400")
    winner_source = min(key400 or full80_sorted, key=lambda row: float(row["score"]))
    winner = candidate_from_summary(winner_source)
    final = runner.run_candidate(winner, FULL_USERS, 400, f"{stage_prefix}_winner_full400")
    return final or winner_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable staged RS/MGCS handover screening.")
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--groups", nargs="+", default=["A/RS", "B/RS", "A/MGCS", "B/MGCS"])
    args = parser.parse_args()

    started = time.time()
    runner = ScreenRunner(started + args.budget_hours * 3600, args.jobs, args.max_candidates)
    final_rows = []
    for group in args.groups:
        constellation, method = group.split("/", 1)
        result = run_group(runner, constellation, method)
        if result:
            final_rows.append(result)
            candidate = candidate_from_summary(result)
            result_csv = Path(str(result["result_csv"]))
            plot_diagnostics(result_csv, candidate, RESULTS / "plots" / f"{constellation}_{method}_best.png")
        if runner.time_remaining() <= 60:
            break
    final_path = RESULTS / "rankings" / "final_best.csv"
    existing_final = read_csv(final_path) if final_path.exists() else []
    final_rows = merge_final_rows(existing_final, final_rows)
    write_rankings("final_best", final_rows)
    report = write_acceptance_report(final_rows)
    manifest = {
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "budget_hours": args.budget_hours,
        "groups": args.groups,
        "runs_started": runner.runs_started,
        "report": str(report),
        "known_rejected": KNOWN_REJECTED,
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
