import argparse
import csv
import json
from pathlib import Path


EXPECTED_USERS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
EXPECTED_METHODS = ["RS", "MSTS", "MGCS", "CAHS"]
METRICS = ["throughput", "delay", "handover"]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_meta(csv_path: Path) -> dict:
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def completeness(rows: list[dict[str, str]]) -> str:
    expected = len(EXPECTED_USERS) * len(EXPECTED_METHODS)
    actual = len(rows)
    return "complete" if actual == expected else f"incomplete ({actual}/{expected} rows)"


def metadata_block(csv_path: Path) -> str:
    meta = read_meta(csv_path)
    if not meta:
        return f"- `{csv_path.name}`: no metadata file found."
    params = meta.get("paper_parameters") or meta.get("calibrated_parameters", {})
    assumptions = meta.get("unpublished_assumptions", {})
    return (
        f"- `{csv_path.name}`: model={meta.get('model', meta.get('variant', 'source_adjusted'))}, "
        f"constellation={meta.get('constellation')}, slots={meta.get('slots')}, "
        f"traffic_mode={meta.get('traffic_mode')}, gateway_count={meta.get('gateway_count')}, "
        f"seed={meta.get('seed')}, jobs={meta.get('jobs')}, rows={meta.get('row_count')}, "
        f"params={params}, assumptions={assumptions}"
    )


def error_summary(compare_path: Path) -> str:
    rows = read_rows(compare_path)
    if not rows:
        return f"No comparison file found: `{compare_path.name}`.\n"
    lines = [
        f"### {compare_path.stem}",
        "",
        "| Method | Metric | Mean rel. error | Max rel. error |",
        "|---|---|---:|---:|",
    ]
    for method in ["CAHS", "MGCS", "MSTS", "RS"]:
        for metric in METRICS:
            vals = [
                float(row["rel_error"])
                for row in rows
                if row["method"] == method and row["metric"] == metric
            ]
            if not vals:
                continue
            lines.append(f"| {method} | {metric} | {sum(vals) / len(vals):.3f} | {max(vals):.3f} |")
    return "\n".join(lines) + "\n"


def cahs_mgcs_table(title: str, csv_path: Path) -> str:
    rows = read_rows(csv_path)
    grouped: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["users"]), {})[row["method"]] = row
    lines = [
        f"### {title}",
        "",
        "| UEs | CAHS/MGCS throughput | Delay gain(ms) | CAHS handover | MGCS handover |",
        "|---:|---:|---:|---:|---:|",
    ]
    for users in sorted(grouped):
        methods = grouped[users]
        if "CAHS" not in methods or "MGCS" not in methods:
            continue
        cahs = methods["CAHS"]
        mgcs = methods["MGCS"]
        ratio = float(cahs["avg_allocated_mhz"]) / float(mgcs["avg_allocated_mhz"])
        delay_gain = float(mgcs["avg_delay_ms"]) - float(cahs["avg_delay_ms"])
        lines.append(
            f"| {users} | {ratio:.3f} | {delay_gain:.2f} | "
            f"{float(cahs['handover_frequency']):.4f} | {float(mgcs['handover_frequency']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def key_points_table(title: str, csv_path: Path) -> str:
    rows = read_rows(csv_path)
    grouped = {(row["method"], int(row["users"])): row for row in rows}
    lines = [
        f"### {title}",
        "",
        "| UEs | Method | Throughput (bps) | Delay(ms) | Handover/slot | f | g | h |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for users in [100, 350, 600]:
        for method in ["CAHS", "MGCS", "MSTS", "RS"]:
            row = grouped.get((method, users))
            if not row:
                continue
            lines.append(
                "| {users} | {method} | {thr:.2f} | {delay:.2f} | {ho:.4f} | {fw:.3f} | {gw:.3f} | {hw:.3f} |".format(
                    users=users,
                    method=method,
                    thr=float(row["avg_allocated_mhz"]) * 1.0e6,
                    delay=float(row["avg_delay_ms"]),
                    ho=float(row["handover_frequency"]),
                    fw=float(row.get("avg_f_weight") or 0),
                    gw=float(row.get("avg_g_weight") or 0),
                    hw=float(row.get("avg_h_weight") or 0),
                )
            )
    return "\n".join(lines) + "\n"


def candidate_diagnostics(title: str, csv_path: Path, compare_path: Path) -> str:
    rows = read_rows(csv_path)
    compare_rows = read_rows(compare_path)
    if not rows:
        return f"### {title}\n\nNo candidate CSV found: `{csv_path.name}`.\n"
    lines = [
        f"### {title}",
        "",
        "| UEs | Allocated (bps) | Requested (bps) | Delay(ms) | Handover | Source HO | Dest HO |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {users} | {alloc:.2f} | {req:.2f} | {delay:.2f} | {ho:.4f} | {sho:.4f} | {dho:.4f} |".format(
                users=int(row["users"]),
                alloc=float(row["avg_allocated_mhz"]) * 1.0e6,
                req=float(row.get("avg_requested_mhz") or 0) * 1.0e6,
                delay=float(row["avg_delay_ms"]),
                ho=float(row["handover_frequency"]),
                sho=float(row.get("source_handover_frequency") or 0),
                dho=float(row.get("destination_handover_frequency") or 0),
            )
        )
    lines.extend(["", error_summary(compare_path)])
    throughput_gaps = [
        row for row in compare_rows
        if row.get("metric") == "throughput" and float(row.get("rel_error", 0)) > 0.15
    ]
    if throughput_gaps:
        lines.extend([
            "",
            "Throughput points still outside 15% relative error:",
            "",
            "| UEs | Target | Result | Relative error |",
            "|---:|---:|---:|---:|",
        ])
        for row in throughput_gaps:
            lines.append(
                f"| {int(row['users'])} | {float(row['target']):.2f} | "
                f"{float(row['result']):.2f} | {float(row['rel_error']):.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results") / "final_reproduction_report.md")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    results = args.results if args.results.is_absolute() else base / args.results
    out = args.out if args.out.is_absolute() else base / args.out

    fig3_csv = results / "final_fig3_A.csv"
    fig4_csv = results / "final_fig4_B.csv"
    fig3_png = results / "final_fig3_A.png"
    fig4_png = results / "final_fig4_B.png"
    cmp3 = results / "compare_final_fig3_A.csv"
    cmp4 = results / "compare_final_fig4_B.csv"
    rebuild3_csv = results / "paper_rebuild_fig3_A.csv"
    rebuild4_csv = results / "paper_rebuild_fig4_B.csv"
    rebuild3_png = results / "paper_rebuild_fig3_A.png"
    rebuild4_png = results / "paper_rebuild_fig4_B.png"
    rebuild_cmp3 = results / "compare_paper_rebuild_fig3_A.csv"
    rebuild_cmp4 = results / "compare_paper_rebuild_fig4_B.csv"
    candidate_b_csv = results / "final_candidate_cahs_B_full400.csv"
    candidate_b_cmp = results / "compare_final_candidate_cahs_B_full400.csv"

    fig3_rows = read_rows(fig3_csv)
    fig4_rows = read_rows(fig4_csv)
    rebuild3_rows = read_rows(rebuild3_csv)
    rebuild4_rows = read_rows(rebuild4_csv)

    content = f"""# Final LEO Handover Reproduction Report

## Source-adjusted configuration

The source-adjusted run keeps the original code path and applies the smallest set of explicit settings that made the numerical baselines stable:

- `traffic_mode=ground_backbone`
- `gateway_count=4`
- `LEO_ISL_BANDWIDTH=4000`
- `C_BAND=500`
- `HANDOVER_COST=0.1`
- `DIRECT_PATH_QUALITY=2.0`
- `AGC_PREDICT_WINDOW=30`
- `slots=400`
- `users=100..600 step 50`
- `methods=RS,MSTS,MGCS,CAHS`

This configuration was selected because `ground_backbone` made propagation delay match the paper much better than UE-to-UE traffic, and `ISL=4000` kept baseline high-load throughput closer to the digitized Fig.4 targets than the source default `2000` or the commented `10000`.

## Paper-rebuild configuration

The paper-rebuild path is independent of `run_experiments.py` and uses `run_paper_rebuild.py` plus `paper_model.py`. It treats the paper formulas and digitized Fig.3/Fig.4 targets as the reference, while preserving the old source-adjusted outputs for comparison.

- GSL capacity uses `C_BAND=500`, `PtGtGs=80 dB`, `noise=-173 dBm/Hz`, and atmospheric loss as explicit parameters.
- `IPQ/SQ` is computed as a shortest-path product of remaining ISL bandwidth quality.
- `AGC` is the average capacity over the next `Delta=30s` by default.
- CAHS utility is `F=f*IPQ + g*AGC + h*handover_control` with default `h=0.1`.
- Delay uses actual GSL plus per-hop ISL distances.
- The default traffic model is `ground_backbone`; `gateway_count=4` remains an unpublished paper-detail assumption.

## Current best B-candidate after gateway diagnostics

The latest B-candidate keeps CAHS on the source path and treats the destination half of the synthetic users as ground-backbone endpoints. It incorporates the useful part of `地面站实现参考包.zip` without transplanting the separate博士论文 Gateway state machine:

- `LEO_SOURCE_LAYOUT=random`
- `LEO_SOURCE_LAT_MIN=-63`, `LEO_SOURCE_LAT_MAX=63`
- `LEO_SOURCE_LON_MIN=-180`, `LEO_SOURCE_LON_MAX=180`
- `LEO_GATEWAY_LAYOUT=left_open`
- `LEO_GATEWAY_ASSIGN=cycle`
- `LEO_DESTINATION_DECISION_MODE=SERVICE_TIME`
- `LEO_IPQ_MODE=pldr_lifetime`
- `LEO_IPQ_LOAD_THRESHOLD=300`
- `LEO_ISL_BANDWIDTH=8000`
- `LEO_HANDOVER_COST=0.18`
- `LEO_HANDOVER_COST_HIGH_LOAD=0.45`
- `LEO_HANDOVER_COST_LOAD_THRESHOLD=500`
- `LEO_DIRECT_PATH_QUALITY=4.0`

Reference-package checks showed that the real six-station coordinates and antenna-weighted assignment improve delay but do not solve the Fig.4 throughput gap. `DELAY`-based destination selection improves some throughput points but causes ping-pong-like source handovers, so the current candidate keeps `SERVICE_TIME`.

## Traceability

{metadata_block(fig3_csv)}
{metadata_block(fig4_csv)}
{metadata_block(rebuild3_csv)}
{metadata_block(rebuild4_csv)}

Result taxonomy:

- **paper target digitized**: values digitized from the published Fig.3/Fig.4 curves.
- **source-adjusted**: the released source path with explicit stability/configuration fixes.
- **paper-rebuild**: the independent implementation of formulas stated in the paper.
- **calibrated hidden-assumption candidate**: paper-rebuild plus load-dependent or unpublished assumptions used only for diagnostics/calibration.
- Figures labeled `combined` or `diagnostic` are not final same-configuration reproduction results.

Completeness:

- `{fig3_csv.name}`: {completeness(fig3_rows)}
- `{fig4_csv.name}`: {completeness(fig4_rows)}
- `{rebuild3_csv.name}`: {completeness(rebuild3_rows)}
- `{rebuild4_csv.name}`: {completeness(rebuild4_rows)}

## Final figures

### Fig.3 / Constellation A

![Final Fig.3 A]({fig3_png.as_posix()})

### Fig.4 / Constellation B

![Final Fig.4 B]({fig4_png.as_posix()})

## Paper comparison

Targets were digitized from the paper figures into `paper_targets_fig3_A.csv` and `paper_targets_fig4_B.csv`.

{error_summary(cmp3)}

{error_summary(cmp4)}

## Paper-rebuild comparison

{error_summary(rebuild_cmp3)}

{error_summary(rebuild_cmp4)}

## Current best B-candidate comparison

{candidate_diagnostics("CAHS-only Fig.4/B full 400-slot candidate", candidate_b_csv, candidate_b_cmp)}

## CAHS vs MGCS

{cahs_mgcs_table("Fig.3 / Constellation A", fig3_csv)}

{cahs_mgcs_table("Fig.4 / Constellation B", fig4_csv)}

## Paper-rebuild CAHS vs MGCS

{cahs_mgcs_table("Paper-rebuild Fig.3 / Constellation A", rebuild3_csv)}

{cahs_mgcs_table("Paper-rebuild Fig.4 / Constellation B", rebuild4_csv)}

## Key points

{key_points_table("Fig.3 / Constellation A", fig3_csv)}

{key_points_table("Fig.4 / Constellation B", fig4_csv)}

## Paper-rebuild key points

{key_points_table("Paper-rebuild Fig.3 / Constellation A", rebuild3_csv)}

{key_points_table("Paper-rebuild Fig.4 / Constellation B", rebuild4_csv)}

## Reproduction status

- Fig.3/A is the closer reproduction: CAHS throughput mean relative error is about 10%, and CAHS delay mean relative error is about 4%.
- Fig.4/B remains imperfect: delay is close, but throughput is still low and handover frequencies are not fully aligned.
- The latest Fig.4/B CAHS-only candidate improves high-load behavior, but 200-450 users remain throughput-limited. At several mid-load points, the simulated requested capacity is already below the digitized paper throughput target, so CAHS cannot reach the paper value without a different traffic/GSL demand model.
- The remaining gap is most likely from hidden simulation details not specified in the paper: gateway/backbone layout, traffic generation, and the exact IPQ/SQ implementation.
- DQN/DRQN was intentionally left out of this final numerical baseline reproduction.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
