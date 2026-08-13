import argparse
import json
import time
from pathlib import Path

from overnight_handover_screen import (
    Candidate,
    FULL_USERS,
    KEY_USERS,
    RESULTS,
    ScreenRunner,
    candidate_from_summary,
    merge_final_rows,
    plot_diagnostics,
    read_csv,
    write_acceptance_report,
    write_rankings,
)


FOCUSED_RESULTS = RESULTS / "focused"


def rs_candidates(constellation: str) -> list[Candidate]:
    return [
        Candidate("RS", constellation, f"destination_mode_{mode.lower()}", {"LEO_DESTINATION_DECISION_MODE": mode})
        for mode in ("CHANNEL_QUALITY", "DISTANCE", "ELEVATION")
    ]


def a_mgcs_candidates() -> list[Candidate]:
    baseline = {
        "LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS": "4",
        "LEO_CHANNEL_QUALITY_MIN_HOLD_SLOTS_POINTS": "",
        "LEO_GSL_RATE_CAP_POINTS": "100:500,350:1000,600:2200",
    }
    return [
        Candidate("MGCS", "A", "hold4_cap2200", baseline),
        *[
            Candidate(
                "MGCS",
                "A",
                f"hold4_cap2200_delayweight_{str(weight).replace('-', 'm').replace('.', 'p')}",
                {**baseline, "LEO_MGCS_DELAY_WEIGHT": str(weight)},
            )
            for weight in (-0.05, -0.10, -0.20, -0.40, -0.75)
        ],
    ]


def run_family(
    runner: ScreenRunner,
    name: str,
    candidates: list[Candidate],
) -> dict[str, object] | None:
    key80 = runner.run_candidates(candidates, KEY_USERS, 80, f"focused_{name}_key80")
    if not key80:
        return None
    key80.sort(key=lambda row: float(row["score"]))
    full80_candidates = [candidate_from_summary(row) for row in key80[:2]]
    full80 = runner.run_candidates(full80_candidates, FULL_USERS, 80, f"focused_{name}_full80")
    if not full80:
        return key80[0]
    full80.sort(key=lambda row: float(row["score"]))
    winner = candidate_from_summary(full80[0])
    key400 = runner.run_candidate(winner, KEY_USERS, 400, f"focused_{name}_key400")
    if not key400:
        return full80[0]
    if float(key400["score"]) > float(full80[0]["score"]) + 0.05:
        return key400
    return runner.run_candidate(winner, FULL_USERS, 400, f"focused_{name}_full400") or key400


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-hours", type=float, default=4.0)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    runner = ScreenRunner(time.time() + args.budget_hours * 3600, args.jobs)
    results = []
    families = [
        ("A_RS", rs_candidates("A")),
        ("B_RS", rs_candidates("B")),
        ("A_MGCS", a_mgcs_candidates()),
    ]
    for name, candidates in families:
        result = run_family(runner, name, candidates)
        if result:
            results.append(result)
        if runner.time_remaining() <= 60:
            break

    final_path = RESULTS / "rankings" / "final_best.csv"
    existing = read_csv(final_path)
    improvements = []
    existing_map = {(row["constellation"], row["method"]): row for row in existing}
    for result in results:
        key = (str(result["constellation"]), str(result["method"]))
        previous = existing_map.get(key)
        if previous is None or (
            int(result["slots"]) == 400
            and len(str(result["users"]).split()) == len(FULL_USERS)
            and float(result["score"]) < float(previous["score"])
        ):
            improvements.append(result)

    merged = merge_final_rows(existing, improvements)
    write_rankings("final_best", merged)
    write_acceptance_report(merged)
    for result in results:
        candidate = candidate_from_summary(result)
        plot_diagnostics(
            Path(str(result["result_csv"])),
            candidate,
            FOCUSED_RESULTS / f"{candidate.constellation}_{candidate.method}_{candidate.name}.png",
        )
    FOCUSED_RESULTS.mkdir(parents=True, exist_ok=True)
    (FOCUSED_RESULTS / "summary.json").write_text(
        json.dumps({"results": results, "accepted_improvements": improvements}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
