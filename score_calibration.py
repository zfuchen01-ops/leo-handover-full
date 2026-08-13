import argparse
import csv
import json
from pathlib import Path


PRIMARY_METHODS = {"MGCS", "CAHS"}
WEIGHTS = {"delay": 0.50, "throughput": 0.30, "handover": 0.20}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_errors(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["method"] not in PRIMARY_METHODS:
            continue
        grouped.setdefault(row["metric"], []).append(float(row["rel_error"]))
    return {
        metric: sum(values) / len(values)
        for metric, values in grouped.items()
        if values
    }


def candidate_score(summary: dict[str, float]) -> float:
    return sum(WEIGHTS[metric] * summary[metric] for metric in WEIGHTS)


def passes_guardrails(
    candidate: dict[str, float],
    baseline: dict[str, float],
    tolerance: float = 0.05,
) -> bool:
    return all(
        candidate[metric] <= baseline[metric] + tolerance
        for metric in ("throughput", "handover")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("compare_csv", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summary = aggregate_errors(read_rows(args.compare_csv))
    result = {
        "compare_csv": str(args.compare_csv),
        "weights": WEIGHTS,
        "summary": summary,
        "score": candidate_score(summary),
    }
    if args.baseline:
        baseline_summary = aggregate_errors(read_rows(args.baseline))
        result["baseline_compare_csv"] = str(args.baseline)
        result["baseline_summary"] = baseline_summary
        result["passes_guardrails"] = passes_guardrails(summary, baseline_summary)

    rendered = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(args.out)
    print(rendered)


if __name__ == "__main__":
    main()
