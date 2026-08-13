import argparse
import csv
import math
from pathlib import Path


METRIC_MAP = {
    "throughput": "avg_allocated_mhz",
    "delay": "avg_delay_ms",
    "handover": "handover_frequency",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    targets = read_csv(args.targets)
    results = read_csv(args.results)
    result_map = {(r["method"], int(r["users"])): r for r in results}

    rows = []
    for target in targets:
        method = target["method"]
        users = int(target["users"])
        metric = target["metric"]
        result = result_map.get((method, users))
        if result is None:
            continue
        target_value = float(target["value"])
        if math.isnan(target_value):
            continue
        result_value = float(result[METRIC_MAP[metric]])
        abs_error = result_value - target_value
        rel_error = abs(abs_error) / abs(target_value) if target_value else 0.0
        rows.append(
            {
                "figure": target["figure"],
                "metric": metric,
                "method": method,
                "users": users,
                "target": target_value,
                "result": result_value,
                "abs_error": abs_error,
                "rel_error": rel_error,
            }
        )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = ["figure", "metric", "method", "users", "target", "result", "abs_error", "rel_error"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for row in rows:
        groups.setdefault((row["metric"], row["method"]), []).append(float(row["rel_error"]))
    print(out)
    for key in sorted(groups):
        vals = groups[key]
        print(f"{key[0]} {key[1]} mean_rel={sum(vals)/len(vals):.4f} max_rel={max(vals):.4f}")


if __name__ == "__main__":
    main()
