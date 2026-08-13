import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = ["RS", "MSTS", "MGCS", "CAHS"]
STYLE = {
    "RS": ("black", "o"),
    "MSTS": ("green", "s"),
    "MGCS": ("blue", "^"),
    "CAHS": ("red", "d"),
}
METRIC_TO_RESULT_FIELD = {
    "throughput": "avg_allocated_mhz",
    "delay": "avg_delay_ms",
    "handover": "handover_frequency",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_targets(path: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in read_csv(path):
        key = (row["method"], row["metric"])
        grouped.setdefault(key, []).append((int(row["users"]), float(row["value"])))
    for values in grouped.values():
        values.sort()
    return grouped


def read_results(paths: list[Path]) -> dict[tuple[str, str], list[tuple[int, float]]]:
    grouped: dict[tuple[str, str], dict[int, float]] = {}
    for path in paths:
        for row in read_csv(path):
            method = row["method"]
            if method not in METHODS:
                continue
            users = int(row["users"])
            for metric, field in METRIC_TO_RESULT_FIELD.items():
                grouped.setdefault((method, metric), {})[users] = float(row[field])
    return {key: sorted(values.items()) for key, values in grouped.items()}


def scale_metric(metric: str, values: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if metric == "throughput":
        return [(users, value * 1.0e6) for users, value in values]
    return values


def plot_series(ax, values, method: str, metric: str, label: str, linestyle: str, alpha: float = 1.0) -> None:
    values = scale_metric(metric, values)
    color, marker = STYLE[method]
    ax.plot(
        [users for users, _ in values],
        [value for _, value in values],
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.9,
        markersize=4.8,
        alpha=alpha,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Fig.4/B paper target vs paper-rebuild")
    args = parser.parse_args()

    targets = read_targets(args.targets)
    results = read_results(args.results)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metric_specs = [
        ("throughput", "Overall throughput of all data flows (bps)"),
        ("delay", "Average propagation delay (ms)"),
        ("handover", "Average handover frequency per slot"),
    ]

    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for method in METHODS:
            target_values = targets.get((method, metric))
            if target_values:
                plot_series(ax, target_values, method, metric, f"{method} paper", "-")
            result_values = results.get((method, metric))
            if result_values:
                plot_series(ax, result_values, method, metric, f"{method} rebuild", "--", alpha=0.85)
        ax.set_xlabel("Number of UEs")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if metric == "handover":
            ax.set_ylim(0.0, 0.95)
        if metric == "throughput":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(args.title)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))

    out = args.out
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()
