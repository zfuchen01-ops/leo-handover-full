import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


STYLE = {
    "RS": ("black", "o"),
    "MSTS": ("green", "s"),
    "MGCS": ("blue", "^"),
    "CAHS": ("red", "d"),
    "UNION_MODE_1": ("purple", "v"),
    "DQN": ("orange", "x"),
    "DRQN": ("tab:brown", "*"),
}
METHOD_ORDER = ["RS", "MSTS", "MGCS", "CAHS"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ordered_groups(grouped):
    methods = METHOD_ORDER + sorted(method for method in grouped if method not in METHOD_ORDER)
    return [(method, grouped[method]) for method in methods if method in grouped]


def metric_value(metric: str, value: str) -> float:
    parsed = float(value)
    return parsed * 1.0e6 if metric.endswith("_mhz") else parsed


def plot_metric(ax, grouped, metric: str, ylabel: str, ylim=None) -> None:
    for method, rows in ordered_groups(grouped):
        rows = sorted(rows, key=lambda r: int(r["users"]))
        color, marker = STYLE.get(method, (None, "o"))
        ax.plot(
            [int(r["users"]) for r in rows],
            [metric_value(metric, r[metric]) for r in rows],
            label=method,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--throughput", choices=["avg_requested_mhz", "avg_allocated_mhz", "last_requested_mhz", "last_allocated_mhz"], default="avg_allocated_mhz")
    parser.add_argument("--out", type=Path, default=Path("results") / "curves.png")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for row in read_rows(args.csv):
        grouped[row["method"]].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if args.title:
        fig.suptitle(args.title)
    plot_metric(axes[0], grouped, args.throughput, "Overall throughput of all data flows (bps)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plot_metric(axes[1], grouped, "avg_delay_ms", "Average propagation delay (ms)")
    plot_metric(
        axes[2],
        grouped,
        "handover_frequency",
        "Average handover frequency per slot",
        ylim=(0.0, 0.95),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94) if args.title else None)

    out = args.out
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
