import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


USERS = list(range(100, 601, 50))
METHODS = ["RS", "MSTS", "MGCS", "CAHS"]

# The lower-left legend in Fig. 3(b) covers the first CAHS markers. These
# values are read manually from the visible line position around the legend.
MANUAL_OVERRIDES = {
    ("fig3_A", "delay", "CAHS", 100): 72.15,
    ("fig3_A", "delay", "CAHS", 150): 71.72,
    ("fig3_A", "delay", "CAHS", 200): 71.98,
}

# Page 5 was rendered at 3x from the PDF. The coordinates below are absolute
# pixels in the rendered page. Each plot uses the first and last x tick centers
# plus two labeled y-gridline anchors, which is more stable than cropping the
# whole subplot frame.
FIGURES = {
    "fig3_A": {
        "page": Path("results/paper_pages/zhou_page_5.png"),
        "plots": {
            "throughput": {
                "x_range": (256, 626),
                "y_scale": ((209, 70000.0), (383, 20000.0)),
                "legend": (245, 192, 335, 288),
                "sample_y": (185, 406),
            },
            "delay": {
                "x_range": (740, 1134),
                "y_scale": ((193, 79.0), (385, 72.0)),
                "legend": (728, 318, 830, 395),
                "sample_y": (185, 420),
            },
            "handover": {
                "x_range": (1251, 1638),
                "y_scale": ((207, 0.7), (378, 0.1)),
                "legend": (1560, 250, 1650, 345),
                "sample_y": (185, 406),
            },
        },
    },
    "fig4_B": {
        "page": Path("results/paper_pages/zhou_page_5.png"),
        "plots": {
            "throughput": {
                "x_range": (264, 627),
                "y_scale": ((616, 100000.0), (810, 20000.0)),
                "legend": (252, 614, 342, 708),
                "sample_y": (607, 828),
            },
            "delay": {
                "x_range": (740, 1131),
                "y_scale": ((636, 82.0), (792, 74.0)),
                "legend": (728, 672, 830, 765),
                "sample_y": (607, 828),
            },
            "handover": {
                "x_range": (1251, 1638),
                "y_scale": ((635, 0.8), (779, 0.2)),
                "legend": (1560, 680, 1650, 775),
                "sample_y": (607, 828),
            },
        },
    },
}


def method_mask(hsv: np.ndarray, method: str) -> np.ndarray:
    if method == "CAHS":
        mask1 = cv2.inRange(hsv, np.array([0, 70, 80]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 70, 80]), np.array([180, 255, 255]))
        return mask1 | mask2
    if method == "MSTS":
        return cv2.inRange(hsv, np.array([35, 35, 35]), np.array([95, 255, 210]))
    if method == "MGCS":
        return cv2.inRange(hsv, np.array([95, 35, 35]), np.array([150, 255, 255]))
    if method == "RS":
        gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        return cv2.inRange(gray, 0, 70)
    raise ValueError(method)


def y_to_value(y: float, y_scale: tuple[tuple[int, float], tuple[int, float]]) -> float:
    (y_a, value_a), (y_b, value_b) = y_scale
    return value_a + (y - y_a) / (y_b - y_a) * (value_b - value_a)


def digitize_plot(
    image: np.ndarray,
    x_range: tuple[int, int],
    y_scale: tuple[tuple[int, float], tuple[int, float]],
    sample_y: tuple[int, int],
    legend: tuple[int, int, int, int] | None = None,
) -> dict[str, list[float]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    x_start, x_end = x_range
    y_top, y_bottom = sample_y
    xs = [int(round(x_start + (u - 100) / 500 * (x_end - x_start))) for u in USERS]
    values: dict[str, list[float]] = {m: [] for m in METHODS}
    for method in METHODS:
        mask = method_mask(hsv, method)
        mask[: y_top - 4, :] = 0
        mask[y_bottom + 5 :, :] = 0
        if legend:
            lx0, ly0, lx1, ly1 = legend
            mask[ly0:ly1, lx0:lx1] = 0
        for x in xs:
            left = max(0, x - 8)
            right = min(mask.shape[1], x + 9)
            window = mask[y_top : y_bottom + 1, left:right]
            ys, _ = np.where(window > 0)
            if len(ys) == 0:
                values[method].append(float("nan"))
                continue
            y = float(np.median(ys) + y_top)
            values[method].append(y_to_value(y, y_scale))
    return values


def write_targets(name: str, rows: list[dict[str, float | int | str]], out_dir: Path) -> Path:
    out = out_dir / f"paper_targets_{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = ["figure", "metric", "method", "users", "value"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = args.out_dir if args.out_dir.is_absolute() else base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for figure, cfg in FIGURES.items():
        page = cfg["page"]
        page = page if page.is_absolute() else base / page
        image = cv2.imread(str(page))
        if image is None:
            raise FileNotFoundError(page)
        rows = []
        for metric, plot_cfg in cfg["plots"].items():
            digitized = digitize_plot(
                image,
                plot_cfg["x_range"],
                plot_cfg["y_scale"],
                plot_cfg["sample_y"],
                plot_cfg.get("legend"),
            )
            for method, vals in digitized.items():
                for users, value in zip(USERS, vals):
                    value = MANUAL_OVERRIDES.get((figure, metric, method, users), value)
                    rows.append(
                        {
                            "figure": figure,
                            "metric": metric,
                            "method": method,
                            "users": users,
                            "value": value,
                        }
                    )
        out = write_targets(figure, rows, out_dir)
        print(out)


if __name__ == "__main__":
    main()
