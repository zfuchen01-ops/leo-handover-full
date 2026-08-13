import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=BASE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_jobs = min(8, max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--jobs", type=int, default=default_jobs)
    parser.add_argument("--quick", action="store_true", help="Run a small diagnostic subset before the full paper grid.")
    parser.add_argument("--slots", type=int, default=400)
    parser.add_argument("--variant", choices=["paper", "calibrated"], default="paper")
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    users = ["100", "350", "600"] if args.quick else ["100", "150", "200", "250", "300", "350", "400", "450", "500", "550", "600"]
    suffix = "_quick" if args.quick else ""
    prefix = "" if args.variant == "paper" else "calibrated_"
    run(
        [
            py,
            "run_experiments.py",
            "--constellation",
            "A",
            "--users",
            *users,
            "--methods",
            "RS",
            "MSTS",
            "MGCS",
            "CAHS",
            "--slots",
            str(args.slots),
            "--jobs",
            str(args.jobs),
            "--variant",
            args.variant,
            "--out",
            f"results\\{prefix}fig3_A{suffix}.csv",
        ]
    )
    run([py, "plot_results.py", f"results\\{prefix}fig3_A{suffix}.csv", "--out", f"results\\{prefix}fig3_A{suffix}.png", "--title", f"{args.variant.title()} reproduction - Constellation A{suffix}"])

    run(
        [
            py,
            "run_experiments.py",
            "--constellation",
            "B",
            "--users",
            *users,
            "--methods",
            "RS",
            "MSTS",
            "MGCS",
            "CAHS",
            "--slots",
            str(args.slots),
            "--jobs",
            str(args.jobs),
            "--variant",
            args.variant,
            "--out",
            f"results\\{prefix}fig4_B{suffix}.csv",
        ]
    )
    run([py, "plot_results.py", f"results\\{prefix}fig4_B{suffix}.csv", "--out", f"results\\{prefix}fig4_B{suffix}.png", "--title", f"{args.variant.title()} reproduction - Constellation B{suffix}"])
    if not args.quick:
        run([py, "make_report.py"])

    if args.notify:
        run([py, "notify_qq.py", f"LEO论文完整复现实验已完成，报告: {BASE / 'results' / 'reproduction_report.md'}"])


if __name__ == "__main__":
    main()
