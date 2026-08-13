import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def last_line(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return text[-1] if text else ""


def progress_bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else width
    return "#" * filled + "." * (width - filled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constellation", choices=["A", "B"], required=True)
    parser.add_argument("--users", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--slots", type=int, default=400)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    results_dir = BASE / "results" / "jobs"
    logs_dir = BASE / "results" / "job_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = 0
    for users in args.users:
        for method in args.methods:
            job_csv = results_dir / f"{args.constellation}_{users}_{method}.csv"
            if read_csv(job_csv):
                print(f"skip existing users={users} method={method}", flush=True)
                skipped += 1
                continue
            pending.append((users, method, job_csv, logs_dir / f"{args.constellation}_{users}_{method}.log"))

    running: list[tuple[int, str, Path, Path, subprocess.Popen]] = []
    total_jobs = skipped + len(pending)
    completed = skipped
    while pending or running:
        while pending and len(running) < args.parallel:
            users, method, job_csv, log_path = pending.pop(0)
            log_file = log_path.open("w", encoding="utf-8")
            env = dict(**{k: v for k, v in __import__("os").environ.items()})
            env["LEO_PROGRESS_EVERY"] = str(args.progress_every)
            cmd = [
                sys.executable,
                "run_experiments.py",
                "--constellation",
                args.constellation,
                "--users",
                str(users),
                "--methods",
                method,
                "--slots",
                str(args.slots),
                "--jobs",
                "1",
                "--out",
                str(job_csv.relative_to(BASE)),
            ]
            proc = subprocess.Popen(cmd, cwd=BASE, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            log_file.close()
            running.append((users, method, job_csv, log_path, proc))
            print(f"[{progress_bar(completed, total_jobs)}] {completed}/{total_jobs} complete; started users={users} method={method}", flush=True)

        time.sleep(30)
        still_running = []
        for users, method, job_csv, log_path, proc in running:
            code = proc.poll()
            print(f"status users={users} method={method} code={code} last='{last_line(log_path)}'", flush=True)
            if code is None:
                still_running.append((users, method, job_csv, log_path, proc))
            elif code != 0:
                raise SystemExit(f"job failed users={users} method={method}, see {log_path}")
            else:
                completed += 1
                print(f"[{progress_bar(completed, total_jobs)}] {completed}/{total_jobs} complete", flush=True)
        running = still_running

    rows = []
    for path in sorted(results_dir.glob(f"{args.constellation}_*.csv")):
        rows.extend(read_csv(path))
    rows.sort(key=lambda r: (int(r["users"]), str(r["method"])))
    out = args.out if args.out.is_absolute() else BASE / args.out
    write_csv(rows, out)
    print(out, flush=True)


if __name__ == "__main__":
    main()
