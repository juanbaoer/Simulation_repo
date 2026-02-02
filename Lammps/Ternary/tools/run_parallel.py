#!/usr/bin/env python3
"""Run embarrassingly parallel LAMMPS jobs with per-run seeds."""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _find_ternary_root(in_file: Path) -> Path | None:
    p = in_file.resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "config" / "ka2d.table").exists():
            return parent
    return None


def _next_runs_dir(root: Path, tag: str | None) -> Path:
    date_tag = tag or _dt.date.today().isoformat()
    for i in range(1, 1000):
        candidate = root / f"runs_{date_tag}_{i:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to find a free runs directory (001-999).")


def _resolve_lmp(lmp: str) -> str:
    if os.path.isabs(lmp) or "/" in lmp:
        if not os.path.exists(lmp):
            raise FileNotFoundError(f"LAMMPS executable not found: {lmp}")
        return lmp
    found = shutil.which(lmp)
    if not found:
        raise FileNotFoundError(f"LAMMPS executable not found in PATH: {lmp}")
    return found


def _physical_cores() -> int | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True)
        return int(out.strip())
    except Exception:
        return None


def _default_workers() -> int:
    phys = _physical_cores()
    if phys:
        return max(1, phys - 1)
    return max(1, (os.cpu_count() or 1) - 1)


def _build_cmd(lmp: str, in_file: Path, seed: int, runid: str) -> list[str]:
    return [
        lmp,
        "-in",
        str(in_file),
        "-var",
        "seed",
        str(seed),
        "-var",
        "runid",
        runid,
        "-log",
        "log.lammps",
        "-screen",
        "screen.out",
    ]


def _is_done(run_dir: Path, runid: str) -> bool:
    if (run_dir / "done.flag").exists():
        return True
    if (run_dir / f"T0p5_sample_{runid}.restart").exists():
        return True
    return False


def _load_meta(run_dir: Path) -> dict | None:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _write_meta(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _run_one(job: dict) -> dict:
    run_dir: Path = job["run_dir"]
    runid: str = job["runid"]
    seed: int = job["seed"]
    in_file: Path = job["in_file"]
    lmp: str = job["lmp"]
    env: dict = job["env"]
    dry_run: bool = job["dry_run"]

    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_meta.json"

    meta = {
        "run_id": runid,
        "seed": seed,
        "run_dir": str(run_dir),
        "in_file": str(in_file),
        "lmp": lmp,
        "command": _build_cmd(lmp, in_file, seed, runid),
        "status": "running",
        "start_time": _dt.datetime.now().isoformat(),
    }
    _write_meta(meta_path, meta)

    if dry_run:
        meta["status"] = "dry_run"
        meta["end_time"] = _dt.datetime.now().isoformat()
        meta["duration_sec"] = 0.0
        _write_meta(meta_path, meta)
        return meta

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    start = time.time()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(
            _build_cmd(lmp, in_file, seed, runid),
            cwd=run_dir,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )

    duration = time.time() - start
    meta["returncode"] = proc.returncode
    meta["duration_sec"] = round(duration, 3)
    meta["end_time"] = _dt.datetime.now().isoformat()
    if proc.returncode == 0:
        meta["status"] = "success"
        (run_dir / "done.flag").write_text("ok\n")
    else:
        meta["status"] = "failed"

    _write_meta(meta_path, meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parallel LAMMPS jobs.")
    parser.add_argument("--n_runs", type=int, required=True, help="Number of runs")
    parser.add_argument("--base_seed", type=int, required=True, help="Base seed")
    parser.add_argument("--stride", type=int, default=100, help="Seed stride")
    parser.add_argument("--lmp", required=True, help="LAMMPS executable (e.g., lmp_mpi)")
    parser.add_argument("--in", dest="in_file", required=True, help="LAMMPS input file")
    parser.add_argument("--max_workers", type=int, default=0, help="Max parallel workers")
    parser.add_argument("--start_id", type=int, default=1, help="Start run id")
    parser.add_argument("--runid_width", type=int, default=4, help="Zero padding width")
    parser.add_argument("--tag", default=None, help="Runs directory tag (YYYY-MM-DD or custom)")
    parser.add_argument("--dry_run", action="store_true", help="Print plan only")
    parser.add_argument("--rerun_failed", action="store_true", help="Rerun failed runs")
    parser.add_argument("--omp_threads", type=int, default=1, help="OMP_NUM_THREADS")

    args = parser.parse_args()

    in_file = Path(args.in_file).expanduser().resolve()
    if not in_file.exists():
        print(f"Input file not found: {in_file}", file=sys.stderr)
        return 2

    ternary_root = _find_ternary_root(in_file)
    if not ternary_root:
        print("Unable to locate Ternary root (missing config/ka2d.table).", file=sys.stderr)
        return 2

    lmp = _resolve_lmp(args.lmp)

    runs_root = _next_runs_dir(ternary_root, args.tag)

    # Ensure relative table path works from run directories
    test_run_dir = runs_root / f"run_{'0'*args.runid_width}"
    test_table = (test_run_dir / "../../config/ka2d.table").resolve()
    if not test_table.exists():
        print(
            "Relative table path check failed. From a run dir, ../../config/ka2d.table was not found.\n"
            f"Expected at: {test_table}",
            file=sys.stderr,
        )
        return 2

    max_workers = args.max_workers if args.max_workers > 0 else _default_workers()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)

    jobs: list[dict] = []
    for i in range(args.start_id, args.start_id + args.n_runs):
        runid = str(i).zfill(args.runid_width)
        seed = args.base_seed + (i - args.start_id) * args.stride
        run_dir = runs_root / f"run_{runid}"

        if run_dir.exists():
            if _is_done(run_dir, runid):
                jobs.append({
                    "runid": runid,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "status": "skipped_done",
                })
                continue
            if args.rerun_failed:
                meta = _load_meta(run_dir)
                if meta and meta.get("status") == "failed":
                    pass
                else:
                    jobs.append({
                        "runid": runid,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "status": "skipped_existing",
                    })
                    continue
            else:
                jobs.append({
                    "runid": runid,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "status": "skipped_existing",
                })
                continue

        jobs.append({
            "runid": runid,
            "seed": seed,
            "run_dir": run_dir,
            "in_file": in_file,
            "lmp": lmp,
            "env": env,
            "dry_run": args.dry_run,
        })

    if args.dry_run:
        print(f"Runs root: {runs_root}")
        for job in jobs:
            if job.get("status"):
                print(f"SKIP {job['runid']}: {job['status']}")
                continue
            cmd = _build_cmd(lmp, in_file, job["seed"], job["runid"])
            print(f"RUN {job['runid']} seed={job['seed']} dir={job['run_dir']}")
            print("  " + " ".join(cmd))
        return 0

    runs_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = []
        for job in jobs:
            if job.get("status"):
                results.append(job)
                continue
            futures.append(exe.submit(_run_one, job))
        for fut in as_completed(futures):
            results.append(fut.result())

    # Write summary
    summary_path = runs_root / "summary.json"
    # Normalize any Path objects in results for JSON output
    normalized = []
    for r in results:
        r2 = dict(r)
        if isinstance(r2.get("run_dir"), Path):
            r2["run_dir"] = str(r2["run_dir"])
        normalized.append(r2)
    summary_path.write_text(json.dumps(normalized, indent=2, sort_keys=True))

    csv_path = runs_root / "summary.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "run_id",
            "seed",
            "status",
            "returncode",
            "duration_sec",
            "run_dir",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in normalized:
            w.writerow({
                "run_id": r.get("run_id") or r.get("runid"),
                "seed": r.get("seed"),
                "status": r.get("status"),
                "returncode": r.get("returncode"),
                "duration_sec": r.get("duration_sec"),
                "run_dir": r.get("run_dir"),
            })

    print(f"Done. Results: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
