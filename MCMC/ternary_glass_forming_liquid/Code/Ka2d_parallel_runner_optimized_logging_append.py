from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Sequence, Tuple


"""
Parallel runner for Ka2d_random_seed.py. Each run gets a unique MC seed,
while the type shuffling uses the fixed seed inside Ka2d_random_seed.

Edit the CONFIG section below and run:
  python Simulation_repo/MCMC/ternary_glass_forming_liquid/Code/Ka2d_parallel_runner.py
"""

# ============================== CONFIG ==============================
# Output
OUT_DIR = "/Users/xuhengyuan/Downloads/Notes/Simulation_repo/MCMC/ternary_glass_forming_liquid/Data/Parallel_Ka2d_runs/N43_L6_beta2"
OVERWRITE = False
SUMMARY_FILENAME = "summary.jsonl"

# Parallelism
WORKERS = 10               # 0 => os.cpu_count()
NUMBA_THREADS = 2         # threads per worker process
DISABLE_TQDM = True       # suppress per-worker progress bars

# Simulation sweep/sample settings
N_SAMPLES = 50000
BURN_IN_SWEEPS = 2_000_000
SAMPLE_INTERVAL = 2000
SWEEPS = 0                # used only when N_SAMPLES == 0

# System/grid
N = 43
RHO = 43/36
RCUT_FACTOR = 2.5
MAX_DISP = 0.2
TARGET_ACCEPT = 0.2

# Parameter set + composition
TERNARY = True
COMPOSITION_RATIO = (20,11,12)  # ternary: 3 ints; binary: 2 ints

# Job grid / naming
TEMPS = [0.5]
SEED = 2026
REPLICATES = 10
SERIES = "50Ksample_training"
# ============================ END CONFIG ============================


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    N: int
    T: float
    rho: float
    sweeps: int
    seed: int
    ternary: bool
    composition_ratio: Tuple[int, ...]
    n_samples: int
    burn_in_sweeps: int
    sample_interval: int
    rcut_factor: float
    max_disp: float
    target_accept: float
    save_path: str


def _ensure_import_path() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)


def _set_thread_env(num_threads: int) -> None:
    nt = str(int(num_threads))
    os.environ.setdefault("NUMBA_NUM_THREADS", nt)
    os.environ.setdefault("OMP_NUM_THREADS", nt)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", nt)
    os.environ.setdefault("MKL_NUM_THREADS", nt)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", nt)


def _run_one(job: JobSpec, numba_threads: int, disable_tqdm: bool) -> dict:
    _set_thread_env(numba_threads)
    _ensure_import_path()

    import importlib
    import numpy as np

    ka = importlib.import_module("Ka2d_random_seed_optimized_logging_append")

    os.makedirs(os.path.dirname(job.save_path), exist_ok=True)
    root, _ = os.path.splitext(job.save_path)
    log_path = root + "__worker.log"
    out = ka.run_ka2d_mc(
        N=job.N,
        T=job.T,
        sweeps=job.sweeps,
        rho=job.rho,
        seed=job.seed,
        ternary=job.ternary,
        composition_ratio=job.composition_ratio,
        n_samples=job.n_samples,
        burn_in_sweeps=job.burn_in_sweeps,
        sample_interval=job.sample_interval,
        rcut_factor=job.rcut_factor,
        max_disp=job.max_disp,
        target_accept=job.target_accept,
        save_samples_path=job.save_path,
        log_path=log_path,
        job_id=job.job_id,
    )

    acc_mean = out.get("acceptance_mean", None)
    if acc_mean is None:
        acc_path = out.get("acceptance_path", None)
        if acc_path:
            acc_arr = np.load(acc_path, mmap_mode="r")
            acc_mean = float(np.mean(acc_arr)) if acc_arr.size else None
            del acc_arr
    final_max_disp = out.get("final_max_disp", None)
    final_max_disp = float(final_max_disp) if final_max_disp is not None else None
    manifest_path = out.get("manifest", None)
    return {
        "job": asdict(job),
        "acceptance_mean": acc_mean,
        "final_max_disp": final_max_disp,
        "manifest": manifest_path,
        "saved": job.save_path,
        "log_path": log_path,
    }


def _job_id(N: int, rho: float, T: float, seed: int) -> str:
    return f"N{N}_rho{rho:g}_T{T:g}_seed{seed}"

def _fmt_float_token(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def _fmt_samples_token(n: int) -> str:
    n = int(n)
    if n % 1_000_000 == 0 and n != 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0 and n != 0:
        return f"{n // 1_000}k"
    return str(n)


def _default_basename(*, N: int, T: float, rho: float, n_samples: int) -> str:
    return f"N{N}_T{_fmt_float_token(T)}_rho{_fmt_float_token(rho)}_sample{_fmt_samples_token(n_samples)}"


def build_jobs(
    *,
    out_dir: str,
    N: int,
    rho: float,
    Ts: Sequence[float],
    seed: int,
    replicates: int,
    series: str,
    sweeps: int,
    ternary: bool,
    composition_ratio: Tuple[int, ...],
    n_samples: int,
    burn_in_sweeps: int,
    sample_interval: int,
    rcut_factor: float,
    max_disp: float,
    target_accept: float,
    overwrite: bool,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    os.makedirs(out_dir, exist_ok=True)
    base_seed = int(seed)
    seed_counter = base_seed
    nrep = int(replicates)
    for T in Ts:
        base = _default_basename(N=N, T=float(T), rho=float(rho), n_samples=int(n_samples))
        for r in range(1, nrep + 1):
            if series:
                jid = f"{base}_{series}.{r}"
            else:
                jid = f"{base}_{r}"
            # Place each job's outputs in its own folder to avoid clutter:
            #   <out_dir>/<job_id>/data.npz (with manifest/chunks alongside) and worker.log
            job_dir = os.path.join(out_dir, jid)
            save_path = os.path.join(job_dir, "data.npz")
            if (not overwrite) and os.path.exists(save_path):
                continue
            jobs.append(
                JobSpec(
                    job_id=jid,
                    N=N,
                    T=float(T),
                    rho=float(rho),
                    sweeps=int(sweeps),
                    seed=seed_counter,  # different MC seed per run_id; types seed fixed in Ka2d_random_seed
                    ternary=bool(ternary),
                    composition_ratio=tuple(int(x) for x in composition_ratio),
                    n_samples=int(n_samples),
                    burn_in_sweeps=int(burn_in_sweeps),
                    sample_interval=int(sample_interval),
                    rcut_factor=float(rcut_factor),
                    max_disp=float(max_disp),
                    target_accept=float(target_accept),
                    save_path=save_path,
                )
            )
            seed_counter += 1
    return jobs


def main(argv: Iterable[str] | None = None) -> int:
    if argv is not None and list(argv):
        raise SystemExit("This script is configured via globals at the top; do not pass CLI args.")

    out_dir = os.path.abspath(OUT_DIR)
    if not TEMPS:
        raise SystemExit("CONFIG error: TEMPS is empty.")
    if REPLICATES <= 0:
        raise SystemExit("CONFIG error: REPLICATES must be >= 1.")
    if TERNARY and len(COMPOSITION_RATIO) != 3:
        raise SystemExit("CONFIG error: COMPOSITION_RATIO must have 3 ints for ternary.")
    if (not TERNARY) and len(COMPOSITION_RATIO) != 2:
        raise SystemExit("CONFIG error: COMPOSITION_RATIO must have 2 ints for binary.")

    jobs = build_jobs(
        out_dir=out_dir,
        N=N,
        rho=RHO,
        Ts=TEMPS,
        seed=SEED,
        replicates=REPLICATES,
        series=SERIES,
        sweeps=SWEEPS,
        ternary=TERNARY,
        composition_ratio=COMPOSITION_RATIO,
        n_samples=N_SAMPLES,
        burn_in_sweeps=BURN_IN_SWEEPS,
        sample_interval=SAMPLE_INTERVAL,
        rcut_factor=RCUT_FACTOR,
        max_disp=MAX_DISP,
        target_accept=TARGET_ACCEPT,
        overwrite=OVERWRITE,
    )
    if not jobs:
        print("No jobs to run (all outputs exist and --overwrite not set).")
        return 0

    workers = WORKERS if WORKERS and WORKERS > 0 else (os.cpu_count() or 1)
    summary_path = os.path.join(out_dir, SUMMARY_FILENAME)

    print(f"Jobs: {len(jobs)} | workers: {workers} | out: {out_dir}")
    with ProcessPoolExecutor(max_workers=workers) as ex, open(summary_path, "a", encoding="utf-8") as fsum:
        futures = [
            ex.submit(_run_one, job, NUMBA_THREADS, bool(DISABLE_TQDM))
            for job in jobs
        ]
        done = 0
        for fut in as_completed(futures):
            rec = fut.result()
            fsum.write(json.dumps(rec) + "\n")
            fsum.flush()
            jid = rec.get("job", {}).get("job_id", "<unknown>")
            acc_mean = rec.get("acceptance_mean", None)
            fmd = rec.get("final_max_disp", None)
            manifest = rec.get("manifest", None)
            acc_s = f"{acc_mean:.6g}" if acc_mean is not None else "n/a"
            fmd_s = f"{fmd:.6g}" if fmd is not None else "n/a"
            extra = f" manifest={manifest}" if manifest else ""
            print(f"{jid}: acc={acc_s}  final_max_disp={fmd_s}{extra}")
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"Completed {done}/{len(futures)}")

    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
