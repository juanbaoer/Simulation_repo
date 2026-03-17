# KA2D Monte Carlo (ternary/binary Lennard-Jones) — Code Notes

## Overview
This directory contains a 2D Kob–Andersen (KA) Lennard-Jones Monte Carlo (MC) simulator with periodic boundary conditions and a smooth cutoff. The core simulation is implemented in Numba-accelerated functions for performance, with Python coordinating configuration, logging, and incremental output.

## File map
- `Ka2d_random_seed_optimized_logging_append.py` — Single-run MC simulation with Numba hot loops, chunked output, and append-style logging.
- `Ka2d_parallel_runner_optimized_logging_append.py` — Parallel launcher that runs many independent single-run simulations across a grid of parameters and seeds.
- `Ka2d_optimzied_energy.py` — Single-run variant with an optimized local dE computation (one interaction traversal per move).
- `benchmark_acceptance_rule.py` — Micro-benchmark for exp-based vs log-based Metropolis acceptance under Numba.
- `validate_optimized_energy.py` — Correctness + timing comparison between the original and optimized single-run implementations.

## Execution flow
### Single run
- `KA2DMCSimulatorNumba.run()` orchestrates the run.
- Work is delegated to `_run_sweeps_numba()`, which calls `_sweep_numba()` for each sweep.
- Each sweep attempts **N** single-particle moves (random particle index per attempt).
- Optional phases:
  - **Burn-in:** `burn_in_sweeps` with no samples saved.
  - **Sampling:** `sample_interval` sweeps per sample; the first window is unsaved, the rest are saved.
- `ChunkedStorage` writes acceptance history and (optionally) sampled positions to `__chunks/` and a manifest JSON.

### Parallel runner
- `ProcessPoolExecutor` launches jobs concurrently.
- Each worker imports the single-run module and calls `run_ka2d_mc(...)`.
- Outputs go to `<OUT_DIR>/<job_id>/data.npz` plus a manifest and chunked arrays.

## Energy / dE computation (current approach)
- Per move, the original code computes:
  - `old_Ei` = particle i energy with all others
  - `new_Ei` = particle i energy after trial move
  - `dE = new_Ei - old_Ei`
- This uses `_particle_energy_numba(...)` twice per move; each call loops over **all N particles** and checks cutoffs.
- Complexity intuition (no neighbor list):
  - **Per attempt:** O(N) pair checks
  - **Per sweep:** O(N^2) checks
- The optimized variant replaces this with a **single traversal** that accumulates both old and new contributions.

## Randomness & reproducibility
- A single base seed is split via `SeedSequence` into:
  - NumPy RNG for initial positions
  - Numba RNG for MC moves (seeded once per run)
- Type shuffling uses a **fixed** seed (`TYPES_TENSOR_SEED`) to keep composition ordering consistent across runs.
- Determinism holds for single-threaded runs; parallel execution can change RNG call order.

## Numba acceleration
Hot loops are Numba-jitted:
- `_particle_energy_numba`, `_total_energy_numba`
- `_sweep_numba`, `_run_sweeps_numba`
- `_energies_from_samples_numba` (post-processing)

Call graph (hot path):
`run()` → `_run_block()` → `_run_sweeps_numba()` → `_sweep_numba()` → `_particle_energy_numba()`

## Logging / outputs
- Progress logs: `ProgressLogger` writes line-buffered entries (timestamp, seed, sweep, acceptance mean) to a `__worker.log` file.
- Samples & acceptance: `ChunkedStorage` uses `np.memmap` to write large arrays incrementally under `<save_path_root>__chunks/`.
- Manifest JSON: `<save_path_root>__manifest.json` collects metadata and file paths; a small pointer JSON is also written at `save_samples_path`.
- Output frequency is controlled by:
  - `sweeps` (if `n_samples == 0`)
  - `burn_in_sweeps`, `sample_interval`, and `n_samples` (if sampling)

## Known bottlenecks and planned optimizations
1. **Local dE computed twice per move** → optimized by a one-pass local energy difference calculation (implemented in `Ka2d_optimzied_energy.py`).
2. **Metropolis acceptance form** → benchmark exp-based vs log-based rules under Numba to decide whether the log form is beneficial (`benchmark_acceptance_rule.py`).
