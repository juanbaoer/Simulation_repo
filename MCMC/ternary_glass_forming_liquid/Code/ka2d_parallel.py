"""
Parallel runner for independent KA2D MC chains.

Notes for macOS: multiprocessing uses spawn; keep pool creation inside a
`if __name__ == "__main__":` guard when invoking this module as a script.
Deterministic runs expect NUMBA_NUM_THREADS=1 (set per worker).
"""
from __future__ import annotations

import copy
import os
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# The single-chain API lives in the sibling modules.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    import Ka2d_random_seed_fast as fast_mod  # type: ignore
except Exception:  # pragma: no cover - fallback
    fast_mod = None  # type: ignore
import Ka2d_random_seed as slow_mod  # type: ignore


def _load_sim_module(prefer_fast: bool = True):
    if prefer_fast and fast_mod is not None:
        return fast_mod
    return slow_mod


def _strip_traces(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove large per-sweep traces to reduce IPC overhead.
    """
    out = dict(result)
    for key in ("acceptance", "energies", "first_window_real_positions"):
        if key in out:
            out[key] = None
    return out


def _aggregate(chains: List[Dict[str, Any]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    n = len(chains)
    if n == 0:
        return agg

    # Running sums to avoid concatenating huge arrays.
    e_sum = 0.0
    e_count = 0
    acc_sum = 0.0
    acc_count = 0
    max_disp_sum = 0.0
    for res in chains:
        if res.get("samples_energies") is not None:
            arr = np.asarray(res["samples_energies"])
            e_sum += float(arr.sum())
            e_count += arr.size
        if res.get("acceptance") is not None:
            arr = np.asarray(res["acceptance"])
            acc_sum += float(arr.sum())
            acc_count += arr.size
        if "final_max_disp" in res:
            max_disp_sum += float(res["final_max_disp"])

    if e_count > 0:
        agg["mean_samples_energy"] = e_sum / e_count
    if acc_count > 0:
        agg["mean_acceptance"] = acc_sum / acc_count
    agg["final_max_disp_mean"] = max_disp_sum / n
    return agg


def _worker_run(
    cfg,
    seed: int,
    return_traces: bool,
    warmup: bool,
    prefer_fast: bool,
) -> tuple:
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        import numba  # local import in worker
        numba.set_num_threads(1)
    except Exception:
        pass

    mod = _load_sim_module(prefer_fast)
    params = mod.default_ternary_params()
    # Clone cfg and inject seed + disable saving to avoid contention.
    cfg_local = copy.deepcopy(cfg)
    cfg_local.seed = seed
    cfg_local.save_samples_path = None

    if warmup:
        warm_cfg = replace(
            cfg_local,
            sweeps=1,
            n_samples=0,
            burn_in_sweeps=0,
            sample_interval=1,
            save_samples_path=None,
        )
        warm_sim = mod.KA2DMCSimulatorNumba(params, warm_cfg)
        warm_sim.run()

    sim = mod.KA2DMCSimulatorNumba(params, cfg_local)
    t0 = time.perf_counter()
    result = sim.run()
    elapsed = time.perf_counter() - t0
    if not return_traces:
        result = _strip_traces(result)
    return result, elapsed


def run_ka2d_mc_parallel(
    cfg,
    n_chains: int,
    base_seed: int,
    n_workers: Optional[int] = None,
    method: str = "spawn",
    chunk: int = 1,
    *,
    return_traces: bool = True,
    aggregate: bool = True,
    warmup: bool = True,
    prefer_fast: bool = True,
) -> Dict[str, Any]:
    """
    Run multiple independent KA2D MC chains in parallel (spawn-safe).

    Warning: when n_chains > 1, save_samples_path is disabled per chain to
    avoid clobbering; pass unique paths per chain manually if you need files.
    """
    if n_chains <= 0:
        raise ValueError("n_chains must be positive")
    if cfg.save_samples_path and n_chains > 1:
        raise ValueError("Parallel runs require unique save_samples_path per chain")

    ss = np.random.SeedSequence(base_seed)
    seeds = [
        int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in ss.spawn(n_chains)
    ]

    ctx = __import__("multiprocessing").get_context(method)
    worker_args = [
        (cfg, seed, return_traces, warmup, prefer_fast) for seed in seeds
    ]

    chains: List[Dict[str, Any]] = []
    per_chain_time: List[float] = []
    t_wall_start = time.perf_counter()
    with ctx.Pool(processes=n_workers) as pool:
        for res, elapsed in pool.starmap(_worker_run, worker_args, chunksize=chunk):
            chains.append(res)
            per_chain_time.append(elapsed)
    wall_time = time.perf_counter() - t_wall_start
    # starmap preserves input order
    ordered = chains
    ordered_time = per_chain_time

    out: Dict[str, Any] = {
        "chains": ordered,
        "seeds": seeds,
        "wall_time_s": wall_time,
        "per_chain_time_s": ordered_time,
    }
    if aggregate:
        out["aggregate"] = _aggregate(ordered)
    return out


__all__ = ["run_ka2d_mc_parallel"]
