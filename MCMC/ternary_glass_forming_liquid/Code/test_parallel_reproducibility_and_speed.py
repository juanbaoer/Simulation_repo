import os
import sys
import time
import copy
from pathlib import Path

import numpy as np

os.environ["NUMBA_NUM_THREADS"] = "1"
import numba  # noqa: E402

numba.set_num_threads(1)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ka2d_parallel as parallel  # noqa: E402

try:
    import Ka2d_random_seed_fast as fast_mod  # noqa: E402
    _SIM_MOD = fast_mod
except Exception:  # pragma: no cover
    import Ka2d_random_seed as slow_mod  # noqa: E402
    _SIM_MOD = slow_mod


def _derive_seeds(base_seed: int, n: int):
    ss = np.random.SeedSequence(base_seed)
    return [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in ss.spawn(n)]


def _run_single_chain(seed: int, cfg):
    params = _SIM_MOD.default_ternary_params()
    cfg_local = copy.deepcopy(cfg)
    cfg_local.seed = seed
    sim = _SIM_MOD.KA2DMCSimulatorNumba(params, cfg_local)
    return sim.run()


def _assert_equal_dicts(a, b):
    assert set(a.keys()) == set(b.keys())
    for key in a:
        av = a[key]
        bv = b[key]
        if av is None or bv is None:
            assert av is None and bv is None
            continue
        np.testing.assert_array_equal(np.asarray(av), np.asarray(bv))


def test_parallel_matches_sequential():
    cfg = _SIM_MOD.MCConfig(
        N=43,
        T=0.32,
        rho=43 / 36,
        seed=0,
        burn_in_sweeps=50,
        sample_interval=5,
        n_samples=6,
        sweeps=0,
        composition_ratio=(20, 11, 12),
    )
    n_chains = 4
    base_seed = 123
    par_res = parallel.run_ka2d_mc_parallel(
        cfg,
        n_chains=n_chains,
        base_seed=base_seed,
        n_workers=2,
        method="spawn",
        chunk=1,
        return_traces=True,
        aggregate=False,
        warmup=False,
        prefer_fast=True,
    )
    seeds = par_res["seeds"]
    seq_results = [_run_single_chain(s, cfg) for s in seeds]
    for p, s in zip(par_res["chains"], seq_results):
        _assert_equal_dicts(p, s)


def test_parallel_speed_compared_to_sequential():
    cfg = _SIM_MOD.MCConfig(
        N=43,
        T=0.32,
        rho=43 / 36,
        seed=0,
        burn_in_sweeps=200,
        sample_interval=5,
        n_samples=50,
        sweeps=0,
        composition_ratio=(20, 11, 12),
    )
    n_chains = 4
    base_seed = 999
    seeds = _derive_seeds(base_seed, n_chains)

    t0 = time.perf_counter()
    seq_out = [_run_single_chain(s, cfg) for s in seeds]
    seq_time = time.perf_counter() - t0
    assert len(seq_out) == n_chains

    t0 = time.perf_counter()
    par_out = parallel.run_ka2d_mc_parallel(
        cfg,
        n_chains=n_chains,
        base_seed=base_seed,
        n_workers=n_chains,
        method="spawn",
        chunk=1,
        return_traces=False,
        aggregate=True,
        warmup=True,
        prefer_fast=True,
    )
    par_time = time.perf_counter() - t0
    speedup = seq_time / par_time if par_time > 0 else float("inf")
    # Do not assert on speed; just ensure outputs exist and print for visibility.
    assert len(par_out["chains"]) == n_chains
    print(f"sequential {seq_time:.3f}s vs parallel {par_time:.3f}s, speedup {speedup:.2f}x")

if __name__ == "__main__":
    test_parallel_matches_sequential()
    test_parallel_speed_compared_to_sequential()