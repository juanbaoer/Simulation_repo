import os
import sys
import time
import pathlib

os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import numba

# Determinism: force single-thread numba.
numba.set_num_threads(1)

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Ka2d_random_seed as slow  # noqa: E402
import Ka2d_random_seed_fast as fast  # noqa: E402


def _sim_class(mod):
    return getattr(mod, "KA2DMCSimulatorNumbaFast", mod.KA2DMCSimulatorNumba)


def run_sim(mod, cfg_kwargs):
    params = mod.default_ternary_params()
    cfg = mod.MCConfig(**cfg_kwargs)
    sim = _sim_class(mod)(params, cfg)
    return sim.run()


def test_fast_matches_reference():
    cfg = dict(
        N=43,
        T=0.32,
        rho=43 / 36,
        seed=0,
        burn_in_sweeps=200,
        sample_interval=5,
        n_samples=20,
        sweeps=0,
        composition_ratio=(20, 11, 12),
    )
    ref = run_sim(slow, cfg)
    fast_res = run_sim(fast, cfg)

    assert set(ref.keys()) == set(fast_res.keys())
    for key in ref:
        r = ref[key]
        f = fast_res[key]
        if r is None or f is None:
            assert r is None and f is None
            continue
        np.testing.assert_array_equal(np.asarray(f), np.asarray(r))


def test_speed_benchmark():
    warm_cfg = dict(
        N=43,
        T=0.32,
        rho=43 / 36,
        seed=1,
        burn_in_sweeps=20,
        sample_interval=2,
        n_samples=3,
        sweeps=0,
        composition_ratio=(20, 11, 12),
    )
    run_sim(slow, warm_cfg)
    run_sim(fast, warm_cfg)

    bench_cfg = dict(
        N=43,
        T=0.32,
        rho=43 / 36,
        seed=2,
        burn_in_sweeps=2000,
        sample_interval=1000,
        n_samples=2000,
        sweeps=0,
        composition_ratio=(20, 11, 12),
    )
    t0 = time.perf_counter()
    run_sim(slow, bench_cfg)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_sim(fast, bench_cfg)
    t_fast = time.perf_counter() - t0

    speedup = t_ref / t_fast if t_fast > 0 else float("inf")
    print(f"reference: {t_ref:.4f}s, fast: {t_fast:.4f}s, speedup: {speedup:.2f}x")

if __name__ == "__main__":
    test_fast_matches_reference()
    test_speed_benchmark()