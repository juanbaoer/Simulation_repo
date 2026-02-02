"""
Random (non-deterministic) wrapper for `Ka2d_real_trajactory.py`.

This module intentionally does NOT use a fixed seed:
- Python-side RNGs (type shuffling / initialization) use OS entropy.
- Numba-side `np.random.*` inside `@njit` uses Numba's RNG state without seeding.

If you want reproducibility, use `Ka2d_real_trajactory.py` instead.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import Ka2d_real_trajactory as ka  # noqa: E402


def run_ka2d_mc(
    N: int = 176,
    T: float = 0.5,
    sweeps: int = 2000,
    rho: float = 1.19,
    seed: Optional[int] = None,  # accepted for API-compat; intentionally ignored
    ternary: bool = True,
    composition_ratio: Tuple[int, ...] = (5, 3, 3),
    n_samples: int = 0,
    burn_in_sweeps: int = 5,
    sample_interval: int = 1,
    save_samples_path: Optional[str] = None,
) -> dict:
    _ = seed  # explicitly ignored
    params = ka.default_ternary_params() if ternary else ka.example_binary_params()
    cfg = ka.MCConfig(
        N=N,
        T=T,
        sweeps=sweeps,
        rho=rho,
        seed=None,  # force non-deterministic behavior
        composition_ratio=composition_ratio,
        n_samples=n_samples,
        burn_in_sweeps=burn_in_sweeps,
        sample_interval=sample_interval,
        save_samples_path=save_samples_path,
    )
    sim = ka.KA2DMCSimulatorNumba(params, cfg)
    return sim.run()


if __name__ == "__main__":
    # Example usage: same API as `Ka2d_real_trajactory.py`, but fully random.
    # Run 10 runs of N=43, rho=43/36, T=0.5 KA2D MC simulation, save each of the configurations in /Users/xuhengyuan/Downloads/Notes/Simulation_repo/MCMC/ternary_glass_forming_liquid/Data/Random
    import pathlib
    output_dir = pathlib.Path(
        "/Users/xuhengyuan/Downloads/Notes/Simulation_repo/MCMC/ternary_glass_forming_liquid/Data/Random"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for run_id in range(7, 17):
        save_path = output_dir / f"N43_rho_1p19_T_0p5_random_1M_{run_id}.npy"
        result = run_ka2d_mc(
            N=43,
            T=0.5,
            sweeps=0,
            rho=43/36,
            ternary=True,
            composition_ratio=(20, 11, 12),
            n_samples=1000000,
            burn_in_sweeps=5000000,
            sample_interval=30,
            save_samples_path=str(save_path),
        )
        print(f"Completed run {run_id}, results saved to {save_path}")
