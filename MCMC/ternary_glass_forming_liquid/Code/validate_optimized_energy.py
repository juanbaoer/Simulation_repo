from __future__ import annotations

import time
import tempfile
import importlib.util
import sys
from pathlib import Path
import numpy as np


def load_module_no_cache(name: str, path: Path):
    # Load module source and force Numba cache=False to avoid locator errors.
    src = path.read_text(encoding="utf-8")
    src = src.replace("cache=True", "cache=False")
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


CODE_DIR = Path(__file__).resolve().parent
orig = load_module_no_cache("ka2d_orig_no_cache", CODE_DIR / "Ka2d_random_seed_optimized_logging_append.py")
opt = load_module_no_cache("ka2d_opt_no_cache", CODE_DIR / "Ka2d_optimzied_energy.py")


def rel_diff(a: float, b: float) -> float:
    denom = max(1.0, abs(a), abs(b))
    return abs(a - b) / denom


def run_with_samples(mod, out_dir: Path, seed: int) -> dict:
    save_path = out_dir / f"{mod.__name__}_data.npz"
    return mod.run_ka2d_mc(
        N=44,
        T=0.5,
        sweeps=0,
        rho=1.194,
        seed=seed,
        ternary=True,
        composition_ratio=(5, 3, 3),
        n_samples=6,
        burn_in_sweeps=50,
        sample_interval=10,
        rcut_factor=2.5,
        max_disp=0.095,
        target_accept=0.35,
        save_samples_path=str(save_path),
    )


def extract_metrics(mod, out: dict) -> dict:
    manifest = out.get("manifest")
    full = mod.load_full_output(manifest, mmap_mode="r")

    acceptance = full.get("acceptance")
    acc_mean = float(np.mean(acceptance)) if acceptance is not None else None

    positions = full.get("positions")
    energies = mod._energies_from_samples_numba(
        positions,
        full.get("types"),
        full.get("box"),
        full.get("eps"),
        full.get("sig"),
        float(full.get("rcut_factor")),
        float(full.get("C0")),
        float(full.get("C2")),
        float(full.get("C4")),
    )
    e_mean = float(np.mean(energies))
    e_var = float(np.var(energies))

    initial = full.get("initial_positions")
    final_real = full.get("final_real_positions")
    msd = float(np.mean(np.sum((final_real - initial) ** 2, axis=1)))

    return {
        "acc_mean": acc_mean,
        "e_mean": e_mean,
        "e_var": e_var,
        "msd": msd,
    }


def time_run(mod, repeats: int, seed: int) -> list[float]:
    times: list[float] = []
    for r in range(repeats):
        t0 = time.perf_counter()
        mod.run_ka2d_mc(
            N=44,
            T=0.5,
            sweeps=300,
            rho=1.194,
            seed=seed + r,
            ternary=True,
            composition_ratio=(5, 3, 3),
            n_samples=0,
            burn_in_sweeps=0,
            sample_interval=1,
            rcut_factor=2.5,
            max_disp=0.095,
            target_accept=0.35,
            save_samples_path=None,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def stats(times: list[float]) -> tuple[float, float]:
    avg = float(np.mean(times))
    std = float(np.std(times, ddof=0))
    return avg, std


def main() -> None:
    seed = 1234

    # --- Consistency check ---
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        out_orig = run_with_samples(orig, out_dir, seed)
        out_opt = run_with_samples(opt, out_dir, seed)

        m_orig = extract_metrics(orig, out_orig)
        m_opt = extract_metrics(opt, out_opt)

    print("Consistency metrics (orig vs optimized):")
    for key in ("acc_mean", "e_mean", "e_var", "msd"):
        a = m_orig[key]
        b = m_opt[key]
        rd = rel_diff(a, b)
        print(f"  {key}: orig={a:.6f}  opt={b:.6f}  rel_diff={rd:.3e}")

    tol = {
        "acc_mean": 1e-10,
        "e_mean": 1e-8,
        "e_var": 1e-8,
        "msd": 1e-8,
    }
    ok = True
    for key, t in tol.items():
        if rel_diff(m_orig[key], m_opt[key]) > t:
            ok = False
    print("Consistency:")
    print("  PASS" if ok else "  FAIL (outside tolerance)")

    # --- Performance check ---
    # Warm-up (compile Numba)
    orig.run_ka2d_mc(N=44, T=0.5, sweeps=10, rho=1.194, seed=seed, ternary=True,
                    composition_ratio=(5, 3, 3), n_samples=0, burn_in_sweeps=0,
                    sample_interval=1, rcut_factor=2.5, max_disp=0.095,
                    target_accept=0.35, save_samples_path=None)
    opt.run_ka2d_mc(N=44, T=0.5, sweeps=10, rho=1.194, seed=seed, ternary=True,
                   composition_ratio=(5, 3, 3), n_samples=0, burn_in_sweeps=0,
                   sample_interval=1, rcut_factor=2.5, max_disp=0.095,
                   target_accept=0.35, save_samples_path=None)

    repeats = 5
    times_orig = time_run(orig, repeats, seed)
    times_opt = time_run(opt, repeats, seed)
    avg_o, std_o = stats(times_orig)
    avg_p, std_p = stats(times_opt)
    speedup = avg_o / avg_p if avg_p > 0 else float('inf')

    print("Timing (steady-state, n_samples=0):")
    print(f"  orig avg={avg_o:.6f}s  std={std_o:.6f}s")
    print(f"  opt  avg={avg_p:.6f}s  std={std_p:.6f}s")
    print(f"  speedup (orig/opt)={speedup:.3f}x")


if __name__ == "__main__":
    main()
