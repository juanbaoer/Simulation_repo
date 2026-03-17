from __future__ import annotations

import time
import numpy as np

try:
    from numba import njit
except Exception as e:
    raise RuntimeError("This benchmark requires numba.") from e


def generate_dE(n: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dE = rng.normal(loc=0.2, scale=1.0, size=n)
    # Add a light tail to mix magnitudes and signs
    tail = rng.exponential(scale=0.5, size=n) * rng.choice([-1.0, 1.0], size=n)
    dE = dE + 0.5 * tail
    return dE.astype(np.float64)


@njit
def _seed_numba(seed_int: int) -> None:
    np.random.seed(seed_int)


@njit(fastmath=True)
def accept_exp_numba(dE: np.ndarray, beta: float) -> int:
    acc = 0
    for i in range(dE.size):
        if np.random.random() < np.exp(-beta * dE[i]):
            acc += 1
    return acc


@njit(fastmath=True)
def accept_log_numba(dE: np.ndarray, beta: float) -> int:
    acc = 0
    for i in range(dE.size):
        u = np.random.random()
        if dE[i] <= 0.0 or np.log(u) < -beta * dE[i]:
            acc += 1
    return acc


def acceptance_numpy_exp(dE: np.ndarray, beta: float, u: np.ndarray) -> int:
    return int(np.sum(u < np.exp(-beta * dE)))


def acceptance_numpy_log(dE: np.ndarray, beta: float, u: np.ndarray) -> int:
    return int(np.sum((dE <= 0.0) | (np.log(u) < -beta * dE)))


def _stats(times: list[float]) -> tuple[float, float]:
    avg = float(np.mean(times))
    std = float(np.std(times, ddof=0))
    return avg, std


def main() -> None:
    n = 1_000_000
    beta = 2.0
    seed = 777

    dE = generate_dE(n, seed=seed)

    # --- Correctness check (NumPy, shared RNG stream) ---
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    acc_exp_np = acceptance_numpy_exp(dE, beta, u)
    acc_log_np = acceptance_numpy_log(dE, beta, u)
    print("NumPy acceptance:")
    print(f"  exp  count={acc_exp_np}  rate={acc_exp_np / n:.6f}")
    print(f"  log  count={acc_log_np}  rate={acc_log_np / n:.6f}")
    print(f"  diff count={acc_log_np - acc_exp_np}")

    # --- Correctness check (Numba, seeded RNG) ---
    _seed_numba(seed)
    acc_exp_nb = accept_exp_numba(dE, beta)
    _seed_numba(seed)
    acc_log_nb = accept_log_numba(dE, beta)
    print("Numba acceptance (seeded):")
    print(f"  exp  count={acc_exp_nb}  rate={acc_exp_nb / n:.6f}")
    print(f"  log  count={acc_log_nb}  rate={acc_log_nb / n:.6f}")
    print(f"  diff count={acc_log_nb - acc_exp_nb}")

    # --- Warm-up (exclude compilation) ---
    _seed_numba(seed)
    accept_exp_numba(dE[:1000], beta)
    _seed_numba(seed)
    accept_log_numba(dE[:1000], beta)

    # --- Performance benchmark ---
    repeats = 7
    times_exp = []
    times_log = []

    for r in range(repeats):
        _seed_numba(seed + r)
        t0 = time.perf_counter()
        accept_exp_numba(dE, beta)
        t1 = time.perf_counter()
        times_exp.append(t1 - t0)

    for r in range(repeats):
        _seed_numba(seed + r)
        t0 = time.perf_counter()
        accept_log_numba(dE, beta)
        t1 = time.perf_counter()
        times_log.append(t1 - t0)

    avg_exp, std_exp = _stats(times_exp)
    avg_log, std_log = _stats(times_log)
    speedup = avg_exp / avg_log if avg_log > 0 else float('inf')

    print("Timing (Numba, steady-state):")
    print(f"  exp  avg={avg_exp:.6f}s  std={std_exp:.6f}s")
    print(f"  log  avg={avg_log:.6f}s  std={std_log:.6f}s")
    print(f"  speedup (exp/log)={speedup:.3f}x")

    # Decision rule: log must be faster and separated beyond variability
    stable = (avg_log + std_log) < (avg_exp - std_exp)
    if stable and avg_log < avg_exp:
        decision = "log acceptance appears faster under Numba"
    elif avg_log < avg_exp:
        decision = "log acceptance slightly faster, but not clearly separated from noise"
    else:
        decision = "exp acceptance is faster (or equal) under Numba"

    print("Decision:")
    print(f"  {decision}")


if __name__ == "__main__":
    main()
