from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Tuple, Optional, List, Dict
import numpy as np

try:
    from numba import njit, prange
except Exception as e:
    raise RuntimeError(
        "This module requires numba. Install with `pip install numba`."
    ) from e

from tqdm import trange
Array = np.ndarray

# Fixed seed used only for the randomized type tensor generation
TYPES_TENSOR_SEED = 314159

# ========================== Parameters ==========================

@dataclass
class KA2DParams:
    eps: Array  # shape (T, T)
    sig: Array  # shape (T, T)
    C0: float = 0.04049023795
    C2: float = -0.00970155098
    C4: float = 0.00062012616
    rcut_factor: float = 2.5

    @property
    def n_types(self) -> int:
        return self.eps.shape[0]

    def rcut(self) -> Array:
        return self.rcut_factor * self.sig


def default_ternary_params() -> KA2DParams:
    eps = np.array([[1.00, 1.50, 0.75],
                    [1.50, 0.50, 1.50],
                    [0.75, 1.50, 0.75]], dtype=float)
    sig = np.array([[1.00, 0.80, 0.90],
                    [0.80, 0.88, 0.80],
                    [0.90, 0.80, 0.94]], dtype=float)
    # eps = np.array([[1.00, 1.00, 1.00],
    #                 [1.00, 1.00, 1.00],
    #                 [1.00, 1.00, 1.00]], dtype=float)
    # sig = np.array([[1.00, 1.00, 1.00],
    #                 [1.00, 1.00, 1.00],
    #                 [1.00, 1.00, 1.00]], dtype=float)
    return KA2DParams(eps=eps, sig=sig)


def example_binary_params() -> KA2DParams:
    eps = np.array([[1.0, 1.5],
                    [1.5, 0.5]], dtype=float)
    sig = np.array([[1.0, 0.8],
                    [0.8, 0.88]], dtype=float)
    return KA2DParams(eps=eps, sig=sig)

# ========================== Config ==========================

@dataclass
class MCConfig:
    N: int = 44
    T: float = 0.5
    rho: float = 1.194
    max_disp: float = 0.085
    sweeps: int = 2000
    seed: Optional[int] = 0
    autotune: bool = True
    target_accept: float = 0.35
    tune_interval_sweeps: int = 25
    composition_ratio: Tuple[int, ...] = (5, 3, 3)
    composition_counts: Optional[Tuple[int, ...]] = None

    n_samples: int = 0
    burn_in_sweeps: int = 5000
    sample_interval: int = 10
    save_samples_path: Optional[str] = None


@njit(cache=True, fastmath=True)
def _pbc_delta(dr: Array, Lx: float, Ly: float) -> None:
    """
    In-place minimum image for dr[:,0], dr[:,1].
    """
    n = dr.shape[0]
    for i in range(n):
        # x
        dx = dr[i, 0]
        dx -= Lx * np.rint(dx / Lx)
        dr[i, 0] = dx
        # y
        dy = dr[i, 1]
        dy -= Ly * np.rint(dy / Ly)
        dr[i, 1] = dy

@njit(cache=True, fastmath=True)
def _particle_energy_numba(i: int,
                           pos: Array,
                           types: Array,
                           eps: Array,
                           sig: Array,
                           C0: float, C2: float, C4: float,
                           rcut2: Array,
                           Lx: float, Ly: float) -> float:
    """
    Energy of particle i with all others (Kob–Andersen LJ + smooth tail),
    working in r^2 and applying cutoff in s^2 = r^2 / sigma^2.
    """
    ri0 = pos[i, 0]
    ri1 = pos[i, 1]
    ti = types[i]
    N = pos.shape[0]
    E = 0.0
    for j in range(N):
        if j == i:
            continue
        # minimum image
        dx = pos[j, 0] - ri0
        dx -= Lx * np.rint(dx / Lx)
        dy = pos[j, 1] - ri1
        dy -= Ly * np.rint(dy / Ly)
        r2 = dx * dx + dy * dy
        # params
        tj = types[j]
        if r2 <= 0.0 or r2 >= rcut2[ti, tj]:
            continue
        sig_ab = sig[ti, tj]
        s2 = r2 / (sig_ab * sig_ab)  # s^2
        inv2 = 1.0 / s2       # = sigma^2 / r^2
        inv6 = inv2 * inv2 * inv2
        inv12 = inv6 * inv6
        g = (inv12 - inv6) + C0 + C2 * s2 + C4 * (s2 * s2)
        E += 4.0 * eps[ti, tj] * g
    return E

@njit(cache=True, fastmath=True, parallel=True)
def _total_energy_numba(pos: Array, types: Array,
                        eps: Array, sig: Array,
                        C0: float, C2: float, C4: float,
                        rcut2: Array,
                        Lx: float, Ly: float) -> float:
    """
    0.5 * sum_i E_i to avoid double counting. Parallel over i.
    """
    N = pos.shape[0]
    energy = 0.0
    for i in prange(N):
        energy += _particle_energy_numba(i, pos, types, eps, sig, C0, C2, C4,
                                      rcut2, Lx, Ly)
    return 0.5 * energy

@njit(cache=True, fastmath=True)
def _sweep_numba(pos: Array, pos_real: Array, types: Array,
                 eps: Array, sig: Array,
                 C0: float, C2: float, C4: float,
                 rcut2: Array,
                 Lx: float, Ly: float,
                 beta: float, max_disp: float,
                 energy: float) -> tuple:
    """
    One MC sweep (N attempted single-particle moves). Returns:
    (accepts, total_trials, new_energy).
    """

    N = pos.shape[0]
    accepts = 0
    for _ in range(N):
        i = np.random.randint(0, N)
        oldx = pos[i, 0]
        oldy = pos[i, 1]

        old_Ei = _particle_energy_numba(i, pos, types, eps, sig, C0, C2, C4,
                                        rcut2, Lx, Ly)

        # raw trial displacement (unwrapped)
        dx_trial = (np.random.random() - 0.5) * 2.0 * max_disp
        dy_trial = (np.random.random() - 0.5) * 2.0 * max_disp
        tx_raw = oldx + dx_trial
        ty_raw = oldy + dy_trial
        # wrapped coordinates
        tx = tx_raw % Lx
        ty = ty_raw % Ly

        pos[i, 0] = tx
        pos[i, 1] = ty
        # update real (unwrapped) coords tentatively
        pos_real[i, 0] += dx_trial
        pos_real[i, 1] += dy_trial

        new_Ei = _particle_energy_numba(i, pos, types, eps, sig, C0, C2, C4,
                                        rcut2, Lx, Ly)
        dE = new_Ei - old_Ei

        if  np.random.random() < np.exp(-beta * dE):
            energy += dE
            accepts += 1
        else:
            # revert wrapped
            pos[i, 0] = oldx
            pos[i, 1] = oldy
            # revert real
            pos_real[i, 0] -= dx_trial
            pos_real[i, 1] -= dy_trial
    return accepts, N, energy

@njit(cache=True, fastmath=True)
def _run_sweeps_numba(pos: Array, pos_real: Array, types: Array,
                      eps: Array, sig: Array,
                      C0: float, C2: float, C4: float,
                      rcut2: Array,
                      Lx: float, Ly: float,
                      beta: float,
                      max_disp0: float,
                      sweeps: int,
                      autotune: bool, target_accept: float, tune_interval: int,
                      energy0: float) -> tuple:
    """
    Run `sweeps` sweeps with optional autotuning.
    Returns (energies[...], accepts[...], final_max_disp, final_energy)
    """
    energies = np.empty(sweeps, dtype=np.float64)
    accepts_hist = np.empty(sweeps, dtype=np.float64)
    max_disp = max_disp0
    energy = energy0

    for s in range(sweeps):
        acc, tot, energy = _sweep_numba(pos, pos_real, types, eps, sig,
                                        C0, C2, C4, rcut2,
                                        Lx, Ly, beta, max_disp, energy)
        energies[s] = energy
        accepts_hist[s] = acc / tot

        if autotune and ((s + 1) % tune_interval == 0):
            # mean over recent window
            start = max(0, s + 1 - tune_interval)
            recent = 0.0
            for k in range(start, s + 1):
                recent += accepts_hist[k]
            recent /= (s + 1 - start)
            if recent < target_accept:
                max_disp *= 0.9
            else:
                max_disp *= 1.1
            if max_disp < 0.01:
                max_disp = 0.01
            elif max_disp > 0.8:
                max_disp = 0.8

    return energies, accepts_hist, max_disp, energy


@njit(cache=True)
def _seed_numba(seed_int: int) -> None:
    """
    Seed Numba's legacy RNG used by np.random.* inside njit regions.
    """
    np.random.seed(seed_int)


# ========================== Energy from samples (Numba) ==========================
@njit(cache=True, fastmath=True, parallel=True)
def _energies_from_samples_numba(
    samples_pos: Array,   # (S, N, 2)
    types: Array,         # (N,)
    box: Array,           # (2,) or (S,2)
    eps: Array,           # (T, T)
    sig: Array,           # (T, T)
    rcut_factor: float,
    C0: float, C2: float, C4: float,
) -> Array:
    """
    Total energies for each saved sample, shape (S,), using the same KA form as the simulator.
    """
    S = samples_pos.shape[0]
    N = samples_pos.shape[1]
    out = np.empty(S, dtype=np.float64)

    if box.ndim == 1:
        Lx = float(box[0])
        Ly = float(box[1])
        for s in prange(S):
            E = 0.0
            for i in range(N - 1):
                xi0 = samples_pos[s, i, 0]
                xi1 = samples_pos[s, i, 1]
                ti = types[i]
                for j in range(i + 1, N):
                    dx = samples_pos[s, j, 0] - xi0
                    dx -= Lx * np.rint(dx / Lx)
                    dy = samples_pos[s, j, 1] - xi1
                    dy -= Ly * np.rint(dy / Ly)
                    r2 = dx * dx + dy * dy
                    if r2 <= 0.0:
                        continue
                    tj = types[j]
                    sig_ab = sig[ti, tj]
                    rc = rcut_factor * sig_ab
                    if r2 >= rc * rc:
                        continue
                    s2 = r2 / (sig_ab * sig_ab)
                    inv2 = 1.0 / s2
                    inv6 = inv2 * inv2 * inv2
                    inv12 = inv6 * inv6
                    g = (inv12 - inv6) + C0 + C2 * s2 + C4 * (s2 * s2)
                    E += 4.0 * eps[ti, tj] * g
            out[s] = E
        return out

    for s in prange(S):
        Lx = float(box[s, 0])
        Ly = float(box[s, 1])
        E = 0.0
        for i in range(N - 1):
            xi0 = samples_pos[s, i, 0]
            xi1 = samples_pos[s, i, 1]
            ti = types[i]
            for j in range(i + 1, N):
                dx = samples_pos[s, j, 0] - xi0
                dx -= Lx * np.rint(dx / Lx)
                dy = samples_pos[s, j, 1] - xi1
                dy -= Ly * np.rint(dy / Ly)
                r2 = dx * dx + dy * dy
                if r2 <= 0.0:
                    continue
                tj = types[j]
                sig_ab = sig[ti, tj]
                rc = rcut_factor * sig_ab
                if r2 >= rc * rc:
                    continue
                s2 = r2 / (sig_ab * sig_ab)
                inv2 = 1.0 / s2
                inv6 = inv2 * inv2 * inv2
                inv12 = inv6 * inv6
                g = (inv12 - inv6) + C0 + C2 * s2 + C4 * (s2 * s2)
                E += 4.0 * eps[ti, tj] * g
        out[s] = E
    return out


# ========================== Fast, Numba MC ==========================
class KA2DMCSimulatorNumba:
    """
    Numba-accelerated KA 2D MC (no neighbor list).
    Public API mirrors the vectorized version for easy swapping.

    Determinism: a single base seed drives NumPy (PCG64) and Numba's
    legacy np.random.* RNG on CPU single-thread runs. Parallel execution
    can reorder RNG calls and is not guaranteed deterministic.
    """

    def __init__(self, params: KA2DParams, cfg: MCConfig):
        self.p = params
        self.cfg = cfg

        # Unified seeding: derive substreams for NumPy + Numba from one base seed.
        self.base_seed = cfg.seed
        if self.base_seed is not None:
            ss = np.random.SeedSequence(self.base_seed)
            child_pos, child_numba = ss.spawn(2)
            self._rng_pos = np.random.default_rng(child_pos)
            self._seed_numba_int = int(child_numba.generate_state(1, dtype=np.uint32)[0])
        else:
            self._rng_pos = np.random.default_rng()
            self._seed_numba_int = None
        # Types RNG uses a fixed seed so types_tensor shuffling is constant across runs
        self._rng_types = np.random.default_rng(TYPES_TENSOR_SEED)

        self._numba_rng_seeded = False

        # Decide counts per type
        if cfg.composition_counts is None:
            ratio = np.array(cfg.composition_ratio, dtype=np.float64)
            frac = ratio / ratio.sum()
            counts = np.floor(frac * cfg.N).astype(np.int32)
            remainder = cfg.N - counts.sum()
            counts[np.argmax(frac)] += remainder
            self.counts = counts
        else:
            counts = np.array(cfg.composition_counts, dtype=np.int32)
            if len(counts) != self.p.n_types or counts.sum() != cfg.N:
                raise ValueError("composition_counts must match n_types and sum to N")
            self.counts = counts

        # Box
        L = float(np.sqrt(cfg.N / cfg.rho))
        self.box = np.array([L, L], dtype=np.float64)

        # Types array (0..T-1), shuffled
        types = np.concatenate([
            np.full(c, t, dtype=np.int32) for t, c in enumerate(self.counts)
        ])
        self._rng_types.shuffle(types)
        self.types = types

        # Initial positions: jittered lattice (pure NumPy)
        # self.pos = self._init_lattice_positions(cfg.N, self.box, rng).astype(np.float64)
        D = self.box.shape[0]  
        self.pos = self._rng_pos.random((cfg.N, D)) * self.box  # each coord in [0, L)
        # Real (unwrapped) coordinates mirror initial wrapped positions
        self.pos_real = self.pos.copy()
        # Preserve a copy of the initial positions for reproducibility checks
        self.initial_positions = self.pos.copy()

        # Precompute constants
        self.C0, self.C2, self.C4 = float(self.p.C0), float(self.p.C2), float(self.p.C4)
        self.rcut2 = (self.p.rcut() ** 2).astype(np.float64)  # per-pair cutoff^2

        # Seed Numba RNG (done inside run)
        # Initial energy
        self.energy = _total_energy_numba(self.pos, self.types,
                                          self.p.eps, self.p.sig,
                                          self.C0, self.C2, self.C4,
                                          self.rcut2,
                                          self.box[0], self.box[1])

    @staticmethod
    def _init_lattice_positions(N: int, box: Array, rng) -> Array:
        Lx, Ly = float(box[0]), float(box[1])
        nx = int(np.sqrt(N))
        ny = (N + nx - 1) // nx
        xs = (np.arange(nx) + 0.5) * (Lx / nx)
        ys = (np.arange(ny) + 0.5) * (Ly / ny)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)[:N]
        jitter = 0.05 * min(Lx / nx, Ly / ny)
        pts += (rng.random((N, 2)) - 0.5) * 2 * jitter
        # wrap just in case
        pts[:, 0] %= Lx
        pts[:, 1] %= Ly
        return pts

    def _run_block(self, nsweeps: int, beta: float, max_disp: float,
               energies: List[float], acc_hist: List[float],
               record_sweeps: bool = True,
               autotune: Optional[bool] = None) -> float:
        use_autotune = self.cfg.autotune if autotune is None else autotune
        e_arr, a_arr, max_disp, self.energy = _run_sweeps_numba(
            self.pos, self.pos_real, self.types,
            self.p.eps, self.p.sig,
            self.C0, self.C2, self.C4, self.rcut2,
            self.box[0], self.box[1],
            1.0 / self.cfg.T,
            max_disp, nsweeps,
            use_autotune, self.cfg.target_accept, self.cfg.tune_interval_sweeps,
            self.energy)
        
        if record_sweeps:                  # <-- only keep sweep traces when asked
            energies.extend(e_arr.tolist())
            acc_hist.extend(a_arr.tolist())
        return max_disp

    def run(self) -> Dict[str, Array]:
        beta = 1.0 / self.cfg.T
        max_disp = float(self.cfg.max_disp)
        energies: List[float] = []
        acc_hist: List[float] = []

        # Seed Numba's RNG once per run for deterministic sweeps
        if (self._seed_numba_int is not None) and (not self._numba_rng_seeded):
            _seed_numba(self._seed_numba_int)
            self._numba_rng_seeded = True

        if self.cfg.n_samples <= 0:
            max_disp = self._run_block(self.cfg.sweeps, beta, max_disp, energies, acc_hist)
            return {
                "energies": np.asarray(energies, dtype=np.float64),
                "acceptance": np.asarray(acc_hist, dtype=np.float64),
                "final_max_disp": max_disp,
                "final_positions": self.pos.copy(),
                "final_real_positions": self.pos_real.copy(),
                "initial_positions": self.initial_positions.copy(),
                "types": self.types.copy(),
                "box": self.box.copy(),
            }

        # --- Relaxation (burn-in) phase with interval snapshots ---
        si = self.cfg.sample_interval
        remaining = int(self.cfg.burn_in_sweeps)
        # relax_wrapped_list: List[np.ndarray] = []
        # relax_real_list: List[np.ndarray] = []
        # relax_wrapped_list.append(self.pos.copy())
        # relax_real_list.append(self.pos_real.copy())
        while remaining >= si:
            max_disp = self._run_block(si, beta, max_disp, energies, acc_hist, record_sweeps=False)
            # relax_wrapped_list.append(self.pos.copy())
            # relax_real_list.append(self.pos_real.copy())
            remaining -= si
        if remaining > 0:
            # run leftover sweeps and record a final snapshot
            max_disp = self._run_block(remaining, beta, max_disp, energies, acc_hist, record_sweeps=False)
            # relax_wrapped_list.append(self.pos.copy())
            # relax_real_list.append(self.pos_real.copy())

        # relaxation_positions = np.stack(relax_wrapped_list, axis=0)          # (R, N, 2)
        # relaxation_real_positions = np.stack(relax_real_list, axis=0)         # (R, N, 2)
        # relaxation_final_wrapped = relaxation_positions[-1]
        # relaxation_final_real = relaxation_real_positions[-1]

        # --- Sampling phase ---
        N = len(self.pos)
        nS = int(self.cfg.n_samples)
        samples_pos = np.empty((nS, N, 2), dtype=np.float64)
        samples_real_pos = np.empty((nS, N, 2), dtype=np.float64)
        samples_E = np.empty(nS, dtype=np.float64)
        # Freeze proposal step size during sampling to keep kernel stationary
        sampling_autotune = False

        # NEW: record real positions every sweep within the first window
        si = int(self.cfg.sample_interval)
        first_window_real_positions: Optional[np.ndarray] = None

        if nS > 0 and si > 0:
            # Run first interval sweep-by-sweep and capture real positions
            first_window_real_positions = np.empty((si, N, 2), dtype=np.float64)
            for k in range(si):
                max_disp = self._run_block(1, beta, max_disp, energies, acc_hist,
                                           record_sweeps=True, autotune=sampling_autotune)
                first_window_real_positions[k] = self.pos_real
            print("Captured first sampling window real positions:", first_window_real_positions.shape)
            # First regular sample at the end of the first interval
            samples_pos[0] = self.pos
            samples_real_pos[0] = self.pos_real
            samples_E[0] = self.energy

            # Remaining samples at the usual interval
            for s_idx in trange(1, nS):
                max_disp = self._run_block(self.cfg.sample_interval, beta, max_disp, energies, acc_hist,
                                           record_sweeps=True, autotune=sampling_autotune)
                samples_pos[s_idx] = self.pos
                samples_real_pos[s_idx] = self.pos_real
                samples_E[s_idx] = self.energy
        else:
            # Fallback (no samples or zero interval)
            for s_idx in trange(nS):
                max_disp = self._run_block(self.cfg.sample_interval, beta, max_disp, energies, acc_hist,
                                           record_sweeps=True, autotune=sampling_autotune)
                samples_pos[s_idx] = self.pos
                samples_real_pos[s_idx] = self.pos_real
                samples_E[s_idx] = self.energy

        # Convert acceptance history once for reuse below
        acceptance_array = np.asarray(acc_hist, dtype=np.float64)
        final_max_disp = max_disp

        if self.cfg.save_samples_path is not None:
            save_dir = os.path.dirname(self.cfg.save_samples_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            np.savez_compressed(
                self.cfg.save_samples_path,
                positions=samples_pos,
                positions_real=samples_real_pos,
                energies=samples_E,
                types=self.types,
                box=self.box,
                T=self.cfg.T,
                rho=self.cfg.rho,
                burn_in_sweeps=self.cfg.burn_in_sweeps,
                sample_interval=self.cfg.sample_interval,
                n_samples=self.cfg.n_samples,
                final_max_disp=final_max_disp,
                initial_positions=self.initial_positions,
                # Acceptance ratio time series (per recorded sweep)
                acceptance=acceptance_array,
                acceptance_sweep=np.arange(acceptance_array.size, dtype=np.int64),
                # NEW: per-sweep real positions in the first sampling window
                first_window_real_positions=first_window_real_positions,
                # Simulation config for reproducibility
                seed=self.cfg.seed,
                sweeps=self.cfg.sweeps,
                max_disp=self.cfg.max_disp,
                autotune=self.cfg.autotune,
                target_accept=self.cfg.target_accept,
                tune_interval_sweeps=self.cfg.tune_interval_sweeps,
                composition_ratio=np.asarray(self.cfg.composition_ratio, dtype=np.int32),
                composition_counts=self.counts,
                N=self.cfg.N,
                eps=self.p.eps,
                sig=self.p.sig,
                C0=self.p.C0,
                C2=self.p.C2,
                C4=self.p.C4,
                rcut_factor=self.p.rcut_factor,
            )

        return {
            "energies": samples_E,
            "acceptance": acceptance_array,
            "final_max_disp": final_max_disp,
            "final_positions": self.pos.copy(),
            "final_real_positions": self.pos_real.copy(),
            "initial_positions": self.initial_positions.copy(),
            # ...existing fields...
            "types": self.types.copy(),
            "box": self.box.copy(),
            "samples_positions": samples_pos,
            "samples_real_positions": samples_real_pos,
            "samples_energies": samples_E,
            # NEW: per-sweep real positions in the first sampling window
            "first_window_real_positions": first_window_real_positions,
        }

# ========================== Convenience API ==========================

def run_ka2d_mc(
    N: int = 176,
    T: float = 0.5,
    sweeps: int = 2000,
    rho: float = 1.19,
    seed: int = 0,
    ternary: bool = True,
    composition_ratio: Tuple[int, ...] = (5, 3, 3),
    n_samples: int = 0,
    burn_in_sweeps: int = 5,
    sample_interval: int = 1,
    save_samples_path: Optional[str] = None,
) -> dict:
    params = default_ternary_params() if ternary else example_binary_params()
    cfg = MCConfig(
        N=N, T=T, sweeps=sweeps, rho=rho, seed=seed,
        composition_ratio=composition_ratio,
        n_samples=n_samples,
        burn_in_sweeps=burn_in_sweeps,
        sample_interval=sample_interval,
        save_samples_path=save_samples_path,
    )
    sim = KA2DMCSimulatorNumba(params, cfg)
    return sim.run()


# ========================== Example usage ==========================

if __name__ == "__main__":
    #10 independent runs with different random seeds
    import pathlib
    output_dir = pathlib.Path(
        "/Users/xuhengyuan/Downloads/Notes/Simulation_repo/MCMC/ternary_glass_forming_liquid/Data/Testing/N22_rho1p19_T0p67"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use different fixed seeds for each run
    # Use computer time as seed
    for run_id in range(5):
        save_path = output_dir / f"{run_id}_Number_100k.npz"
        result = run_ka2d_mc(
            N=22,
            T=1/1.5,
            sweeps=0,
            composition_ratio=(5, 3, 3),
            rho=1.19,
            seed=run_id,
            n_samples=5_00_000,
            burn_in_sweeps=1_000_000,
            sample_interval=100,
            save_samples_path=str(save_path),
        )
        print(f"Run {run_id} completed, results saved to {save_path}")
        print("acceptance rate:", np.mean(result["acceptance"]))
        print("final max displacement:", result["final_max_disp"])
    print("All runs completed.")

    
