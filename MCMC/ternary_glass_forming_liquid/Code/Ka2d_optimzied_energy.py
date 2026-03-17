from __future__ import annotations
from dataclasses import dataclass
import json
import os
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Callable
import numpy as np

try:
    from numba import njit, prange
except Exception as e:
    raise RuntimeError(
        "This module requires numba. Install with `pip install numba`."
    ) from e

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
    max_disp: float = 0.095
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
    log_path: Optional[str] = None
    job_id: Optional[str] = None


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

@njit(cache=True, fastmath=True)
def _particle_energy_diff_numba(i: int,
                                pos: Array,
                                types: Array,
                                eps: Array,
                                sig: Array,
                                C0: float, C2: float, C4: float,
                                rcut2: Array,
                                Lx: float, Ly: float,
                                oldx: float, oldy: float,
                                newx: float, newy: float) -> float:
    """
    Local energy difference for particle i from a trial move.

    Only interactions involving particle i change, so the Metropolis dE
    can be computed from i's local neighborhood alone.

    This implementation computes old and new contributions in one traversal
    over j, avoiding two full i-vs-all loops per attempted move.
    """
    ti = types[i]
    N = pos.shape[0]
    dE = 0.0
    for j in range(N):
        if j == i:
            continue
        tj = types[j]
        rc2 = rcut2[ti, tj]

        # old position contribution
        dx_old = pos[j, 0] - oldx
        dx_old -= Lx * np.rint(dx_old / Lx)
        dy_old = pos[j, 1] - oldy
        dy_old -= Ly * np.rint(dy_old / Ly)
        r2_old = dx_old * dx_old + dy_old * dy_old

        if r2_old > 0.0 and r2_old < rc2:
            sig_ab = sig[ti, tj]
            s2 = r2_old / (sig_ab * sig_ab)
            inv2 = 1.0 / s2
            inv6 = inv2 * inv2 * inv2
            inv12 = inv6 * inv6
            g = (inv12 - inv6) + C0 + C2 * s2 + C4 * (s2 * s2)
            dE -= 4.0 * eps[ti, tj] * g

        # new position contribution
        dx_new = pos[j, 0] - newx
        dx_new -= Lx * np.rint(dx_new / Lx)
        dy_new = pos[j, 1] - newy
        dy_new -= Ly * np.rint(dy_new / Ly)
        r2_new = dx_new * dx_new + dy_new * dy_new

        if r2_new > 0.0 and r2_new < rc2:
            sig_ab = sig[ti, tj]
            s2 = r2_new / (sig_ab * sig_ab)
            inv2 = 1.0 / s2
            inv6 = inv2 * inv2 * inv2
            inv12 = inv6 * inv6
            g = (inv12 - inv6) + C0 + C2 * s2 + C4 * (s2 * s2)
            dE += 4.0 * eps[ti, tj] * g
    return dE


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
        i = np.random.randint(0, N)   # detail balance
        oldx = pos[i, 0]
        oldy = pos[i, 1]

        # raw trial displacement (unwrapped)
        dx_trial = (np.random.random() - 0.5) * 2.0 * max_disp
        dy_trial = (np.random.random() - 0.5) * 2.0 * max_disp
        tx_raw = oldx + dx_trial
        ty_raw = oldy + dy_trial
        # wrapped coordinates
        tx = tx_raw % Lx
        ty = ty_raw % Ly

        # Local dE is sufficient because only pairs involving i change.
        # Compute old and new contributions in one traversal over neighbors.
        dE = _particle_energy_diff_numba(i, pos, types, eps, sig, C0, C2, C4,
                                         rcut2, Lx, Ly, oldx, oldy, tx, ty)

        if np.random.random() < np.exp(-beta * dE):
            pos[i, 0] = tx
            pos[i, 1] = ty
            pos_real[i, 0] += dx_trial
            pos_real[i, 1] += dy_trial
            energy += dE
            accepts += 1
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
    Returns (accepts[...], final_max_disp, final_energy)
    """
    accepts_hist = np.empty(sweeps, dtype=np.float64)
    max_disp = max_disp0
    energy = energy0

    for s in range(sweeps):
        acc, tot, energy = _sweep_numba(pos, pos_real, types, eps, sig,
                                        C0, C2, C4, rcut2,
                                        Lx, Ly, beta, max_disp, energy)
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

    return accepts_hist, max_disp, energy


@njit(cache=True)
def _seed_numba(seed_int: int) -> None:
    """
    Seed Numba's legacy RNG used by np.random.* inside njit regions.
    """
    np.random.seed(seed_int)


class ProgressLogger:
    """
    Lightweight per-run logger that flushes every write for tail -f monitoring.
    """
    def __init__(self, path: Optional[str], job_id: str, seed: Optional[int]):
        self.path = path
        self.job_id = job_id
        self.seed = seed
        self._fh = None
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # line-buffered for frequent flushes
            self._fh = open(path, "a", encoding="utf-8", buffering=1)

    def log(self, sweep: int, window: int, samples_written: int,
            acceptance_mean: Optional[float], note: str = "") -> None:
        if self._fh is None:
            return
        ts = datetime.now().isoformat(timespec="seconds")
        acc_s = f"{acceptance_mean:.6g}" if acceptance_mean is not None else "n/a"
        line = (
            f"{ts} job={self.job_id} seed={self.seed} sweep={sweep} "
            f"window={window} samples_written={samples_written} acc_mean={acc_s}"
        )
        if note:
            line += f" note={note}"
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class ChunkedStorage:
    """
    Incremental writers for samples and acceptance without accumulating in RAM.
    Creates deterministic file names inside <base>__chunks/.
    """
    def __init__(self, base_path: Optional[str], n_saved: int, N: int, acc_len: int):
        self.base_path = base_path
        self.n_saved = n_saved
        self.N = N
        self.acc_len = acc_len
        self.acc_written = 0
        self.sample_written = 0

        if base_path is None:
            self.chunk_dir = None
            self.positions = None
            self.positions_real = None
            self.acceptance = None
            self.acceptance_sweep = None
            self.paths = {}
            return

        root, _ = os.path.splitext(base_path)
        self.chunk_dir = root + "__chunks"
        os.makedirs(self.chunk_dir, exist_ok=True)

        self.paths = {
            "positions": os.path.join(self.chunk_dir, "positions.npy") if n_saved > 0 else None,
            "positions_real": os.path.join(self.chunk_dir, "positions_real.npy") if n_saved > 0 else None,
            "acceptance": os.path.join(self.chunk_dir, "acceptance.npy") if acc_len > 0 else None,
            "acceptance_sweep": os.path.join(self.chunk_dir, "acceptance_sweep.npy") if acc_len > 0 else None,
            "types": os.path.join(self.chunk_dir, "types.npy"),
            "box": os.path.join(self.chunk_dir, "box.npy"),
            "initial_positions": os.path.join(self.chunk_dir, "initial_positions.npy"),
            "final_positions": os.path.join(self.chunk_dir, "final_positions.npy"),
            "final_real_positions": os.path.join(self.chunk_dir, "final_real_positions.npy"),
            "manifest": root + "__manifest.json",
        }

        self.positions = None
        self.positions_real = None
        if n_saved > 0:
            self.positions = np.lib.format.open_memmap(
                self.paths["positions"], mode="w+", dtype=np.float64, shape=(n_saved, N, 2)
            )
            self.positions_real = np.lib.format.open_memmap(
                self.paths["positions_real"], mode="w+", dtype=np.float64, shape=(n_saved, N, 2)
            )

        self.acceptance = None
        self.acceptance_sweep = None
        if acc_len > 0:
            self.acceptance = np.lib.format.open_memmap(
                self.paths["acceptance"], mode="w+", dtype=np.float64, shape=(acc_len,)
            )
            self.acceptance_sweep = np.lib.format.open_memmap(
                self.paths["acceptance_sweep"], mode="w+", dtype=np.int64, shape=(acc_len,)
            )

    def write_acceptance(self, arr: Array) -> None:
        if self.acceptance is None or arr.size == 0:
            return
        n = arr.size
        end = self.acc_written + n
        self.acceptance[self.acc_written:end] = arr
        self.acceptance_sweep[self.acc_written:end] = np.arange(self.acc_written, end, dtype=np.int64)
        self.acc_written = end

    def write_sample(self, pos: Array, pos_real: Array) -> None:
        if self.positions is None:
            return
        if self.sample_written >= self.n_saved:
            raise RuntimeError("Attempted to write more samples than allocated.")
        self.positions[self.sample_written] = pos
        self.positions_real[self.sample_written] = pos_real
        self.sample_written += 1

    def finalize_manifest(self, meta: Dict[str, object]) -> Optional[str]:
        if self.chunk_dir is None:
            return None
        # flush memmaps
        for mm in (self.positions, self.positions_real, self.acceptance, self.acceptance_sweep):
            if mm is not None:
                mm.flush()

        manifest = {
            "version": 1,
            "n_saved": self.n_saved,
            "acc_len": self.acc_len,
            "paths": self.paths,
            "meta": meta,
        }
        with open(self.paths["manifest"], "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        if self.base_path is not None:
            # Write a small pointer file at the original save path for compatibility/debugging.
            with open(self.base_path, "w", encoding="utf-8") as ph:
                json.dump({"manifest": self.paths["manifest"], "chunks": self.chunk_dir}, ph)
        return self.paths["manifest"]


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
               accept_writer: Optional[Callable[[Array], None]],
               record_sweeps: bool = True,
               autotune: Optional[bool] = None) -> tuple:
        use_autotune = self.cfg.autotune if autotune is None else autotune
        a_arr, max_disp, self.energy = _run_sweeps_numba(
            self.pos, self.pos_real, self.types,
            self.p.eps, self.p.sig,
            self.C0, self.C2, self.C4, self.rcut2,
            self.box[0], self.box[1],
            1.0 / self.cfg.T,
            max_disp, nsweeps,
            use_autotune, self.cfg.target_accept, self.cfg.tune_interval_sweeps,
            self.energy)
        
        if record_sweeps and accept_writer is not None:
            accept_writer(a_arr)
        return max_disp, (a_arr if record_sweeps else None)

    def run(self) -> Dict[str, Array]:
        beta = 1.0 / self.cfg.T
        max_disp = float(self.cfg.max_disp)
        acc_sum = 0.0
        acc_count = 0

        # Seed Numba's RNG once per run for deterministic sweeps
        if (self._seed_numba_int is not None) and (not self._numba_rng_seeded):
            _seed_numba(self._seed_numba_int)
            self._numba_rng_seeded = True

        total_sweeps_recorded = 0
        total_sweeps_done = 0
        log_path = self.cfg.log_path
        if log_path is None and self.cfg.save_samples_path is not None:
            root, _ = os.path.splitext(self.cfg.save_samples_path)
            log_path = root + "__worker.log"
        logger = ProgressLogger(
            path=log_path,
            job_id=self.cfg.job_id or f"N{self.cfg.N}_T{self.cfg.T}_seed{self.cfg.seed}",
            seed=self.cfg.seed,
        )

        if self.cfg.n_samples <= 0:
            acc_len = int(self.cfg.sweeps) if self.cfg.sweeps > 0 else 0
            storage = ChunkedStorage(self.cfg.save_samples_path, 0, self.cfg.N, acc_len)
            max_disp, a_arr = self._run_block(self.cfg.sweeps, beta, max_disp, storage.write_acceptance)
            if a_arr is not None:
                acc_sum += float(np.sum(a_arr))
                acc_count += a_arr.size
                total_sweeps_recorded += a_arr.size
            total_sweeps_done += int(self.cfg.sweeps)
            acc_count = storage.acc_written if storage.acc_written else acc_count
            logger.log(total_sweeps_done, 0, 0, (acc_sum / acc_count) if acc_count else None,
                       note="run complete (no sampling)")
            if self.cfg.save_samples_path is not None:
                np.save(storage.paths["types"], self.types)
                np.save(storage.paths["box"], self.box)
                np.save(storage.paths["initial_positions"], self.initial_positions)
                np.save(storage.paths["final_positions"], self.pos)
                np.save(storage.paths["final_real_positions"], self.pos_real)
            manifest_path = storage.finalize_manifest({
                "save_samples_path": self.cfg.save_samples_path,
                "T": self.cfg.T,
                "rho": self.cfg.rho,
                "burn_in_sweeps": self.cfg.burn_in_sweeps,
                "sample_interval": self.cfg.sample_interval,
                "n_samples": 0,
                "n_samples_requested": 0,
                "final_max_disp": max_disp,
                "acceptance_written": acc_count,
                "acceptance_expected": acc_len,
                "acceptance_count": acc_count,
                "initial_positions_shape": self.initial_positions.shape,
                "types_shape": self.types.shape,
                "box_shape": self.box.shape,
                "samples_written": storage.sample_written,
            }) if self.cfg.save_samples_path is not None else None

            logger.close()
            return {
                "acceptance_mean": (acc_sum / acc_count) if acc_count else None,
                "acceptance_count": acc_count,
                "acceptance_path": storage.paths.get("acceptance") if self.cfg.save_samples_path else None,
                "final_max_disp": max_disp,
                "final_positions": self.pos.copy(),
                "final_real_positions": self.pos_real.copy(),
                "initial_positions": self.initial_positions.copy(),
                "types": self.types.copy(),
                "box": self.box.copy(),
                "manifest": manifest_path,
            }

        # --- Relaxation (burn-in) phase with interval snapshots ---
        si = self.cfg.sample_interval
        if si <= 0:
            raise ValueError("sample_interval must be > 0 for streaming output.")
        remaining = int(self.cfg.burn_in_sweeps)
        nS = int(self.cfg.n_samples)
        n_saved = max(0, nS - 1)  # skip saving the first sampling window
        acc_len = (si + n_saved * si) if si > 0 else 0
        storage = ChunkedStorage(self.cfg.save_samples_path, n_saved, self.cfg.N, acc_len)

        # Relaxation sweeps (no recording)
        while remaining >= si:
            max_disp, _ = self._run_block(si, beta, max_disp, storage.write_acceptance, record_sweeps=False)
            remaining -= si
            total_sweeps_done += si
        if remaining > 0:
            max_disp, _ = self._run_block(remaining, beta, max_disp, storage.write_acceptance, record_sweeps=False)
            total_sweeps_done += remaining

        # --- Sampling phase ---
        sampling_autotune = False
        # First window (unsaved sample)
        if nS > 0 and si > 0:
            for _ in range(si):
                max_disp, a_arr = self._run_block(1, beta, max_disp, storage.write_acceptance,
                                                  record_sweeps=True, autotune=sampling_autotune)
                if a_arr is not None and a_arr.size:
                    acc_sum += float(np.sum(a_arr))
                    acc_count += a_arr.size
                    total_sweeps_recorded += a_arr.size
                total_sweeps_done += 1
            logger.log(total_sweeps_done, 0, 0, (acc_sum / acc_count) if acc_count else None,
                       note="completed first sampling window (unsaved)")

        # Remaining samples
        samples_written = 0
        for s_idx in range(n_saved):
            max_disp, a_arr = self._run_block(self.cfg.sample_interval, beta, max_disp, storage.write_acceptance,
                                              record_sweeps=True, autotune=sampling_autotune)
            if a_arr is not None and a_arr.size:
                acc_sum += float(np.sum(a_arr))
                acc_count += a_arr.size
                total_sweeps_recorded += a_arr.size
            total_sweeps_done += self.cfg.sample_interval
            storage.write_sample(self.pos, self.pos_real)
            samples_written += 1
            logger.log(total_sweeps_done, s_idx + 1, samples_written,
                       (acc_sum / acc_count) if acc_count else None)

        acc_count = storage.acc_written if storage.acc_written else acc_count
        acceptance_mean = (acc_sum / acc_count) if acc_count else None
        final_max_disp = max_disp

        # Save small metadata + manifest and small arrays
        if self.cfg.save_samples_path is not None:
            np.save(storage.paths["types"], self.types)
            np.save(storage.paths["box"], self.box)
            np.save(storage.paths["initial_positions"], self.initial_positions)
            np.save(storage.paths["final_positions"], self.pos)
            np.save(storage.paths["final_real_positions"], self.pos_real)

            manifest_meta = {
                "save_samples_path": self.cfg.save_samples_path,
                "T": self.cfg.T,
                "rho": self.cfg.rho,
                "burn_in_sweeps": self.cfg.burn_in_sweeps,
                "sample_interval": self.cfg.sample_interval,
                "n_samples": n_saved,
                "n_samples_requested": nS,
                "final_max_disp": final_max_disp,
                "seed": self.cfg.seed,
                "sweeps": self.cfg.sweeps,
                "max_disp": self.cfg.max_disp,
                "autotune": self.cfg.autotune,
                "target_accept": self.cfg.target_accept,
                "tune_interval_sweeps": self.cfg.tune_interval_sweeps,
                "composition_ratio": np.asarray(self.cfg.composition_ratio, dtype=np.int32).tolist(),
                "composition_counts": self.counts.tolist(),
                "N": self.cfg.N,
                "eps": self.p.eps.tolist(),
                "sig": self.p.sig.tolist(),
                "C0": self.p.C0,
                "C2": self.p.C2,
                "C4": self.p.C4,
                "rcut_factor": self.p.rcut_factor,
                "acceptance_count": acc_count,
                "acceptance_expected": acc_len,
                "samples_written": storage.sample_written,
            }
            manifest_path = storage.finalize_manifest(manifest_meta)
        else:
            manifest_path = None

        logger.log(total_sweeps_done, n_saved, samples_written, acceptance_mean, note="run complete")
        logger.close()

        return {
            "acceptance_mean": acceptance_mean,
            "acceptance_count": acc_count,
            "acceptance_path": storage.paths.get("acceptance") if self.cfg.save_samples_path else None,
            "acceptance_sweep_path": storage.paths.get("acceptance_sweep") if self.cfg.save_samples_path else None,
            "final_max_disp": final_max_disp,
            "final_positions": self.pos.copy(),
            "final_real_positions": self.pos_real.copy(),
            "initial_positions": self.initial_positions.copy(),
            "types": self.types.copy(),
            "box": self.box.copy(),
            "types_path": storage.paths.get("types") if self.cfg.save_samples_path else None,
            "box_path": storage.paths.get("box") if self.cfg.save_samples_path else None,
            "positions_path": storage.paths.get("positions") if self.cfg.save_samples_path else None,
            "positions_real_path": storage.paths.get("positions_real") if self.cfg.save_samples_path else None,
            "manifest": manifest_path,
        }

# ========================== Convenience API ==========================

def load_full_output(manifest_path: str, mmap_mode: Optional[str] = "r") -> Dict[str, object]:
    """
    Load a manifest produced by this module and reconstruct arrays matching the
    legacy output schema. Arrays are loaded with mmap_mode="r" by default to
    avoid large RAM spikes; pass None to load fully into memory.
    """
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    paths = manifest.get("paths", {})
    meta = manifest.get("meta", {})

    def _load(path_key: str):
        path = paths.get(path_key)
        if path is None:
            return None
        return np.load(path, mmap_mode=mmap_mode)

    return {
        "positions": _load("positions"),
        "positions_real": _load("positions_real"),
        "acceptance": _load("acceptance"),
        "acceptance_sweep": _load("acceptance_sweep"),
        "types": _load("types"),
        "box": _load("box"),
        "T": meta.get("T"),
        "rho": meta.get("rho"),
        "burn_in_sweeps": meta.get("burn_in_sweeps"),
        "sample_interval": meta.get("sample_interval"),
        "n_samples": meta.get("n_samples"),
        "n_samples_requested": meta.get("n_samples_requested"),
        "final_max_disp": meta.get("final_max_disp"),
        "initial_positions": _load("initial_positions"),
        "final_positions": _load("final_positions"),
        "final_real_positions": _load("final_real_positions"),
        "seed": meta.get("seed"),
        "sweeps": meta.get("sweeps"),
        "max_disp": meta.get("max_disp"),
        "autotune": meta.get("autotune"),
        "target_accept": meta.get("target_accept"),
        "tune_interval_sweeps": meta.get("tune_interval_sweeps"),
        "composition_ratio": np.asarray(meta.get("composition_ratio")) if meta.get("composition_ratio") is not None else None,
        "composition_counts": np.asarray(meta.get("composition_counts")) if meta.get("composition_counts") is not None else None,
        "N": meta.get("N"),
        "eps": np.asarray(meta.get("eps")) if meta.get("eps") is not None else None,
        "sig": np.asarray(meta.get("sig")) if meta.get("sig") is not None else None,
        "C0": meta.get("C0"),
        "C2": meta.get("C2"),
        "C4": meta.get("C4"),
        "rcut_factor": meta.get("rcut_factor"),
        "acceptance_count": meta.get("acceptance_count"),
        "acceptance_expected": meta.get("acceptance_expected"),
        "samples_written": meta.get("samples_written"),
    }

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
    rcut_factor: float = 2.5,
    max_disp: float = 0.095,
    target_accept: float = 0.35,
    save_samples_path: Optional[str] = None,
    log_path: Optional[str] = None,
    job_id: Optional[str] = None,
) -> dict:
    params = default_ternary_params() if ternary else example_binary_params()
    params.rcut_factor = float(rcut_factor)
    cfg = MCConfig(
        N=N, T=T, sweeps=sweeps, rho=rho, seed=seed, max_disp=max_disp, target_accept=target_accept,
        composition_ratio=composition_ratio,
        n_samples=n_samples,
        burn_in_sweeps=burn_in_sweeps,
        sample_interval=sample_interval,
        save_samples_path=save_samples_path,
        log_path=log_path,
        job_id=job_id,
    )
    sim = KA2DMCSimulatorNumba(params, cfg)
    return sim.run()


# ========================== Example usage ==========================

if __name__ == "__main__":
    #10 independent runs with different random seeds
    import pathlib
    output_dir = pathlib.Path(
        "/Users/xuhengyuan/Downloads/Notes/Simulation_repo/MCMC/ternary_glass_forming_liquid/Data/Training/N43_L6_T0p33/Longer_interval_trials"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use different fixed seeds for each run
    # Use computer time as seed
    for run_id in range(5):
        save_path = output_dir / f"N43_L6_T0p33_{run_id}_interval_100k.npz"
        result = run_ka2d_mc(
            N=43,
            T=0.32,
            sweeps=0,
            composition_ratio=(20, 11, 12),
            rho=43/36,
            seed=run_id,
            n_samples=5_00_000,
            burn_in_sweeps=5_000_000,
            sample_interval=100000,
            save_samples_path=str(save_path),
        )
        print(f"Run {run_id} completed, results saved to {save_path}")
        print("acceptance rate:", result.get("acceptance_mean"))
        print("final max displacement:", result.get("final_max_disp"))
    print("All runs completed.")

    

