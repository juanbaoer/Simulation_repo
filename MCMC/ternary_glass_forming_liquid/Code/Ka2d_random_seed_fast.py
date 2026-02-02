from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Tuple, Optional, Dict
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
        dx = dr[i, 0]
        dx -= Lx * np.rint(dx / Lx)
        dr[i, 0] = dx
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
        dx = pos[j, 0] - ri0
        dx -= Lx * np.rint(dx / Lx)
        dy = pos[j, 1] - ri1
        dy -= Ly * np.rint(dy / Ly)
        r2 = dx * dx + dy * dy
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

        dx_trial = (np.random.random() - 0.5) * 2.0 * max_disp
        dy_trial = (np.random.random() - 0.5) * 2.0 * max_disp
        tx_raw = oldx + dx_trial
        ty_raw = oldy + dy_trial
        tx = tx_raw % Lx
        ty = ty_raw % Ly

        pos[i, 0] = tx
        pos[i, 1] = ty
        pos_real[i, 0] += dx_trial
        pos_real[i, 1] += dy_trial

        new_Ei = _particle_energy_numba(i, pos, types, eps, sig, C0, C2, C4,
                                        rcut2, Lx, Ly)
        dE = new_Ei - old_Ei

        if np.random.random() < np.exp(-beta * dE):
            energy += dE
            accepts += 1
        else:
            pos[i, 0] = oldx
            pos[i, 1] = oldy
            pos_real[i, 0] -= dx_trial
            pos_real[i, 1] -= dy_trial
    return accepts, N, energy


@njit(cache=True, fastmath=True)
def _run_sweeps_no_record(pos: Array, pos_real: Array, types: Array,
                          eps: Array, sig: Array,
                          C0: float, C2: float, C4: float,
                          rcut2: Array,
                          Lx: float, Ly: float,
                          beta: float,
                          max_disp0: float,
                          sweeps: int,
                          autotune: bool, target_accept: float, tune_interval: int,
                          energy0: float,
                          acc_buf: Array) -> tuple:
    """
    Run sweeps without recording traces. acc_buf is reused for autotune stats.
    """
    max_disp = max_disp0
    energy = energy0

    for s in range(sweeps):
        acc, tot, energy = _sweep_numba(pos, pos_real, types, eps, sig,
                                        C0, C2, C4, rcut2,
                                        Lx, Ly, beta, max_disp, energy)
        acc_buf[s] = acc / tot

        if autotune and ((s + 1) % tune_interval == 0):
            start = max(0, s + 1 - tune_interval)
            recent = 0.0
            for k in range(start, s + 1):
                recent += acc_buf[k]
            recent /= (s + 1 - start)
            if recent < target_accept:
                max_disp *= 0.9
            else:
                max_disp *= 1.1
            if max_disp < 0.01:
                max_disp = 0.01
            elif max_disp > 0.8:
                max_disp = 0.8

    return max_disp, energy


@njit(cache=True, fastmath=True)
def _run_sweeps_record(pos: Array, pos_real: Array, types: Array,
                       eps: Array, sig: Array,
                       C0: float, C2: float, C4: float,
                       rcut2: Array,
                       Lx: float, Ly: float,
                       beta: float,
                       max_disp0: float,
                       sweeps: int,
                       autotune: bool, target_accept: float, tune_interval: int,
                       energy0: float,
                       energies_out: Array,
                       acc_out: Array,
                       offset: int,
                       first_window_out: Optional[Array]) -> tuple:
    """
    Run sweeps and record per-sweep energy/acceptance into preallocated arrays.
    Optionally capture real positions for the first sampling window.
    """
    max_disp = max_disp0
    energy = energy0
    idx = offset

    for s in range(sweeps):
        acc, tot, energy = _sweep_numba(pos, pos_real, types, eps, sig,
                                        C0, C2, C4, rcut2,
                                        Lx, Ly, beta, max_disp, energy)
        energies_out[idx] = energy
        acc_out[idx] = acc / tot

        if first_window_out is not None and s < first_window_out.shape[0]:
            first_window_out[s] = pos_real

        if autotune and ((s + 1) % tune_interval == 0):
            start = max(0, s + 1 - tune_interval)
            recent = 0.0
            for k in range(start, s + 1):
                recent += acc_out[offset + k]
            recent /= (s + 1 - start)
            if recent < target_accept:
                max_disp *= 0.9
            else:
                max_disp *= 1.1
            if max_disp < 0.01:
                max_disp = 0.01
            elif max_disp > 0.8:
                max_disp = 0.8
        idx += 1

    return max_disp, energy, idx


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


@njit(cache=True, fastmath=True)
def _run_burnin_and_sample(
    pos: Array,
    pos_real: Array,
    types: Array,
    eps: Array,
    sig: Array,
    C0: float,
    C2: float,
    C4: float,
    rcut2: Array,
    Lx: float,
    Ly: float,
    beta: float,
    max_disp0: float,
    burn_in_sweeps: int,
    n_samples: int,
    sample_interval: int,
    autotune_burnin: bool,
    target_accept: float,
    tune_interval: int,
    samples_pos_out: Array,
    samples_real_out: Array,
    samples_E_out: Array,
    energies_trace_out: Array,
    acceptance_trace_out: Array,
    first_window_real_positions_out: Optional[Array],
    energy0: float,
) -> tuple:
    """
    Run burn-in followed by sampling. Records all sampling sweeps into
    preallocated arrays to avoid Python overhead.
    """
    max_disp = max_disp0
    energy = energy0
    trace_offset = 0

    if burn_in_sweeps > 0:
        block = sample_interval
        if block <= 0:
            block = burn_in_sweeps
        acc_buf_len = block if block > 0 else 1
        acc_buf = np.empty(acc_buf_len, dtype=np.float64)
        remaining = burn_in_sweeps
        while block > 0 and remaining >= block:
            max_disp, energy = _run_sweeps_no_record(
                pos, pos_real, types,
                eps, sig, C0, C2, C4, rcut2,
                Lx, Ly, beta,
                max_disp, block,
                autotune_burnin, target_accept, tune_interval,
                energy,
                acc_buf,
            )
            remaining -= block
        if remaining > 0:
            if remaining > acc_buf.shape[0]:
                acc_buf = np.empty(remaining, dtype=np.float64)
            max_disp, energy = _run_sweeps_no_record(
                pos, pos_real, types,
                eps, sig, C0, C2, C4, rcut2,
                Lx, Ly, beta,
                max_disp, remaining,
                autotune_burnin, target_accept, tune_interval,
                energy,
                acc_buf,
            )

    if n_samples > 0 and sample_interval > 0:
        max_disp, energy, trace_offset = _run_sweeps_record(
            pos, pos_real, types,
            eps, sig, C0, C2, C4, rcut2,
            Lx, Ly, beta,
            max_disp, sample_interval,
            False, target_accept, tune_interval,
            energy,
            energies_trace_out,
            acceptance_trace_out,
            trace_offset,
            first_window_real_positions_out,
        )
        samples_pos_out[0] = pos
        samples_real_out[0] = pos_real
        samples_E_out[0] = energy

        for s_idx in range(1, n_samples):
            max_disp, energy, trace_offset = _run_sweeps_record(
                pos, pos_real, types,
                eps, sig, C0, C2, C4, rcut2,
                Lx, Ly, beta,
                max_disp, sample_interval,
                False, target_accept, tune_interval,
                energy,
                energies_trace_out,
                acceptance_trace_out,
                trace_offset,
                None,
            )
            samples_pos_out[s_idx] = pos
            samples_real_out[s_idx] = pos_real
            samples_E_out[s_idx] = energy
    else:
        for s_idx in range(n_samples):
            max_disp, energy, trace_offset = _run_sweeps_record(
                pos, pos_real, types,
                eps, sig, C0, C2, C4, rcut2,
                Lx, Ly, beta,
                max_disp, sample_interval,
                False, target_accept, tune_interval,
                energy,
                energies_trace_out,
                acceptance_trace_out,
                trace_offset,
                None,
            )
            samples_pos_out[s_idx] = pos
            samples_real_out[s_idx] = pos_real
            samples_E_out[s_idx] = energy

    return max_disp, energy, trace_offset


# ========================== Fast, Numba MC ==========================
class KA2DMCSimulatorNumbaFast:
    """
    Numba-accelerated KA 2D MC (no neighbor list).
    Public API mirrors the reference version for easy swapping.
    Deterministic runs assume single-thread numba (NUMBA_NUM_THREADS=1).
    """

    def __init__(self, params: KA2DParams, cfg: MCConfig):
        self.p = params
        self.cfg = cfg

        self.base_seed = cfg.seed
        if self.base_seed is not None:
            ss = np.random.SeedSequence(self.base_seed)
            child_pos, child_numba = ss.spawn(2)
            self._rng_pos = np.random.default_rng(child_pos)
            self._seed_numba_int = int(child_numba.generate_state(1, dtype=np.uint32)[0])
        else:
            self._rng_pos = np.random.default_rng()
            self._seed_numba_int = None
        self._rng_types = np.random.default_rng(TYPES_TENSOR_SEED)

        self._numba_rng_seeded = False

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

        L = float(np.sqrt(cfg.N / cfg.rho))
        self.box = np.array([L, L], dtype=np.float64)

        types = np.concatenate([
            np.full(c, t, dtype=np.int32) for t, c in enumerate(self.counts)
        ])
        self._rng_types.shuffle(types)
        self.types = types

        D = self.box.shape[0]
        self.pos = self._rng_pos.random((cfg.N, D)) * self.box
        self.pos_real = self.pos.copy()
        self.initial_positions = self.pos.copy()

        self.C0, self.C2, self.C4 = float(self.p.C0), float(self.p.C2), float(self.p.C4)
        self.rcut2 = (self.p.rcut() ** 2).astype(np.float64)

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
        pts[:, 0] %= Lx
        pts[:, 1] %= Ly
        return pts

    def _seed_numba_rng_once(self) -> None:
        if (self._seed_numba_int is not None) and (not self._numba_rng_seeded):
            _seed_numba(self._seed_numba_int)
            self._numba_rng_seeded = True

    def run(self) -> Dict[str, Array]:
        beta = 1.0 / self.cfg.T
        max_disp = float(self.cfg.max_disp)
        self._seed_numba_rng_once()

        if self.cfg.n_samples <= 0:
            sweeps = int(self.cfg.sweeps)
            energies_out = np.empty(sweeps, dtype=np.float64)
            acc_out = np.empty(sweeps, dtype=np.float64)
            max_disp, self.energy, _ = _run_sweeps_record(
                self.pos, self.pos_real, self.types,
                self.p.eps, self.p.sig, self.C0, self.C2, self.C4, self.rcut2,
                self.box[0], self.box[1],
                beta,
                max_disp,
                sweeps,
                self.cfg.autotune, self.cfg.target_accept, self.cfg.tune_interval_sweeps,
                self.energy,
                energies_out,
                acc_out,
                0,
                None,
            )
            acceptance_array = acc_out
            return {
                "energies": energies_out,
                "acceptance": acceptance_array,
                "final_max_disp": max_disp,
                "final_positions": self.pos.copy(),
                "final_real_positions": self.pos_real.copy(),
                "initial_positions": self.initial_positions.copy(),
                "types": self.types.copy(),
                "box": self.box.copy(),
            }

        if self.cfg.sample_interval <= 0:
            raise ValueError("sample_interval must be positive when n_samples > 0")

        N = len(self.pos)
        nS = int(self.cfg.n_samples)
        si = int(self.cfg.sample_interval)

        samples_pos = np.empty((nS, N, 2), dtype=np.float64)
        samples_real_pos = np.empty((nS, N, 2), dtype=np.float64)
        samples_E = np.empty(nS, dtype=np.float64)

        trace_len = nS * si
        energies_trace = np.empty(trace_len, dtype=np.float64)
        acceptance_trace = np.empty(trace_len, dtype=np.float64)

        first_window_real_positions: Optional[np.ndarray] = None
        if nS > 0 and si > 0:
            first_window_real_positions = np.empty((si, N, 2), dtype=np.float64)

        final_max_disp, self.energy, trace_used = _run_burnin_and_sample(
            self.pos,
            self.pos_real,
            self.types,
            self.p.eps,
            self.p.sig,
            self.C0,
            self.C2,
            self.C4,
            self.rcut2,
            self.box[0],
            self.box[1],
            beta,
            max_disp,
            int(self.cfg.burn_in_sweeps),
            nS,
            si,
            self.cfg.autotune,
            self.cfg.target_accept,
            self.cfg.tune_interval_sweeps,
            samples_pos,
            samples_real_pos,
            samples_E,
            energies_trace,
            acceptance_trace,
            first_window_real_positions,
            self.energy,
        )

        if trace_used != trace_len:
            raise RuntimeError("Recorded sweep count mismatch in sampling trace")

        acceptance_array = acceptance_trace

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
                acceptance=acceptance_array,
                acceptance_sweep=np.arange(acceptance_array.size, dtype=np.int64),
                first_window_real_positions=first_window_real_positions,
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
            "types": self.types.copy(),
            "box": self.box.copy(),
            "samples_positions": samples_pos,
            "samples_real_positions": samples_real_pos,
            "samples_energies": samples_E,
            "first_window_real_positions": first_window_real_positions,
        }


KA2DMCSimulatorNumba = KA2DMCSimulatorNumbaFast

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
    sim = KA2DMCSimulatorNumbaFast(params, cfg)
    return sim.run()
