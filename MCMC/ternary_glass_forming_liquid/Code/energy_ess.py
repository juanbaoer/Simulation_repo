from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

from Ka2d_random_seed_optimized_logging_append import _total_energy_numba


# -----------------------------
# Data discovery / loading
# -----------------------------

def _pointer_to_manifest(npz_path: Path) -> Path | None:
    try:
        with open(npz_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(data, dict):
        manifest = data.get("manifest") or data.get("manifest_path")
        if manifest:
            return Path(manifest)
    return None


def _sort_run_key(run_name: str) -> tuple[str, float]:
    """
    Stable sort key that handles mixed run-name suffixes like:
    - ...training.1
    - ...training_2.1
    - ...seed123
    Always returns (prefix, numeric_suffix) to avoid type-mixing errors.
    """
    import re

    m = re.search(r"(.*?)(\\d+(?:\\.\\d+)?)$", run_name)
    if m:
        prefix = m.group(1)
        try:
            num = float(m.group(2))
        except Exception:  # noqa: BLE001
            num = float("inf")
        return (prefix, num)
    return (run_name, float("inf"))


def _load_summary_manifests(out_dir: Path) -> List[Tuple[str, Path]]:
    summary_path = out_dir / "summary.jsonl"
    records: List[Tuple[str, Path]] = []
    if not summary_path.exists():
        return records
    with summary_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            job = entry.get("job", {})
            job_id = job.get("job_id") or job.get("id")
            manifest_val = entry.get("manifest") or job.get("manifest")
            manifest_path = None
            if manifest_val:
                manifest_path = Path(manifest_val)
                if not manifest_path.is_absolute():
                    manifest_path = out_dir / manifest_val
            elif job_id:
                manifest_path = out_dir / job_id / "data__manifest.json"
            if manifest_path:
                records.append((job_id or manifest_path.parent.name, manifest_path))
    return records


def discover_run_manifests(out_dir: Path) -> List[Tuple[str, Path]]:
    """Discover per-run manifests and return sorted (run_name, manifest_path)."""
    # Prefer summary.jsonl if available for stable ordering
    summary_records = _load_summary_manifests(out_dir)
    if summary_records:
        manifest_records = [(name, path) for name, path in summary_records if path.exists()]
        manifest_records.sort(key=lambda item: _sort_run_key(item[0]))
        return manifest_records

    manifests: set[Path] = set()

    # Direct manifests inside per-run folders
    for m in out_dir.glob("**/data_manifest.json"):
        manifests.add(m)

    # Common naming convention (data__manifest.json)
    for m in out_dir.glob("**/*__manifest.json"):
        manifests.add(m)

    # Pointer files at legacy save path (data.npz replaced with small JSON)
    for npz in out_dir.glob("**/*.npz"):
        m = _pointer_to_manifest(npz)
        if m:
            manifests.add(m)

    manifest_records = [(m.parent.name, m) for m in sorted(manifests)]
    manifest_records = [(name, path) for name, path in manifest_records if path.exists()]
    manifest_records.sort(key=lambda item: _sort_run_key(item[0]))
    return manifest_records


def _resolve_manifest_path(path: Path) -> Path:
    """Accept a manifest path, run directory, or data.npz pointer and return a manifest path."""
    path = Path(path)
    if path.is_dir():
        candidate = path / "data__manifest.json"
        if candidate.exists():
            return candidate
        # Fallback: first manifest inside directory
        matches = list(path.glob("**/*__manifest.json"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No manifest found under directory: {path}")
    if path.suffix == ".npz":
        manifest = _pointer_to_manifest(path)
        if manifest is None:
            raise ValueError(f"data.npz does not contain a manifest pointer: {path}")
        return manifest
    return path


def load_run_minimal(manifest_path: Path, mmap_mode: str | None = "r") -> Dict[str, object]:
    """Load only required arrays and metadata for energy/ESS calculations."""
    manifest_path = _resolve_manifest_path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    paths = manifest.get("paths", {})
    meta = manifest.get("meta", {})

    def _load(path_key: str, required: bool = False):
        path = paths.get(path_key)
        if path is None:
            if required:
                raise FileNotFoundError(f"Missing path for {path_key} in {manifest_path}")
            return None
        return np.load(path, mmap_mode=mmap_mode)

    return {
        "positions": _load("positions", required=True),
        "positions_real": _load("positions_real", required=False),
        "types": _load("types", required=True),
        "box": _load("box", required=True),
        "T": meta.get("T"),
        "rho": meta.get("rho"),
        "N": meta.get("N"),
        "eps": np.asarray(meta.get("eps")) if meta.get("eps") is not None else None,
        "sig": np.asarray(meta.get("sig")) if meta.get("sig") is not None else None,
        "C0": meta.get("C0"),
        "C2": meta.get("C2"),
        "C4": meta.get("C4"),
        "rcut_factor": meta.get("rcut_factor"),
    }


# -----------------------------
# Merge samples (wrapped)
# -----------------------------

def merge_positions_from_records(
    records: List[Tuple[str, Dict[str, object]]],
    use_wrapped: bool = True,
    merged_path: Path | None = None,
    overwrite: bool = False,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], Dict[str, int]]:
    """
    Merge positions across runs into a single array (memmap if merged_path provided).
    Returns merged array, run_slices (start, end), and run_counts.
    """
    if not records:
        raise ValueError("No records provided for merging.")

    pos_key = "positions" if use_wrapped else "positions_real"

    # Determine total samples and shape
    counts: Dict[str, int] = {}
    n_particles = None
    dim = None
    total = 0
    for name, rec in records:
        pos = rec.get(pos_key)
        if pos is None:
            raise ValueError(f"Missing {pos_key} for run {name}")
        if n_particles is None:
            n_particles = pos.shape[1]
            dim = pos.shape[2]
        counts[name] = int(pos.shape[0])
        total += counts[name]

    if merged_path is not None:
        merged_path = Path(merged_path)
        if merged_path.exists() and not overwrite:
            merged = np.load(merged_path, mmap_mode="r+")
            if merged.shape != (total, n_particles, dim):
                raise ValueError(
                    f"Existing merged array has shape {merged.shape}, expected {(total, n_particles, dim)}"
                )
        else:
            merged = np.lib.format.open_memmap(
                merged_path, mode="w+", dtype=np.float64, shape=(total, n_particles, dim)
            )
    else:
        merged = np.empty((total, n_particles, dim), dtype=np.float64)

    # Copy in order and record slices
    run_slices: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for name, rec in records:
        pos = rec.get(pos_key)
        n = counts[name]
        merged[cursor : cursor + n] = np.asarray(pos, dtype=np.float64)
        run_slices[name] = (cursor, cursor + n)
        cursor += n

    return merged, run_slices, counts


# -----------------------------
# Energy computation
# -----------------------------

def validate_energy_params(records: List[Tuple[str, Dict[str, object]]]) -> Dict[str, object]:
    """Ensure all runs share the same parameters; return params from the first run."""
    if not records:
        raise ValueError("No records provided for validation.")
    _, ref = records[0]
    ref_types = np.asarray(ref["types"], dtype=np.int64)
    ref_eps = np.asarray(ref["eps"], dtype=np.float64)
    ref_sig = np.asarray(ref["sig"], dtype=np.float64)
    ref_box = np.asarray(ref["box"], dtype=np.float64)

    for name, rec in records[1:]:
        types = np.asarray(rec["types"], dtype=np.int64)
        eps = np.asarray(rec["eps"], dtype=np.float64)
        sig = np.asarray(rec["sig"], dtype=np.float64)
        box = np.asarray(rec["box"], dtype=np.float64)
        if not np.array_equal(types, ref_types):
            raise ValueError(f"types mismatch for run {name}")
        if not np.allclose(eps, ref_eps, rtol=0, atol=0):
            raise ValueError(f"eps mismatch for run {name}")
        if not np.allclose(sig, ref_sig, rtol=0, atol=0):
            raise ValueError(f"sig mismatch for run {name}")
        if not np.allclose(box, ref_box, rtol=0, atol=0):
            raise ValueError(f"box mismatch for run {name}")
        if rec.get("C0") != ref.get("C0"):
            raise ValueError(f"C0 mismatch for run {name}")
        if rec.get("C2") != ref.get("C2"):
            raise ValueError(f"C2 mismatch for run {name}")
        if rec.get("C4") != ref.get("C4"):
            raise ValueError(f"C4 mismatch for run {name}")
        if rec.get("rcut_factor") != ref.get("rcut_factor"):
            raise ValueError(f"rcut_factor mismatch for run {name}")
        if rec.get("T") != ref.get("T"):
            raise ValueError(f"T mismatch for run {name}")

    return {
        "types": ref_types,
        "eps": ref_eps,
        "sig": ref_sig,
        "box": ref_box,
        "C0": float(ref.get("C0", 0.0)),
        "C2": float(ref.get("C2", 0.0)),
        "C4": float(ref.get("C4", 0.0)),
        "rcut_factor": float(ref.get("rcut_factor", 2.5)),
        "T": float(ref.get("T")),
        "N": int(ref.get("N")),
    }


def compute_energy_batched(
    positions: np.ndarray,
    types: np.ndarray,
    eps: np.ndarray,
    sig: np.ndarray,
    box: np.ndarray,
    C0: float,
    C2: float,
    C4: float,
    rcut_factor: float,
    batch_size: int = 8192,
    show_progress: bool = True,
) -> Tuple[np.ndarray, float]:
    """Compute total energy for each sample in batches. Uses wrapped positions."""
    n_samples = positions.shape[0]
    types = np.asarray(types, dtype=np.int64)
    eps = np.asarray(eps, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    rcut2 = (rcut_factor * sig) ** 2
    Lx = float(box[0])
    Ly = float(box[1])

    energies = np.empty(n_samples, dtype=np.float64)
    it = range(0, n_samples, batch_size)
    if show_progress:
        if tqdm is None:
            raise ImportError("tqdm is required for progress display. Please install tqdm.")
        it = tqdm(it, desc="Energy batches", unit="batch")

    start = time.perf_counter()
    for start_idx in it:
        end_idx = min(start_idx + batch_size, n_samples)
        for i in range(start_idx, end_idx):
            energies[i] = _total_energy_numba(
                np.asarray(positions[i], dtype=np.float64),
                types,
                eps,
                sig,
                float(C0),
                float(C2),
                float(C4),
                np.asarray(rcut2, dtype=np.float64),
                Lx,
                Ly,
            )
    elapsed = time.perf_counter() - start
    return energies, elapsed


# -----------------------------
# Wrapped/unwrapped checks
# -----------------------------

def wrapped_range_check(positions: np.ndarray, box: np.ndarray, n_check: int = 5) -> Dict[str, np.ndarray]:
    """Check wrapped positions fall in [0, L). Uses a small subset for speed."""
    n = positions.shape[0]
    idx = np.linspace(0, n - 1, num=min(n_check, n), dtype=int)
    subset = np.asarray(positions[idx], dtype=np.float64)
    mins = subset.min(axis=(0, 1))
    maxs = subset.max(axis=(0, 1))
    return {"min": mins, "max": maxs, "box": np.asarray(box, dtype=np.float64)}


def compare_wrapped_unwrapped_energy(
    record: Dict[str, object],
    n_check: int = 3,
    seed: int = 0,
) -> Dict[str, np.ndarray] | None:
    """Compute energy on a few frames using wrapped vs unwrapped positions."""
    positions = record.get("positions")
    positions_real = record.get("positions_real")
    if positions is None or positions_real is None:
        return None

    rng = np.random.default_rng(seed)
    n = positions.shape[0]
    idx = rng.choice(n, size=min(n_check, n), replace=False)

    types = np.asarray(record.get("types"), dtype=np.int64)
    eps = np.asarray(record.get("eps"), dtype=np.float64)
    sig = np.asarray(record.get("sig"), dtype=np.float64)
    box = np.asarray(record.get("box"), dtype=np.float64)
    rcut_factor = float(record.get("rcut_factor", 2.5))
    rcut2 = (rcut_factor * sig) ** 2
    Lx = float(box[0])
    Ly = float(box[1])

    u_wrapped = []
    u_unwrapped = []
    for i in idx:
        u_wrapped.append(
            _total_energy_numba(
                np.asarray(positions[i], dtype=np.float64),
                types,
                eps,
                sig,
                float(record.get("C0", 0.0)),
                float(record.get("C2", 0.0)),
                float(record.get("C4", 0.0)),
                np.asarray(rcut2, dtype=np.float64),
                Lx,
                Ly,
            )
        )
        u_unwrapped.append(
            _total_energy_numba(
                np.asarray(positions_real[i], dtype=np.float64),
                types,
                eps,
                sig,
                float(record.get("C0", 0.0)),
                float(record.get("C2", 0.0)),
                float(record.get("C4", 0.0)),
                np.asarray(rcut2, dtype=np.float64),
                Lx,
                Ly,
            )
        )

    return {
        "idx": np.asarray(idx),
        "u_wrapped": np.asarray(u_wrapped, dtype=np.float64),
        "u_unwrapped": np.asarray(u_unwrapped, dtype=np.float64),
    }


# -----------------------------
# ESS computation
# -----------------------------

def _softmax_logw(logw: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D log-weights in float64."""
    logw = np.asarray(logw, dtype=np.float64)
    m = np.max(logw)
    w = np.exp(logw - m)
    w_sum = np.sum(w)
    return w / w_sum


def compute_ess_vs_beta(U: np.ndarray, beta0: float, beta_list: Iterable[float]) -> np.ndarray:
    """Compute ESS for each beta using importance reweighting."""
    U = np.asarray(U, dtype=np.float64)
    beta_list = np.asarray(list(beta_list), dtype=np.float64)
    ess = np.empty_like(beta_list, dtype=np.float64)

    for i, beta_t in enumerate(beta_list):
        # logw = log_g - log_q_theta = -(beta_t - beta0) * U
        logw = -(beta_t - beta0) * U
        w = _softmax_logw(logw)
        ess[i] = 1.0 / np.sum(w ** 2)
    return ess


# -----------------------------
# Saving utilities
# -----------------------------

def summarize_u_stats(U: np.ndarray) -> Dict[str, float]:
    U = np.asarray(U, dtype=np.float64)
    return {
        "mean": float(np.mean(U)),
        "var": float(np.var(U)),
        "min": float(np.min(U)),
        "max": float(np.max(U)),
        "p01": float(np.percentile(U, 1)),
        "p05": float(np.percentile(U, 5)),
        "p50": float(np.percentile(U, 50)),
        "p95": float(np.percentile(U, 95)),
        "p99": float(np.percentile(U, 99)),
        "n_total": int(U.shape[0]),
    }


def save_ess_outputs(
    results_dir: Path,
    beta_list: np.ndarray,
    ess_values: np.ndarray,
    u_stats: Dict[str, float],
    config: Dict[str, object],
    save_u_all: bool = False,
    U_all: np.ndarray | None = None,
) -> Dict[str, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ess_path = results_dir / "ESS_vs_beta.csv"
    data = np.column_stack([beta_list, ess_values])
    header = "beta,ESS"
    np.savetxt(ess_path, data, delimiter=",", header=header, comments="")

    stats_path = results_dir / "U_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(u_stats, fh, indent=2, sort_keys=True)

    cfg_path = results_dir / "run_config.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)

    u_path = None
    if save_u_all and U_all is not None:
        u_path = results_dir / "U_all.npy"
        np.save(u_path, np.asarray(U_all, dtype=np.float64))

    return {
        "ess_csv": ess_path,
        "stats_json": stats_path,
        "config_json": cfg_path,
        "u_all_npy": u_path,
    }
