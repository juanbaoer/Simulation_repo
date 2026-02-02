#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ParsedInfo:
    root: Path
    input_file: Optional[Path]
    log_file: Optional[Path]
    data_file: Optional[Path]
    dimension: Optional[int]
    box: Optional[Tuple[float, float, float, float, float, float]]
    n_atoms: Optional[int]
    rho: Optional[float]
    temperature: Optional[float]
    dt: Optional[float]
    thermotau: Optional[float]
    variables: Dict[str, float]
    evidence: Dict[str, str]


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def scan_files(root: Path) -> Dict[str, List[Path]]:
    files = [p for p in root.iterdir() if p.is_file()]
    inputs = [p for p in files if p.name == "in" or p.name.startswith("in.") or p.suffix == ".in"]
    logs = [p for p in files if p.name.startswith("log") and p.suffix in {".lammps", ".log", ""}]
    data = [p for p in files if p.suffix == ".data" or p.name.endswith(".data")]
    dumps = [p for p in files if p.name.startswith("dump") or p.suffix in {".dump", ".lammpstrj", ".xyz"}]
    restarts = [p for p in files if p.suffix == ".restart"]
    tables = [p for p in files if p.name.endswith(".table")]
    screens = [p for p in files if p.name.startswith("screen") or p.suffix == ".out"]
    return {
        "files": files,
        "inputs": inputs,
        "logs": logs,
        "data": data,
        "dumps": dumps,
        "restarts": restarts,
        "tables": tables,
        "screens": screens,
    }


def choose_primary_input(inputs: List[Path], logs: List[Path]) -> Optional[Path]:
    if len(inputs) == 1:
        return inputs[0]
    if not inputs:
        return None
    for log in logs:
        text = read_text_safe(log)
        m = re.search(r"-in\s+(\S+)", text)
        if not m:
            m = re.search(r"Use:\s+.*\s-in\s+(\S+)", text)
        if m:
            name = Path(m.group(1)).name
            for ip in inputs:
                if ip.name == name:
                    return ip
    return sorted(inputs, key=lambda p: p.name)[0]


def parse_variables(text: str) -> Dict[str, float]:
    vars_found: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"variable\s+(\w+)\s+equal\s+(.+)$", line)
        if not m:
            continue
        name, expr = m.group(1), m.group(2)
        expr = expr.split("#", 1)[0].strip()
        num = extract_number(expr)
        if num is not None:
            vars_found[name] = num
    return vars_found


def extract_number(expr: str) -> Optional[float]:
    expr = expr.strip()
    if re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", expr):
        try:
            return float(expr)
        except ValueError:
            return None
    return None


def parse_dimension(text: str) -> Optional[int]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("dimension"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def resolve_value(token: str, variables: Dict[str, float]) -> Optional[float]:
    token = token.strip()
    if token.startswith("${") and token.endswith("}"):
        name = token[2:-1]
        return variables.get(name)
    return extract_number(token)


def parse_box_from_data(data_file: Path) -> Optional[Tuple[float, float, float, float, float, float]]:
    text = read_text_safe(data_file)
    xlo = xhi = ylo = yhi = zlo = zhi = None
    for line in text.splitlines():
        if "xlo xhi" in line:
            parts = line.split()
            if len(parts) >= 4:
                xlo, xhi = float(parts[0]), float(parts[1])
        if "ylo yhi" in line:
            parts = line.split()
            if len(parts) >= 4:
                ylo, yhi = float(parts[0]), float(parts[1])
        if "zlo zhi" in line:
            parts = line.split()
            if len(parts) >= 4:
                zlo, zhi = float(parts[0]), float(parts[1])
        if xlo is not None and ylo is not None and zlo is not None:
            break
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        return None
    return (xlo, xhi, ylo, yhi, zlo, zhi)


def parse_natoms_from_data(data_file: Path) -> Optional[int]:
    text = read_text_safe(data_file)
    for line in text.splitlines():
        if line.strip().endswith("atoms"):
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                return int(parts[0])
    return None


def parse_box_from_input(text: str, variables: Dict[str, float]) -> Optional[Tuple[float, float, float, float, float, float]]:
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("region"):
            continue
        if " block " not in f" {line} ":
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        # region name style xlo xhi ylo yhi zlo zhi
        xlo = resolve_value(parts[3], variables)
        xhi = resolve_value(parts[4], variables)
        ylo = resolve_value(parts[5], variables)
        yhi = resolve_value(parts[6], variables)
        zlo = resolve_value(parts[7], variables)
        zhi = resolve_value(parts[8], variables) if len(parts) > 8 else None
        if None not in (xlo, xhi, ylo, yhi, zlo, zhi):
            return (xlo, xhi, ylo, yhi, zlo, zhi)
    return None


def parse_temperature(text: str, variables: Dict[str, float]) -> Optional[float]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("fix") and " nvt " in f" {line} " and " temp " in f" {line} ":
            parts = line.split()
            try:
                idx = parts.index("temp")
            except ValueError:
                continue
            if len(parts) >= idx + 3:
                tstart = resolve_value(parts[idx + 1], variables)
                tstop = resolve_value(parts[idx + 2], variables)
                if tstop is not None:
                    return tstop
                if tstart is not None:
                    return tstart
        if line.startswith("velocity") and " create " in f" {line} ":
            parts = line.split()
            if "create" in parts:
                idx = parts.index("create")
                if len(parts) > idx + 1:
                    t = resolve_value(parts[idx + 1], variables)
                    if t is not None:
                        return t
    for key in ("T", "Tinit", "temp"):
        if key in variables:
            return variables[key]
    return None


def parse_dt(text: str, variables: Dict[str, float]) -> Optional[float]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("timestep"):
            parts = line.split()
            if len(parts) >= 2:
                dt = resolve_value(parts[1], variables)
                if dt is not None:
                    return dt
    if "dt" in variables:
        return variables["dt"]
    return None


def parse_thermotau(text: str, variables: Dict[str, float]) -> Optional[float]:
    for name in ("ThermoTau", "thermo_tau", "Tdamp", "tdamp", "tauT", "damp"):
        if name in variables:
            return variables[name]
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("fix") and " nvt " in f" {line} " and " temp " in f" {line} ":
            parts = line.split()
            try:
                idx = parts.index("temp")
            except ValueError:
                continue
            if len(parts) >= idx + 4:
                tdamp = resolve_value(parts[idx + 3], variables)
                if tdamp is not None:
                    return tdamp
    return None


def parse_natoms_from_input(text: str, variables: Dict[str, float]) -> Optional[int]:
    n = 0
    has = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("variable") and " equal " in line:
            m = re.match(r"variable\s+N(\d+)\s+equal\s+(.+)$", line)
            if m:
                val = extract_number(m.group(2).split("#", 1)[0].strip())
                if val is not None:
                    n += int(val)
                    has = True
    if has:
        return n
    return None


def compute_density(n_atoms: int, box: Tuple[float, float, float, float, float, float], dimension: int) -> float:
    xlo, xhi, ylo, yhi, zlo, zhi = box
    if dimension == 2:
        area = (xhi - xlo) * (yhi - ylo)
        return n_atoms / area
    volume = (xhi - xlo) * (yhi - ylo) * (zhi - zlo)
    return n_atoms / volume


def format_decimal(value: float, max_decimals: int = 4) -> str:
    s = f"{value:.{max_decimals}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def format_decimal_folder(value: float, max_decimals: int = 4) -> str:
    return format_decimal(value, max_decimals).replace(".", "p")


def format_sci(value: float) -> str:
    if value == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(value))))
    mant = value / (10 ** exp)
    mant_str = f"{mant:.6g}".rstrip("0").rstrip(".")
    if mant_str == "1":
        mant_str = "1"
    exp_str = str(exp)
    return f"{mant_str}e{exp_str}"


def format_run_value(value: float, force_sci: bool = False) -> str:
    if value == 0:
        return "0"
    if force_sci or abs(value) < 0.01 or abs(value) >= 1000:
        s = format_sci(value)
    else:
        s = format_decimal(value, 6)
    return s.replace(".", "p")


def parse_info(root: Path) -> ParsedInfo:
    inv = scan_files(root)
    input_file = choose_primary_input(inv["inputs"], inv["logs"])
    log_file = inv["logs"][0] if inv["logs"] else None
    data_file = inv["data"][0] if inv["data"] else None

    text = read_text_safe(input_file) if input_file else ""
    variables = parse_variables(text)
    dimension = parse_dimension(text)
    box = parse_box_from_data(data_file) if data_file else None
    if box is None:
        box = parse_box_from_input(text, variables)
    n_atoms = None
    evidence = {}
    if data_file:
        n_atoms = parse_natoms_from_data(data_file)
        if n_atoms is not None:
            evidence["N"] = f"from data file header: {data_file.name}"
    if n_atoms is None:
        n_atoms = parse_natoms_from_input(text, variables)
        if n_atoms is not None:
            evidence["N"] = "sum of N1/N2/N3 variables"
    temperature = parse_temperature(text, variables)
    if temperature is not None:
        evidence["T"] = "from fix nvt/velocity create in input"
    dt = parse_dt(text, variables)
    if dt is not None:
        evidence["dt"] = "from timestep or dt variable in input"
    thermotau = parse_thermotau(text, variables)
    if thermotau is not None:
        evidence["ThermoTau"] = "from Tdamp (thermostat damping) in input"
    rho = None
    if n_atoms is not None and box is not None and dimension is not None:
        rho = compute_density(n_atoms, box, dimension)
        evidence["rho"] = "computed from N and box bounds"
    if dimension is not None:
        evidence["dimension"] = "from input script"
    if box is not None:
        evidence["box"] = "from data file or region block"
    return ParsedInfo(
        root=root,
        input_file=input_file,
        log_file=log_file,
        data_file=data_file,
        dimension=dimension,
        box=box,
        n_atoms=n_atoms,
        rho=rho,
        temperature=temperature,
        dt=dt,
        thermotau=thermotau,
        variables=variables,
        evidence=evidence,
    )


def parse_info_at(root: Path, run_dir: Path) -> ParsedInfo:
    inv = scan_files(run_dir)
    input_file = choose_primary_input(inv["inputs"], inv["logs"])
    log_file = inv["logs"][0] if inv["logs"] else None
    data_file = inv["data"][0] if inv["data"] else None

    text = read_text_safe(input_file) if input_file else ""
    variables = parse_variables(text)
    dimension = parse_dimension(text)
    box = parse_box_from_data(data_file) if data_file else None
    if box is None:
        box = parse_box_from_input(text, variables)
    n_atoms = None
    evidence = {}
    if data_file:
        n_atoms = parse_natoms_from_data(data_file)
        if n_atoms is not None:
            evidence["N"] = f"from data file header: {data_file.name}"
    if n_atoms is None:
        n_atoms = parse_natoms_from_input(text, variables)
        if n_atoms is not None:
            evidence["N"] = "sum of N1/N2/N3 variables"
    temperature = parse_temperature(text, variables)
    if temperature is not None:
        evidence["T"] = "from fix nvt/velocity create in input"
    dt = parse_dt(text, variables)
    if dt is not None:
        evidence["dt"] = "from timestep or dt variable in input"
    thermotau = parse_thermotau(text, variables)
    if thermotau is not None:
        evidence["ThermoTau"] = "from Tdamp (thermostat damping) in input"
    rho = None
    if n_atoms is not None and box is not None and dimension is not None:
        rho = compute_density(n_atoms, box, dimension)
        evidence["rho"] = "computed from N and box bounds"
    if dimension is not None:
        evidence["dimension"] = "from input script"
    if box is not None:
        evidence["box"] = "from data file or region block"
    return ParsedInfo(
        root=root,
        input_file=input_file,
        log_file=log_file,
        data_file=data_file,
        dimension=dimension,
        box=box,
        n_atoms=n_atoms,
        rho=rho,
        temperature=temperature,
        dt=dt,
        thermotau=thermotau,
        variables=variables,
        evidence=evidence,
    )


def build_paths(info: ParsedInfo) -> Tuple[Path, Path, Path, str, str]:
    if info.n_atoms is None or info.rho is None or info.temperature is None or info.dt is None or info.thermotau is None:
        raise ValueError("Missing required parameters to construct folder names.")
    rho_str = format_decimal_folder(info.rho, 4)
    t_str = format_decimal_folder(info.temperature, 4)
    exp_name = f"N{info.n_atoms}_rho{rho_str}_T{t_str}"
    dt_str = format_run_value(info.dt)
    tau_str = format_run_value(info.thermotau, force_sci=True)
    run_name = f"dt{dt_str}_ThermoTau{tau_str}"
    exp_dir = info.root / exp_name
    run_dir = exp_dir / run_name
    config_dir = info.root / "config"
    return config_dir, exp_dir, run_dir, exp_name, run_name


def build_move_plan(info: ParsedInfo, config_dir: Path, exp_dir: Path, run_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    inv = scan_files(info.root)
    move_plan: List[Tuple[Path, Path]] = []
    notes: List[str] = []
    shared = {info.root / "ka2d.table", info.root / "make_ka2d_table.py"}
    for p in shared:
        if p.exists():
            move_plan.append((p, config_dir / p.name))
    skip_names = {"parse_summary.json", "parse_summary.md", "report.md", "reorg_log.txt"}
    for p in inv["files"]:
        if p.name in skip_names:
            continue
        if p in shared:
            continue
        if p.parent.name == "tools":
            continue
        if p.name == "config":
            continue
        move_plan.append((p, run_dir / p.name))
    notes.append("Shared resources moved to config/: ka2d.table, make_ka2d_table.py")
    return move_plan, notes


def print_dry_run(move_plan: List[Tuple[Path, Path]]) -> None:
    print("DRY-RUN PLAN (source -> destination):")
    for src, dst in move_plan:
        print(f"  {src} -> {dst}")


def write_parse_summary(info: ParsedInfo, config_dir: Path, exp_dir: Path, run_dir: Path) -> None:
    summary = {
        "root": str(info.root),
        "input_file": str(info.input_file) if info.input_file else None,
        "log_file": str(info.log_file) if info.log_file else None,
        "data_file": str(info.data_file) if info.data_file else None,
        "dimension": info.dimension,
        "box": info.box,
        "N": info.n_atoms,
        "rho": info.rho,
        "T": info.temperature,
        "dt": info.dt,
        "ThermoTau": info.thermotau,
        "variables": info.variables,
        "targets": {
            "config_dir": str(config_dir),
            "experiment_dir": str(exp_dir),
            "run_dir": str(run_dir),
        },
        "evidence": info.evidence,
    }
    (info.root / "parse_summary.json").write_text(json.dumps(summary, indent=2))
    md_lines = [
        "# Parse Summary",
        "",
        f"- root: `{info.root}`",
        f"- input_file: `{info.input_file}`" if info.input_file else "- input_file: None",
        f"- log_file: `{info.log_file}`" if info.log_file else "- log_file: None",
        f"- data_file: `{info.data_file}`" if info.data_file else "- data_file: None",
        f"- dimension: {info.dimension}",
        f"- box: {info.box}",
        f"- N: {info.n_atoms}",
        f"- rho: {info.rho}",
        f"- T: {info.temperature}",
        f"- dt: {info.dt}",
        f"- ThermoTau: {info.thermotau}",
        "",
        "## Evidence",
    ]
    for k, v in info.evidence.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")
    md_lines.append("## Targets")
    md_lines.append(f"- config_dir: `{config_dir}`")
    md_lines.append(f"- experiment_dir: `{exp_dir}`")
    md_lines.append(f"- run_dir: `{run_dir}`")
    (info.root / "parse_summary.md").write_text("\n".join(md_lines))


def update_input_paths(input_path: Path, config_dir: Path, run_dir: Path) -> None:
    if not input_path.exists():
        return
    text = input_path.read_text()
    rel = Path("../../config/ka2d.table")
    if "ka2d.table" in text and "config/ka2d.table" not in text:
        text = text.replace("ka2d.table", str(rel))
        input_path.write_text(text)


def execute_moves(move_plan: List[Tuple[Path, Path]], reorg_log: Path) -> None:
    lines: List[str] = []
    for src, dst in move_plan:
        if not src.exists():
            lines.append(f"SKIP (missing): {src} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            lines.append(f"SKIP (exists): {src} -> {dst}")
            continue
        shutil.move(str(src), str(dst))
        lines.append(f"MOVE: {src} -> {dst}")
    reorg_log.write_text("\n".join(lines))


def write_annotated_inputs(input_files: List[Path], out_path: Path) -> None:
    lines: List[str] = []
    for in_file in input_files:
        lines.append(f"===== FILE: {in_file.name} =====")
        raw = in_file.read_text().splitlines()
        for idx, line in enumerate(raw, start=1):
            lines.append(f"{idx:04d}  {line}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def _find_block(lines: List[str], start_pat: str, end_pat: Optional[str] = None) -> Tuple[int, int]:
    start = None
    for i, line in enumerate(lines):
        if re.search(start_pat, line):
            start = i
            break
    if start is None:
        return -1, -1
    if end_pat is None:
        end = start
        while end + 1 < len(lines) and lines[end + 1].strip() != "":
            end += 1
        return start, end
    end = start
    for j in range(start, len(lines)):
        if re.search(end_pat, lines[j]):
            end = j
    return start, end


def _excerpt(lines: List[str], start: int, end: int) -> List[str]:
    if start < 0 or end < 0:
        return []
    return lines[start : end + 1]


def generate_experiment_documentation(info: ParsedInfo, exp_dir: Path, run_dir: Path) -> None:
    doc = exp_dir / "experiment_documentation.md"
    inputs = [p for p in run_dir.iterdir() if p.name == "in" or p.name.startswith("in.") or p.suffix == ".in"]
    inputs = sorted(inputs, key=lambda p: p.name)
    if not inputs:
        doc.write_text("# Experiment Documentation\n\nNo input files found.\n")
        return
    input_file = inputs[0]
    input_lines = input_file.read_text().splitlines()

    annotated = exp_dir / "in_annotated.txt"
    write_annotated_inputs(inputs, annotated)

    sections = [
        {
            "name": "Initialization (units, dimension, atom style, boundary)",
            "start": r"^units\s+",
            "end": r"^boundary\s+",
            "explain": [
                "Sets Lennard-Jones reduced units, 2D dynamics, atomic style, and periodic boundaries.",
                "This defines the fundamental units and dimensionality for all subsequent commands.",
                "Affects all downstream thermo outputs and integration behavior; the 2D setting is crucial for density and area-based normalization.",
            ],
        },
        {
            "name": "User Variables (seed, system size, temperature, damping, timestep)",
            "start": r"^#\s+-+\s+user variables",
            "end": r"^variable\s+dt\s+",
            "explain": [
                "Defines run-time variables: random seed, run id, box length L, initial temperature Tinit, thermostat damping Tdamp, and timestep dt.",
                "These variables are referenced later (e.g., timestep uses dt, thermostat uses Tdamp, and outputs embed runid).",
                "Hidden couplings: dt controls the run length via Neq and Nsamp; Tdamp becomes the ThermoTau in folder naming.",
            ],
        },
        {
            "name": "Run Lengths (equilibration, dump cadence, sampling)",
            "start": r"^#\s+-+\s+run lengths",
            "end": r"^variable\s+Nsamp",
            "explain": [
                "Defines equilibration time in tau (tau_eq), converts it to steps (Neq), and sets dump frequency and number of frames.",
                "Sampling length is derived from nFrames and dumpEvery, tying output size directly to these variables.",
                "Outputs affected: thermo cadence via thermo command and all trajectory dumps.",
            ],
        },
        {
            "name": "Composition (species counts)",
            "start": r"^#\s+composition",
            "end": r"^variable\s+N3\s+",
            "explain": [
                "Specifies the number of particles per type (N1, N2, N3).",
                "The total N is the sum, which determines density together with the box size.",
                "Affects structure initialization and later per-type analysis.",
            ],
        },
        {
            "name": "Simulation Box",
            "start": r"^#\s+-+\s+box",
            "end": r"^create_box\s+",
            "explain": [
                "Defines a 2D box of size L x L with a thin z extent for 2D enforcement.",
                "The region sets the spatial domain; create_box initializes a 3-type system.",
                "This box is used for density (rho = N / A) and output volume/area calculations.",
            ],
        },
        {
            "name": "Create Atoms and Masses",
            "start": r"^#\s+-+\s+create atoms",
            "end": r"^mass\s+",
            "explain": [
                "Randomly places atoms for each type with overlap constraints, then assigns equal mass.",
                "Ensures an initial disordered configuration and consistent mass for dynamics.",
                "This setup affects the initial data snapshot and the initial energy in the log.",
            ],
        },
        {
            "name": "2D Constraint (set z=0)",
            "start": r"^#\s+strictly set all z",
            "end": r"^set\s+group\s+all\s+z",
            "explain": [
                "Forces all atoms to z=0 to enforce strictly 2D positions.",
                "This is essential for interpreting density as area density and for 2D MSD.",
                "Downstream affects trajectories (z coordinate fixed) and analysis dimensionality.",
            ],
        },
        {
            "name": "Pair Potential (tabulated LJ)",
            "start": r"^#\s+-+\s+pair potential",
            "end": r"^pair_coeff\s+3\s+3",
            "explain": [
                "Uses a tabulated pair potential and assigns coefficients for all type pairs from ka2d.table.",
                "Defines all interparticle forces and thus potential energy (pe) in thermo output.",
                "Couples to the shared config/ka2d.table file (relative path from run dir).",
            ],
        },
        {
            "name": "Neighbor List",
            "start": r"^#\s+neighbor list",
            "end": r"^neigh_modify\s+",
            "explain": [
                "Configures neighbor binning and update frequency for efficient force computation.",
                "Stability and performance depend on these settings; they influence force accuracy.",
                "No direct output, but affects all computed energies and dynamics.",
            ],
        },
        {
            "name": "Energy Minimization",
            "start": r"^#\s+-+\s+remove bad contacts",
            "end": r"^minimize\s+",
            "explain": [
                "Runs a conjugate-gradient minimization to remove initial overlaps.",
                "This reduces extreme forces from random placement before dynamics start.",
                "Affects initial energies written to the log and the initial state dumped by write_data.",
            ],
        },
        {
            "name": "Dynamics Setup (timestep and velocity)",
            "start": r"^#\s+-+\s+dynamics",
            "end": r"^velocity\s+",
            "explain": [
                "Sets the timestep using variable dt and initializes velocities at Tinit.",
                "The timestep controls physical time mapping for thermo and MSD.",
                "Velocity initialization affects the initial kinetic energy and temperature in the log.",
            ],
        },
        {
            "name": "Thermostat and 2D Fix",
            "start": r"^#\s+Nose-Hoover NVT",
            "end": r"^fix\s+p2d\s+",
            "explain": [
                "Applies Nose-Hoover NVT thermostat with damping Tdamp and enforces 2D motion.",
                "Tdamp is treated as ThermoTau (thermostat timescale) in run naming.",
                "Thermostat affects temperature control, which appears in thermo output.",
            ],
        },
        {
            "name": "Thermo Output",
            "start": r"^thermo\s+",
            "end": r"^thermo_style\s+",
            "explain": [
                "Sets the thermo output frequency and which properties are printed.",
                "The log file fields (step, temp, pe, ke, etotal, press) drive the Energy notebook.",
                "The presence of pe (not pe/atom) means the output is total potential energy per snapshot.",
            ],
        },
        {
            "name": "Initial Output Snapshots",
            "start": r"^#\s+Save configs",
            "end": r"^write_restart\s+T0p5_init_\$\{runid\}",
            "explain": [
                "Writes initial data and restart files before equilibration.",
                "These outputs allow exact reproduction or restart from the initial state.",
                "Outputs: T0p5_init_*.data and T0p5_init_*.restart.",
            ],
        },
        {
            "name": "Equilibration Run",
            "start": r"^#\s+1\)\s+Equilibration",
            "end": r"^write_restart\s+T0p5_equil_\$\{runid\}",
            "explain": [
                "Runs the equilibration segment for Neq steps and writes a restart.",
                "The print line documents derived values (tau_eq, dt, Neq).",
                "Outputs: T0p5_equil_*.restart and thermo entries during equilibration.",
            ],
        },
        {
            "name": "Sampling Setup (reset time and per-atom radius)",
            "start": r"^#\s+2\)\s+Sampling",
            "end": r"^variable\s+rad\s+atom",
            "explain": [
                "Resets the timestep to 0 for production and defines per-atom radius used in dumps.",
                "This ensures consistent time origin for analysis and includes radius in outputs.",
                "The rad variable appears as v_rad in dump files.",
            ],
        },
        {
            "name": "Trajectory Dumps (wrapped and unwrapped)",
            "start": r"^#\s+wrapped",
            "end": r"^dump_modify\s+du\s+colname",
            "explain": [
                "Defines wrapped and unwrapped dumps with per-atom IDs, types, positions, and radius.",
                "Unwrapped positions (xu, yu, zu) are preferred for MSD to avoid PBC artifacts.",
                "Outputs: traj_wrap_*.dump and traj_unwrap_*.dump.",
            ],
        },
        {
            "name": "Production Run and Final Restart",
            "start": r"^print\s+\"Sampling:",
            "end": r"^write_restart\s+T0p5_sample_\$\{runid\}",
            "explain": [
                "Runs Nsamp steps with the configured dump cadence and writes final restart.",
                "The print line records sampling length and frame count for traceability.",
                "Outputs: traj_*.dump, T0p5_sample_*.restart, and thermo logs.",
            ],
        },
    ]

    outputs = sorted([p.name for p in run_dir.iterdir() if p.is_file()])

    lines: List[str] = [
        "# Experiment Documentation (Guided Walkthrough)",
        "",
        "This walkthrough follows the exact execution order of the LAMMPS input file(s).",
        "",
        "## Parameter Summary (from parsing)",
        f"- N: {info.n_atoms}",
        f"- rho: {info.rho}",
        f"- T: {info.temperature}",
        f"- dt: {info.dt}",
        f"- ThermoTau: {info.thermotau} (thermostat damping time)",
        f"- dimension: {info.dimension}",
        f"- box: {info.box}",
        "",
    ]

    in_map_rows: List[Tuple[str, str, str, str]] = []

    for sec in sections:
        start, end = _find_block(input_lines, sec["start"], sec.get("end"))
        excerpt = _excerpt(input_lines, start, end)
        if not excerpt:
            continue
        lines.append(f"## {sec['name']}")
        lines.append("")
        lines.append(f"**Source:** `{input_file.name}`")
        lines.append("```lammps")
        lines.extend(excerpt)
        lines.append("```")
        lines.append("")
        for sent in sec["explain"]:
            lines.append(f"- {sent}")
        lines.append("")
        key_cmds = ", ".join({ex.split()[0] for ex in excerpt if ex.strip() and not ex.strip().startswith('#')})
        in_map_rows.append((sec["name"], input_file.name, f"{start+1}-{end+1}", key_cmds))

    lines.append("## Output File Glossary")
    for name in outputs:
        if name.endswith(".dump") or name.endswith(".lammpstrj"):
            desc = "Trajectory dump (positions/velocities for analysis)."
        elif name.endswith(".restart"):
            desc = "LAMMPS restart file (binary state for continuation)."
        elif name.endswith(".data"):
            desc = "LAMMPS data snapshot (structure/coordinates)."
        elif name.startswith("log"):
            desc = "LAMMPS log with thermo output."
        elif name.startswith("screen") or name.endswith(".out"):
            desc = "LAMMPS screen/stdout capture."
        elif name.startswith("in.") or name == "in":
            desc = "LAMMPS input script."
        else:
            desc = "Output or auxiliary file."
        lines.append(f"- `{name}`: {desc}")

    lines.append("")
    lines.append("## In-file Map")
    lines.append("")
    lines.append("Discovered input files (execution order):")
    for inp in inputs:
        lines.append(f"- `{inp.name}`: primary input script")
    lines.append("")
    lines.append("| Section | File | Line Range | Key Commands |")
    lines.append("| --- | --- | --- | --- |")
    for sec_name, fname, lrange, keycmds in in_map_rows:
        lines.append(f"| {sec_name} | {fname} | {lrange} | {keycmds} |")

    doc.write_text("\n".join(lines))


def write_report(info: ParsedInfo, config_dir: Path, exp_dir: Path, run_dir: Path, move_plan: List[Tuple[Path, Path]], notes: List[str]) -> None:
    lines = [
        "# Reorganization Report",
        "",
        "## Decisions",
        f"- Primary input file: `{info.input_file}`",
        f"- N from: {info.evidence.get('N','unknown')}",
        f"- rho from: {info.evidence.get('rho','unknown')}",
        f"- T from: {info.evidence.get('T','unknown')}",
        f"- dt from: {info.evidence.get('dt','unknown')}",
        f"- ThermoTau from: {info.evidence.get('ThermoTau','unknown')}",
        "",
        "## Folder Naming",
        f"- experiment folder: `{exp_dir}`",
        f"- run folder: `{run_dir}`",
        "",
        "## Notes",
        *[f"- {n}" for n in notes],
        "",
        "## Move Plan (executed)",
    ]
    for src, dst in move_plan:
        lines.append(f"- {src} -> {dst}")
    (info.root / "report.md").write_text("\n".join(lines))


def already_organized(root: Path) -> bool:
    if not (root / "config").exists():
        return False
    if any(p.name.startswith("N") and "_rho" in p.name and "_T" in p.name for p in root.iterdir() if p.is_dir()):
        # If no loose files other than summaries/tools, treat as organized
        remaining = [p for p in root.iterdir() if p.is_file() and p.name not in {"parse_summary.json", "parse_summary.md", "report.md", "reorg_log.txt"}]
        return len(remaining) == 0
    return False


def find_existing_run(root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    exp_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("N") and "_rho" in p.name and "_T" in p.name])
    if not exp_dirs:
        return None, None
    exp_dir = exp_dirs[0]
    run_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("dt") and "ThermoTau" in p.name])
    if not run_dirs:
        return exp_dir, None
    return exp_dir, run_dirs[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reorganize KA2D ternary LAMMPS experiment folder.")
    parser.add_argument("--root", default=".", help="Root experiment folder (default: .)")
    parser.add_argument("--execute", action="store_true", help="Execute the move plan (default is dry-run only).")
    parser.add_argument("--docs", action="store_true", help="Generate experiment_documentation.md after reorg.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    exp_dir = None
    run_dir = None
    if already_organized(root):
        exp_dir, run_dir = find_existing_run(root)
        if exp_dir and run_dir:
            info = parse_info_at(root, run_dir)
            config_dir = root / "config"
            move_plan = []
            notes = ["Existing organized structure detected; no moves performed."]
            write_parse_summary(info, config_dir, exp_dir, run_dir)
            print_dry_run(move_plan)
            if not args.execute:
                return 0
            if args.docs:
                generate_experiment_documentation(info, exp_dir, run_dir)
            write_report(info, config_dir, exp_dir, run_dir, move_plan, notes)
            return 0

    info = parse_info(root)
    config_dir, exp_dir, run_dir, _, _ = build_paths(info)
    move_plan, notes = build_move_plan(info, config_dir, exp_dir, run_dir)
    write_parse_summary(info, config_dir, exp_dir, run_dir)
    print_dry_run(move_plan)

    if not args.execute:
        return 0

    config_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    execute_moves(move_plan, root / "reorg_log.txt")

    # Update input paths after move
    moved_input = run_dir / info.input_file.name if info.input_file else None
    if moved_input and moved_input.exists():
        update_input_paths(moved_input, config_dir, run_dir)

    if args.docs:
        generate_experiment_documentation(info, exp_dir, run_dir)

    write_report(info, config_dir, exp_dir, run_dir, move_plan, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
