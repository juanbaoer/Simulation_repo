# Experiment Documentation (Guided Walkthrough)

This walkthrough follows the exact execution order of the LAMMPS input file(s).

## Parameter Summary (from parsing)
- N: 43
- rho: 1.1944444444444444
- T: 0.5
- dt: 0.005
- ThermoTau: 1.0 (thermostat damping time)
- dimension: 2
- box: (0.0, 6.0, 0.0, 6.0, -0.5, 0.5)

## Initialization (units, dimension, atom style, boundary)

**Source:** `in.ka2d_T0p5_equil`
```lammps
units           lj
dimension       2
atom_style      atomic
boundary        p p p
```

- Sets Lennard-Jones reduced units, 2D dynamics, atomic style, and periodic boundaries.
- This defines the fundamental units and dimensionality for all subsequent commands.
- Affects all downstream thermo outputs and integration behavior; the 2D setting is crucial for density and area-based normalization.

## User Variables (seed, system size, temperature, damping, timestep)

**Source:** `in.ka2d_T0p5_equil`
```lammps
# ---------- user variables ----------
variable        seed  index 12345
variable        runid index 000
variable        L     equal 6.0
variable        Tinit equal 0.5
variable        Tdamp equal 1.0
variable        dt    equal 0.005
```

- Defines run-time variables: random seed, run id, box length L, initial temperature Tinit, thermostat damping Tdamp, and timestep dt.
- These variables are referenced later (e.g., timestep uses dt, thermostat uses Tdamp, and outputs embed runid).
- Hidden couplings: dt controls the run length via Neq and Nsamp; Tdamp becomes the ThermoTau in folder naming.

## Run Lengths (equilibration, dump cadence, sampling)

**Source:** `in.ka2d_T0p5_equil`
```lammps
# ---- run lengths ----
variable        tau_eq    equal 1.0e5
variable        Neq       equal ceil(v_tau_eq/v_dt)   # 1e5 tau -> 2e7 steps
variable        dumpEvery equal 200                    # 200 steps = 1 tau
variable        nFrames   equal 10000
variable        Nsamp     equal v_nFrames*v_dumpEvery  # 1e4 frames -> 2e6 steps
```

- Defines equilibration time in tau (tau_eq), converts it to steps (Neq), and sets dump frequency and number of frames.
- Sampling length is derived from nFrames and dumpEvery, tying output size directly to these variables.
- Outputs affected: thermo cadence via thermo command and all trajectory dumps.

## Composition (species counts)

**Source:** `in.ka2d_T0p5_equil`
```lammps
# composition (5:3:3 for N=43 -> 19:12:12)
variable        N1 equal 19
variable        N2 equal 12
variable        N3 equal 12
```

- Specifies the number of particles per type (N1, N2, N3).
- The total N is the sum, which determines density together with the box size.
- Affects structure initialization and later per-type analysis.

## Simulation Box

**Source:** `in.ka2d_T0p5_equil`
```lammps
# ---------- box ----------
region          box block 0 ${L} 0 ${L} -0.5 0.5
create_box      3 box
```

- Defines a 2D box of size L x L with a thin z extent for 2D enforcement.
- The region sets the spatial domain; create_box initializes a 3-type system.
- This box is used for density (rho = N / A) and output volume/area calculations.

## Create Atoms and Masses

**Source:** `in.ka2d_T0p5_equil`
```lammps
# ---------- create atoms (random, avoid close overlaps) ----------
create_atoms    1 random ${N1} ${seed}        box overlap 0.70
create_atoms    2 random ${N2} ${seed}+11111  box overlap 0.70
create_atoms    3 random ${N3} ${seed}+22222  box overlap 0.70

mass            * 1.0
```

- Randomly places atoms for each type with overlap constraints, then assigns equal mass.
- Ensures an initial disordered configuration and consistent mass for dynamics.
- This setup affects the initial data snapshot and the initial energy in the log.

## 2D Constraint (set z=0)

**Source:** `in.ka2d_T0p5_equil`
```lammps
# strictly set all z = 0
set             group all z 0.0
```

- Forces all atoms to z=0 to enforce strictly 2D positions.
- This is essential for interpreting density as area density and for 2D MSD.
- Downstream affects trajectories (z coordinate fixed) and analysis dimensionality.

## Pair Potential (tabulated LJ)

**Source:** `in.ka2d_T0p5_equil`
```lammps
# ---------- pair potential (tabulated modified LJ) ----------
pair_style      table spline 10000
pair_coeff      1 1 ../../config/ka2d.table AA
pair_coeff      1 2 ../../config/ka2d.table AB
pair_coeff      1 3 ../../config/ka2d.table AC
pair_coeff      2 2 ../../config/ka2d.table BB
pair_coeff      2 3 ../../config/ka2d.table BC
pair_coeff      3 3 ../../config/ka2d.table CC
```

- Uses a tabulated pair potential and assigns coefficients for all type pairs from ka2d.table.
- Defines all interparticle forces and thus potential energy (pe) in thermo output.
- Couples to the shared config/ka2d.table file (relative path from run dir).

## Neighbor List

**Source:** `in.ka2d_T0p5_equil`
```lammps
# neighbor list (safe)
neighbor        0.4 bin
neigh_modify    delay 0 every 1 check yes
```

- Configures neighbor binning and update frequency for efficient force computation.
- Stability and performance depend on these settings; they influence force accuracy.
- No direct output, but affects all computed energies and dynamics.

## Energy Minimization

**Source:** `in.ka2d_T0p5_equil`
```lammps
# ---------- remove bad contacts (recommended for random init) ----------
min_style       cg
minimize        1e-10 1e-10 10000 100000
```

- Runs a conjugate-gradient minimization to remove initial overlaps.
- This reduces extreme forces from random placement before dynamics start.
- Affects initial energies written to the log and the initial state dumped by write_data.

## Dynamics Setup (timestep and velocity)

**Source:** `in.ka2d_T0p5_equil`
```lammps
# ---------- dynamics ----------
timestep        ${dt}
velocity        all create ${Tinit} ${seed}+33333 mom yes rot no dist gaussian
```

- Sets the timestep using variable dt and initializes velocities at Tinit.
- The timestep controls physical time mapping for thermo and MSD.
- Velocity initialization affects the initial kinetic energy and temperature in the log.

## Thermostat and 2D Fix

**Source:** `in.ka2d_T0p5_equil`
```lammps
# Nose-Hoover NVT
fix             nvt all nvt temp ${Tinit} ${Tinit} ${Tdamp}

# IMPORTANT for 2D: enforce2d should be last fix in script
fix             p2d all enforce2d
```

- Applies Nose-Hoover NVT thermostat with damping Tdamp and enforces 2D motion.
- Tdamp is treated as ThermoTau (thermostat timescale) in run naming.
- Thermostat affects temperature control, which appears in thermo output.

## Thermo Output

**Source:** `in.ka2d_T0p5_equil`
```lammps
thermo          200
thermo_style    custom step temp pe ke etotal press
```

- Sets the thermo output frequency and which properties are printed.
- The log file fields (step, temp, pe, ke, etotal, press) drive the Energy notebook.
- The presence of pe (not pe/atom) means the output is total potential energy per snapshot.

## Initial Output Snapshots

**Source:** `in.ka2d_T0p5_equil`
```lammps
# Save configs for debugging/reproducibility (initial)
write_data      T0p5_init_${runid}.data
write_restart   T0p5_init_${runid}.restart
```

- Writes initial data and restart files before equilibration.
- These outputs allow exact reproduction or restart from the initial state.
- Outputs: T0p5_init_*.data and T0p5_init_*.restart.

## Equilibration Run

**Source:** `in.ka2d_T0p5_equil`
```lammps
# 1) Equilibration: 1e5 tau
# =========================
print "Equilibration: tau_eq=${tau_eq}, dt=${dt} => Neq=${Neq} steps"
run             ${Neq}
write_restart   T0p5_equil_${runid}.restart
```

- Runs the equilibration segment for Neq steps and writes a restart.
- The print line documents derived values (tau_eq, dt, Neq).
- Outputs: T0p5_equil_*.restart and thermo entries during equilibration.

## Sampling Setup (reset time and per-atom radius)

**Source:** `in.ka2d_T0p5_equil`
```lammps
# 2) Sampling: 1e4 frames, dump every 200 steps
# =========================
reset_timestep  0

# radii_by_type = (0.500, 0.44, 0.47) for type 1/2/3
variable        rad atom "(type==1)*0.500 + (type==2)*0.440 + (type==3)*0.470"
```

- Resets the timestep to 0 for production and defines per-atom radius used in dumps.
- This ensures consistent time origin for analysis and includes radius in outputs.
- The rad variable appears as v_rad in dump files.

## Trajectory Dumps (wrapped and unwrapped)

**Source:** `in.ka2d_T0p5_equil`
```lammps
# wrapped
dump            dw all custom ${dumpEvery} traj_wrap_${runid}.dump id type x y z v_rad
dump_modify     dw sort id
dump_modify     dw pbc yes
dump_modify     dw colname v_rad radius

# unwrapped
dump            du all custom ${dumpEvery} traj_unwrap_${runid}.dump id type xu yu zu ix iy iz v_rad
dump_modify     du sort id
dump_modify     du colname v_rad radius
```

- Defines wrapped and unwrapped dumps with per-atom IDs, types, positions, and radius.
- Unwrapped positions (xu, yu, zu) are preferred for MSD to avoid PBC artifacts.
- Outputs: traj_wrap_*.dump and traj_unwrap_*.dump.

## Production Run and Final Restart

**Source:** `in.ka2d_T0p5_equil`
```lammps
print "Sampling: Nsamp=${Nsamp} steps, dumpEvery=${dumpEvery} => frames=${nFrames}"
run             ${Nsamp}

write_restart   T0p5_sample_${runid}.restart
```

- Runs Nsamp steps with the configured dump cadence and writes final restart.
- The print line records sampling length and frame count for traceability.
- Outputs: traj_*.dump, T0p5_sample_*.restart, and thermo logs.

## Output File Glossary
- `T0p5_equil_000.restart`: LAMMPS restart file (binary state for continuation).
- `T0p5_init_000.data`: LAMMPS data snapshot (structure/coordinates).
- `T0p5_init_000.restart`: LAMMPS restart file (binary state for continuation).
- `T0p5_sample_000.restart`: LAMMPS restart file (binary state for continuation).
- `in.ka2d_T0p5_equil`: LAMMPS input script.
- `log_000.lammps`: LAMMPS log with thermo output.
- `screen_000.out`: LAMMPS screen/stdout capture.
- `traj_unwrap_000.dump`: Trajectory dump (positions/velocities for analysis).
- `traj_wrap_000.dump`: Trajectory dump (positions/velocities for analysis).

## In-file Map

Discovered input files (execution order):
- `in.ka2d_T0p5_equil`: primary input script

| Section | File | Line Range | Key Commands |
| --- | --- | --- | --- |
| Initialization (units, dimension, atom style, boundary) | in.ka2d_T0p5_equil | 4-7 | boundary, units, dimension, atom_style |
| User Variables (seed, system size, temperature, damping, timestep) | in.ka2d_T0p5_equil | 9-15 | variable |
| Run Lengths (equilibration, dump cadence, sampling) | in.ka2d_T0p5_equil | 17-22 | variable |
| Composition (species counts) | in.ka2d_T0p5_equil | 24-27 | variable |
| Simulation Box | in.ka2d_T0p5_equil | 29-31 | region, create_box |
| Create Atoms and Masses | in.ka2d_T0p5_equil | 33-38 | mass, create_atoms |
| 2D Constraint (set z=0) | in.ka2d_T0p5_equil | 40-41 | set |
| Pair Potential (tabulated LJ) | in.ka2d_T0p5_equil | 43-50 | pair_coeff, pair_style |
| Neighbor List | in.ka2d_T0p5_equil | 52-54 | neighbor, neigh_modify |
| Energy Minimization | in.ka2d_T0p5_equil | 56-58 | min_style, minimize |
| Dynamics Setup (timestep and velocity) | in.ka2d_T0p5_equil | 60-62 | velocity, timestep |
| Thermostat and 2D Fix | in.ka2d_T0p5_equil | 64-68 | fix |
| Thermo Output | in.ka2d_T0p5_equil | 70-71 | thermo, thermo_style |
| Initial Output Snapshots | in.ka2d_T0p5_equil | 73-75 | write_data, write_restart |
| Equilibration Run | in.ka2d_T0p5_equil | 78-82 | print, write_restart, run |
| Sampling Setup (reset time and per-atom radius) | in.ka2d_T0p5_equil | 85-90 | reset_timestep, variable |
| Trajectory Dumps (wrapped and unwrapped) | in.ka2d_T0p5_equil | 92-101 | dump, dump_modify |
| Production Run and Final Restart | in.ka2d_T0p5_equil | 103-106 | print, write_restart, run |