# Parse Summary

- root: `/Users/xuhengyuan/Downloads/Notes/Simulation_repo/Lammps/Ternary`
- input_file: `/Users/xuhengyuan/Downloads/Notes/Simulation_repo/Lammps/Ternary/N43_rho1p1944_T0p5/dt5e-3_ThermoTau1e0/in.ka2d_T0p5_equil`
- log_file: `/Users/xuhengyuan/Downloads/Notes/Simulation_repo/Lammps/Ternary/N43_rho1p1944_T0p5/dt5e-3_ThermoTau1e0/log_000.lammps`
- data_file: `/Users/xuhengyuan/Downloads/Notes/Simulation_repo/Lammps/Ternary/N43_rho1p1944_T0p5/dt5e-3_ThermoTau1e0/T0p5_init_000.data`
- dimension: 2
- box: (0.0, 6.0, 0.0, 6.0, -0.5, 0.5)
- N: 43
- rho: 1.1944444444444444
- T: 0.5
- dt: 0.005
- ThermoTau: 1.0

## Evidence
- N: from data file header: T0p5_init_000.data
- T: from fix nvt/velocity create in input
- dt: from timestep or dt variable in input
- ThermoTau: from Tdamp (thermostat damping) in input
- rho: computed from N and box bounds
- dimension: from input script
- box: from data file or region block

## Targets
- config_dir: `/Users/xuhengyuan/Downloads/Notes/Simulation_repo/Lammps/Ternary/config`
- experiment_dir: `/Users/xuhengyuan/Downloads/Notes/Simulation_repo/Lammps/Ternary/N43_rho1p1944_T0p5`
- run_dir: `/Users/xuhengyuan/Downloads/Notes/Simulation_repo/Lammps/Ternary/N43_rho1p1944_T0p5/dt5e-3_ThermoTau1e0`