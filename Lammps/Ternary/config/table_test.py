import numpy as np
import re

def read_table_sections(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    sections = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        # section name line like: "AA"
        name = line.split()[0]
        # next non-empty line should contain N ... R ...
        i += 1
        while i < len(lines) and (lines[i].strip()=='' or lines[i].lstrip().startswith('#')):
            i += 1
        hdr = lines[i].strip()
        m = re.search(r'\bN\s+(\d+)\b', hdr)
        if not m:
            # not a section; skip
            i += 1
            continue
        n = int(m.group(1))
        i += 1
        data = []
        while i < len(lines) and len(data) < n:
            s = lines[i].strip()
            if s and not s.startswith('#'):
                parts = s.split()
                # assume: idx r E F (4 cols)
                if len(parts) >= 4:
                    data.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1
        arr = np.array(data)
        sections[name] = arr  # columns: r, E, F
    return sections

secs = read_table_sections("/Users/xuhengyuan/Downloads/Notes/Simulation_repo/Lammps/Ternary/config/ka2d.table")
for name, arr in secs.items():
    r, E, F = arr[:,0], arr[:,1], arr[:,2]
    dE = np.zeros_like(E)
    dE[1:-1] = (E[2:] - E[:-2])/(r[2:] - r[:-2])
    dE[0] = (E[1]-E[0])/(r[1]-r[0])
    dE[-1] = (E[-1]-E[-2])/(r[-1]-r[-2])
    mismatch = np.abs(F + dE)
    k = mismatch.argmax()
    print(f"{name}: worst idx={k}, r={r[k]:.6g}, E={E[k]:.6g}, F={F[k]:.6g}, |F + dE/dr|={mismatch[k]:.3e}")
