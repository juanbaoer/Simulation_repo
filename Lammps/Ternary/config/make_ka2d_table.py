import numpy as np

# ---- KA2D parameters (from paper/SI) ----
# epsilon_ab and sigma_ab for types 1,2,3 (symmetric)
eps = {
    (1, 1): 1.0,  (1, 2): 1.5,  (2, 2): 0.5,
    (1, 3): 0.75, (2, 3): 1.5,  (3, 3): 0.75,
}
sig = {
    (1, 1): 1.0,  (1, 2): 0.8,  (2, 2): 0.88,
    (1, 3): 0.9,  (2, 3): 0.8,  (3, 3): 0.94,
}

# smoothing constants (continuous up to 2nd derivative at rcut)
C0 = 0.04049023795
C2 = -0.00970155098
C4 = 0.00062012616

def sym_get(d, a, b):
    i, j = (a, b) if a <= b else (b, a)
    return d[(i, j)]

# ---- table resolution ----
N = 10000

def V_and_F(r, epsilon, sigma):
    """
    V(r) = 4*eps * [ (sigma/r)^12 - (sigma/r)^6 + C0 + C2*(r/sigma)^2 + C4*(r/sigma)^4 ]
    F(r) = -dV/dr  (radial force magnitude)
    """
    inv = sigma / r
    inv6 = inv**6
    inv12 = inv6**2
    x = r / sigma

    V = 4.0 * epsilon * (inv12 - inv6 + C0 + C2 * x**2 + C4 * x**4)

    # d/dr(inv12 - inv6) = -12*sigma^12*r^-13 + 6*sigma^6*r^-7
    # d/dr(C2*x^2) = C2 * 2r/sigma^2
    # d/dr(C4*x^4) = C4 * 4r^3/sigma^4
    dVdr = 4.0 * epsilon * (
        (-12.0 * sigma**12) * r**(-13) +
        (  6.0 * sigma**6 ) * r**(-7)  +
        (  2.0 * C2 * r / sigma**2 ) +
        (  4.0 * C4 * r**3 / sigma**4 )
    )
    F = -dVdr
    return V, F

def write_section(f, section_name, epsilon, sigma):
    rcut = 2.5 * sigma
    # rmin must be >0; choose small enough to avoid "r < rmin" errors
    rmin = 0.2 * sigma

    r = np.linspace(rmin, rcut, N)
    V, F = V_and_F(r, epsilon, sigma)

    f.write(f"{section_name}\n")
    f.write(f"N {N} R {rmin:.10f} {rcut:.10f}\n\n")
    for i in range(N):
        f.write(f"{i+1:d} {r[i]:.10f} {V[i]:.12e} {F[i]:.12e}\n")
    f.write("\n")

with open("ka2d.table", "w") as f:
    f.write("# KA2D modified Lennard-Jones potential table for LAMMPS pair_style table\n")
    f.write("# Columns: index r E(r) F(r) ; F = -dE/dr\n\n")

    # Match these names with your LAMMPS pair_coeff keywords:
    sections = {
        (1,1): "AA",
        (1,2): "AB",
        (1,3): "AC",
        (2,2): "BB",
        (2,3): "BC",
        (3,3): "CC",
    }

    for (a,b), name in sections.items():
        e = sym_get(eps, a, b)
        s = sym_get(sig, a, b)
        write_section(f, name, e, s)

print("Wrote ka2d.table")
