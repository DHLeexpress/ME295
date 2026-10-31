# Parameter regimes: our PDMS layer vs the third-party flex users found in the survey.
# For a linear-elastic solid the explicit stability limit of an element of size l is h_crit ~ l / c_p with
# c_p = sqrt((lambda + 2 mu) / rho) (P-wave speed); CFL = h * c_p / l. Under the discrete integrator the stiffness is
# implicit (no stability limit), but the effective metric M + h^2 K has a stiffness-to-mass ratio ~ CFL^2 per vertex,
# which is what the PCG solve (3x3 block preconditioner) must handle. With F1 (StVK lambda doubled in MuJoCo) the
# effective c_p uses 2*lambda. All inputs are from the cited sources (see NOTES.md); nothing here is measured.
# usage: python regimes.py -> regimes.json
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def row(name, E, nu, rho, lmin, h, note, lam_factor=1.0):
    lam, mu = lame(E, nu)
    M = lam_factor * lam + 2 * mu
    cp = math.sqrt(M / rho)
    r = dict(case=name, E=E, nu=nu, rho=rho, l_min_mm=lmin * 1e3, h_ms=None if h is None else h * 1e3,
             lambda_factor=lam_factor, c_p_m_s=round(cp, 2), h_crit_us=round(1e6 * lmin / cp, 2), note=note)
    if h is not None:
        r["CFL"] = round(h * cp / lmin, 2)
        r["CFL_sq_(~h2K/m)"] = round((h * cp / lmin) ** 2, 1)
    return r


rows = []
rho_pdms = 965.0
for nu in (0.45, 0.49):
    for grid, sp in (("11x15x3", (0.051 / 10, 0.071 / 14, 0.0015 / 2)),
                     ("21x29x4", (0.051 / 20, 0.071 / 28, 0.0015 / 3)),
                     ("41x57x5", (0.051 / 40, 0.071 / 56, 0.0015 / 4))):
        lmin = min(sp)
        for lf in (1.0, 2.0):
            rows.append(row(f"ours PDMS {grid} nu={nu}", 1e6, nu, rho_pdms, lmin, 5e-4,
                            f"cell {sp[0]*1e3:.3f} x {sp[1]*1e3:.3f} x {sp[2]*1e3:.3f} mm, aspect {max(sp)/min(sp):.1f}; "
                            + ("true lambda" if lf == 1 else "MuJoCo StVK as shipped (F1: 2*lambda)"), lf))
# third-party / official cases (inputs from the sources; see NOTES.md)
rows.append(row("badinkajink/hand flex cap (MuJoCo 3.14, discrete+CG)", 1e7, 0.45, 1200.0, 1e-3, 1e-3,
                "1 mm hex columns, two 1.5 mm layers; ran at 1 ms under discrete; dropped for cost/friction", 2.0))
rows.append(row("Msornerrrr LEAP flexpad (unused variant)", 1e5, 0.45, 4e-3 / (2.5e-3 * 20e-3 * 20e-3), 2.5e-3, None,
                "grid 2x3x3, spacing 2.5/10/10 mm, mass 4 g -> rho 4000 kg/m3; timestep not in the XML", 2.0))
rows.append(row("MetalSim mesh_sweep rod (explicit)", 1e5, 0.4, 1000.0, 5e-3, 2.5e-4,
                "4-across 5 mm cells diverged at dt 2.5e-4 under explicit flex elasticity (their doc)", 2.0))
rows.append(row("official floppy.xml (Euler)", 5e4, 0.2, None or 25.0 / (2.3 * 0.3 * 0.3), 0.1, 1e-3,
                "0.1 m cells; dev: 1 ms is 'quite close to the stability limit' (discussion #2652)", 2.0))
json.dump(rows, open(os.path.join(HERE, "regimes.json"), "w"), indent=1)
for r in rows:
    print({k: r[k] for k in ("case", "lambda_factor", "c_p_m_s", "l_min_mm", "h_crit_us", "CFL") if k in r})
