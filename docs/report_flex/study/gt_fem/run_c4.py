"""run_c4.py -- verification of axi_fem.py and the C4 (centred circular flat punch) reference.

  python run_c4.py verify     -> c4_verify.json (patch tests, confined disc)
  python run_c4.py runs       -> runs_c4/*.json (a = 5, 2.5 mm; frictionless (C4) and no-slip (C4n);
                                 R_out = 25.5 mm (inscribed circle of the pad) and 51 mm; levels m = 1, 2, 4, 8)
"""
import os, sys, json, time
import numpy as np
import axi_fem as ax

HERE = os.path.dirname(os.path.abspath(__file__))
T = 1.5
COMBOS = [("full", 0.45), ("full", 0.49), ("mixed", 0.45), ("mixed", 0.49), ("mixed", 0.499)]


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def verify():
    out = {}
    br = ax.with_lines(0, 6.0, [2.0], [2.0, 6.0], 0.1, 1.0, 1.4)
    bz = ax.with_lines(0, T, [], [0, T], 0.1, 0.4, 1.4)
    rng = np.random.default_rng(0)
    br[1:-1] += 0.1 * np.diff(br)[:-1] * rng.uniform(-1, 1, br.size - 2)     # irregular sizes
    g = ax.AxiGrid(br, bz)
    R, Z = np.meshgrid(g.r, g.z, indexing="ij"); R, Z = R.ravel(), Z.ravel()
    fields = {"uniaxial_strain_z": (0 * R, Z, np.array([0, 1, 0, 0.])),
              "radial_plus_axial": (R, Z, np.array([1, 1, 1, 0.])),
              "quadratic_equilibrated": None}       # u_r = r z, u_z = C z^2, C = -(lam+mu)/(lam+2mu): body-force free
    vol = np.pi * 6.0 ** 2 * T
    I, Kk = np.meshgrid(np.arange(g.nr), np.arange(g.nz), indexing="ij"); I, Kk = I.ravel(), Kk.ravel()
    bnd = np.nonzero((I == 0) | (I == g.nr - 1) | (Kk == 0) | (Kk == g.nz - 1))[0]
    for form in ("full", "mixed"):
        for nu in (0.3, 0.45, 0.499):
            K = ax.stiffness(g, 1.0, nu, form)
            lam, mu = lame(1.0, nu)
            D = np.array([[lam + 2 * mu, lam, lam, 0], [lam, lam + 2 * mu, lam, 0], [lam, lam, lam + 2 * mu, 0],
                          [0, 0, 0, mu]])
            for name, fld in fields.items():
                if fld is None:
                    C = -(lam + mu) / (lam + 2 * mu)
                    ur, uz = R * Z, C * Z ** 2
                    xr, wr = np.polynomial.legendre.leggauss(8)
                    rq, zq = 3.0 * (xr + 1), 0.75 * (xr + 1)
                    RQ, ZQ = np.meshgrid(rq, zq, indexing="ij"); WQ = np.outer(3.0 * wr, 0.75 * wr) * 2 * np.pi * RQ
                    ep = np.stack([ZQ, 2 * C * ZQ, ZQ, RQ], -1)                  # (e_rr, e_zz, e_tt, g_rz)
                    W = 0.5 * np.sum(WQ * np.einsum("abi,ij,abj->ab", ep, D, ep))
                else:
                    ur, uz, eps = fld
                    W = 0.5 * eps @ D @ eps * vol
                Uex = np.stack([ur, uz], 1).ravel()
                fixed = np.concatenate([2 * bnd, 2 * bnd + 1])
                U, Rf, res = ax.solve(K, fixed, Uex[fixed])
                out["%s/nu%.3f/%s" % (form, nu, name)] = dict(
                    max_nodal_err=float(np.max(np.abs(U - Uex))), energy_ratio=float(0.5 * U @ (K @ U) / W))
    # confined disc: full no-slip platen, rollers on r = R_out -> K = M pi R^2 / t
    for form in ("full", "mixed"):
        for nu in (0.45, 0.49, 0.499):
            K = ax.stiffness(g, 1.0, nu, form)
            fixed, vals, pun = ax.punch_bcs(g, 6.0, "noslip", 1.0, lateral="roller")
            U, Rf, res = ax.solve(K, fixed, vals)
            Fz = Rf[2 * pun + 1].sum()
            lam, mu = lame(1.0, nu)
            ex = (lam + 2 * mu) * np.pi * 36 / T
            out["confined/%s/nu%.3f" % (form, nu)] = dict(K_N_per_mm=float(Fz), exact=float(ex), rel_err=float(Fz / ex - 1))
    json.dump(out, open(os.path.join(HERE, "c4_verify.json"), "w"), indent=1)
    worst = max(abs(v.get("max_nodal_err", 0)) for v in out.values())
    worst_e = max(abs(v.get("energy_ratio", 1) - 1) for v in out.values())
    worst_c = max(abs(v.get("rel_err", 0)) for v in out.values())
    print("axi patch: max nodal err %.1e, max |energy ratio - 1| %.1e; confined max rel err %.1e" % (worst, worst_e, worst_c))


def mesh(a, R, m):
    br = ax.with_lines(0, R, [a], [a, R], 0.04, 1.0, 1.3)
    bz = ax.with_lines(0, T, [], [0.0, T], 0.04, 0.25, 1.3)
    return ax.subdivide(br, m), ax.subdivide(bz, m)


def one(a, R, bc, form, nu, m):
    t0 = time.time()
    br, bz = mesh(a, R, m)
    g = ax.AxiGrid(br, bz)
    K = ax.stiffness(g, 1.0, nu, form)
    fixed, vals, pun = ax.punch_bcs(g, a, bc, 1.0)
    U, Rf, res = ax.solve(K, fixed, vals)
    Fz = float(Rf[2 * pun + 1].sum())
    E_ = 0.5 * float(U @ (K @ U))
    Uz0 = U[2 * g.node(np.arange(g.nr), 0) + 1]
    Ur0 = U[2 * g.node(np.arange(g.nr), 0)]
    Rz0 = Rf[2 * g.node(np.arange(g.nr), 0) + 1]
    Rb = Rf[2 * g.node(np.arange(g.nr), g.nz - 1) + 1]
    return dict(a_mm=a, R_out_mm=R, bc=bc, form=form, nu=nu, m=m, n_elem=[g.ner, g.nez], n_dof=g.ndof,
                hmin_mm=float(min(np.diff(br).min(), np.diff(bz).min())),
                K_N_per_mm=Fz, K_energy_N_per_mm=2 * E_, F_on_blade_z_N_per_mm=float(-Rb.sum()), rel_residual=res,
                r_mm=g.r.tolist(), uz_free_per_delta=Uz0.tolist(), ur_free_per_delta=Ur0.tolist(),
                Rz_free_nodal_N_per_mm=Rz0.tolist(), t_s=round(time.time() - t0, 2))


def runs():
    os.makedirs(os.path.join(HERE, "runs_c4"), exist_ok=True)
    jobs = []
    for a in (5.0, 2.5):
        for R in (25.5, 51.0):
            for bc in ("frictionless", "noslip"):
                for form, nu in COMBOS:
                    if R == 51.0 and (form, nu) not in (("full", 0.45), ("mixed", 0.499)):
                        continue
                    for m in (1, 2, 4, 8):
                        jobs.append((a, R, bc, form, nu, m))
    for (a, R, bc, form, nu, m) in jobs:
        fn = os.path.join(HERE, "runs_c4", "C4_a%.1f_R%.1f_%s_%s_nu%.3f_m%d.json" % (a, R, bc, form, nu, m))
        if os.path.exists(fn):
            continue
        o = one(a, R, bc, form, nu, m)
        json.dump(o, open(fn, "w"))
        print("a=%.1f R=%.1f %-12s %-5s nu=%.3f m=%d ndof=%7d K=%.7f N/um K_E=%.7f res=%.0e %.1fs" % (
            a, R, bc, form, nu, m, o["n_dof"], o["K_N_per_mm"] / 1e3, o["K_energy_N_per_mm"] / 1e3,
            o["rel_residual"], o["t_s"]), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    {"verify": verify, "runs": runs}[sys.argv[1]]()
