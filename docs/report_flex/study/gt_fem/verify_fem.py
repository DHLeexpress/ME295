"""verify_fem.py -- code verification of fem_layer.py / cases.py (writes verify_fem.json).

V1  Kronecker assembly == element-by-element isoparametric 27-node assembly (box meshes), both forms.
V2  P0 patch tests (uniaxial strain, simple shear, hydrostatic): exact linear field on the boundary nodes,
    interior solved; interior nodal error and energy ratio (U.K.U / (2 W_exact)); on a graded box mesh
    (Kronecker) and on a trilinearly distorted mesh (element-by-element), full and mixed forms.
V3  G1 confined layer: K = M A / t exactly.
V4  Method of manufactured solutions: smooth non-polynomial u, consistent body force, Dirichlet data =
    nodal interpolant; L2 and H1-seminorm error rates on uniform and graded nested meshes (expect 3 and 2).
V5  Linear solver: multifrontal Cholesky (NDCholesky) vs SuperLU.
V6  Symmetry reductions: quarter/half models vs the full model (C1, C1f, C2, C3).
V7  graded_breaks: first element at a singular line and largest element vs the requested hmin / hmax.
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
import fem_layer as fl
import cases

RES = {}
NUS = (0.3, 0.45, 0.49, 0.499)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ V1
def v1():
    g = fl.Grid3D(np.array([0, 0.7, 1.5, 3.0]), np.array([0, 1.0, 1.6, 2.0, 3.1]), np.array([0, 0.5, 1.5]))
    out = {}
    for form in ("full", "mixed"):
        for nu in NUS:
            K1 = fl.stiffness(g, 1.0, nu, form)
            K2 = fl.stiffness_elementwise(g, 1.0, nu, form)
            out["%s_nu%.3f" % (form, nu)] = float(abs(K1 - K2).max() / abs(K2).max())
    RES["V1_kron_vs_elementwise_max_rel_diff"] = out
    log("V1", out)


# ------------------------------------------------------------------------------------------------ V2
STATES = {   # displacement gradient H (u = H x); exact energy density W = 0.5 eps:C:eps
    "uniaxial_strain_zz": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1e-3]]),
    "simple_shear_yz": np.array([[0, 0, 0], [0, 0, 1e-3], [0, 0, 0]]),       # u_y = g z
    "hydrostatic": 1e-3 * np.eye(3),
}


def exact_energy_density(H, nu, E=1.0):
    lam, mu, _ = fl.lame(E, nu)
    eps = 0.5 * (H + H.T)
    return 0.5 * (lam * np.trace(eps) ** 2 + 2 * mu * np.sum(eps * eps))


def patch(grid, K, X, H, nu):
    n = grid.N
    u_ex = (X @ H.T).ravel()                     # DOF 3*node + comp
    ia, ib, ic = [a.ravel() for a in np.meshgrid(np.arange(grid.nx), np.arange(grid.ny), np.arange(grid.nz),
                                                 indexing="ij")]
    bnd = (ia == 0) | (ia == grid.nx - 1) | (ib == 0) | (ib == grid.ny - 1) | (ic == 0) | (ic == grid.nz - 1)
    fixed = (3 * np.nonzero(bnd)[0][:, None] + np.arange(3)).ravel()
    U, R, info = fl.solve_dirichlet(K, fixed, u_ex[fixed], method="superlu")
    U = U[:, 0]
    err = np.max(np.abs(U - u_ex)) / np.max(np.abs(u_ex))
    vol = np.ptp(X[:, 0]) * np.ptp(X[:, 1]) * np.ptp(X[:, 2]) if K is not None else None
    return float(err), float(0.5 * U @ (K @ U))


def v2():
    out = {}
    # (a) graded box mesh, Kronecker assembly
    bx = fl.graded_breaks(0, 4.0, [1.3], [4.0], 0.05, 1.0, 1.5)
    by = fl.graded_breaks(0, 3.0, [], [0.0, 3.0], 0.05, 1.0, 1.5)
    bz = fl.graded_breaks(0, 1.5, [], [0.0, 1.5], 0.05, 0.5, 1.5)
    g = fl.Grid3D(bx, by, bz)
    X = g.coords()
    vol = 4.0 * 3.0 * 1.5
    for form in ("full", "mixed"):
        for nu in NUS:
            K = fl.stiffness(g, 1.0, nu, form)
            for s, H in STATES.items():
                err, en = patch(g, K, X, H, nu)
                out["box_kron/%s/nu%.3f/%s" % (form, nu, s)] = dict(
                    max_interior_rel_err=err, energy_ratio=en / (exact_energy_density(H, nu) * vol))
    # (b) trilinearly distorted mesh, element-by-element isoparametric assembly
    g = fl.Grid3D(np.linspace(0, 2, 4), np.linspace(0, 2, 4), np.linspace(0, 1.5, 3))
    Xb = g.coords()
    rng = np.random.default_rng(1)
    # perturb the element-corner nodes (even indices) that are not on the boundary planes, then put the
    # mid-side/-face/-body nodes at the trilinear positions -> straight-edged distorted hexahedra
    nx, ny, nz = g.nx, g.ny, g.nz
    P = Xb.reshape(nx, ny, nz, 3).copy()
    for a in range(0, nx, 2):
        for b in range(0, ny, 2):
            for c in range(0, nz, 2):
                if 0 < a < nx - 1 and 0 < b < ny - 1 and 0 < c < nz - 1:
                    P[a, b, c] += rng.uniform(-0.15, 0.15, 3)
                elif 0 < a < nx - 1 and 0 < b < ny - 1:        # on z faces: move in-plane only
                    P[a, b, c, :2] += rng.uniform(-0.15, 0.15, 2)
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                if a % 2 == 0 and b % 2 == 0 and c % 2 == 0:
                    continue
                a0, a1 = (a, a) if a % 2 == 0 else (a - 1, a + 1)
                b0, b1 = (b, b) if b % 2 == 0 else (b - 1, b + 1)
                c0, c1 = (c, c) if c % 2 == 0 else (c - 1, c + 1)
                P[a, b, c] = np.mean([P[i, j, k] for i in (a0, a1) for j in (b0, b1) for k in (c0, c1)], axis=0)
    Xd = P.reshape(-1, 3)
    vol = 2.0 * 2.0 * 1.5                       # boundary planes unchanged -> same volume
    for form in ("full", "mixed"):
        for nu in (0.45, 0.499):
            K = fl.stiffness_elementwise(g, 1.0, nu, form, coords=Xd)
            for s, H in STATES.items():
                err, en = patch(g, K, Xd, H, nu)
                out["distorted_elementwise/%s/nu%.3f/%s" % (form, nu, s)] = dict(
                    max_interior_rel_err=err, energy_ratio=en / (exact_energy_density(H, nu) * vol))
    RES["V2_patch_tests"] = out
    worst = max(abs(v["energy_ratio"] - 1) for v in out.values())
    worst_e = max(v["max_interior_rel_err"] for v in out.values())
    log("V2 patch tests: max |energy ratio - 1| = %.2e, max interior nodal rel. error = %.2e" % (worst, worst_e))


# ------------------------------------------------------------------------------------------------ V3
def v3():
    out = {}
    for form in ("full", "mixed"):
        for nu in (0.45, 0.49, 0.499):
            r = cases.run("G1", nu, form, 1, p=dict(hmin=0.3, hmax=4.0, hzmax=0.75), verbose=False)
            exact = fl.confined_modulus(1.0, nu) * 51.0 * 71.0 / 1.5
            out["%s/nu%.3f" % (form, nu)] = dict(K_N_per_mm=r["F_platen_N_per_mm"][2], exact_N_per_mm=exact,
                                                 rel_err=r["F_platen_N_per_mm"][2] / exact - 1.0,
                                                 rel_err_energy=r["K_energy_N_per_mm"] / exact - 1.0)
    RES["V3_G1_confined"] = out
    log("V3 G1:", {k: "%.1e" % v["rel_err"] for k, v in out.items()})


# ------------------------------------------------------------------------------------------------ V4
def mms_problem():
    # u_i = sum of products of sines with O(1) wavelengths on a thin box 3 x 2 x 1.5
    u = [
        [(1.0, ((1.1, 0.3), (0.9, 0.2), (1.7, 0.1))), (0.3, ((2.0, 0.0), (0.5, 1.0), (1.0, 0.4)))],
        [(0.8, ((0.7, 0.5), (1.3, 0.1), (1.2, 0.9)))],
        [(0.6, ((1.5, 0.2), (1.1, 0.7), (2.1, 0.3))), (0.2, ((0.4, 1.2), (2.2, 0.0), (0.8, 0.5)))],
    ]
    return u, (3.0, 2.0, 1.5)


def mms_errors(grid, U, u_terms):
    X, Y, Z, W, V = fl.tensor_eval(grid, U)
    l2 = sum(np.sum(W * (V[i] - fl.sep_eval(u_terms[i], X, Y, Z)) ** 2) for i in range(3))
    h1 = 0.0
    for d in range(3):
        _, _, _, _, Vd = fl.tensor_eval(grid, U, deriv=d)
        h1 += sum(np.sum(W * (Vd[i] - fl.sep_eval([fl.sep_diff(t, d) for t in u_terms[i]], X, Y, Z)) ** 2)
                  for i in range(3))
    return float(np.sqrt(l2)), float(np.sqrt(h1))


def mms_solve(grid, u_terms, nu, form):
    K = fl.stiffness(grid, 1.0, nu, form)
    f = fl.sep_load(grid, fl.mms_body_force(u_terms, 1.0, nu))
    Xn = grid.m1[0].x
    Yn = grid.m1[1].x
    Zn = grid.m1[2].x
    u_nodes = np.stack([fl.sep_eval(u_terms[i], Xn, Yn, Zn).ravel() for i in range(3)], axis=1).ravel()
    ia, ib, ic = [a.ravel() for a in np.meshgrid(np.arange(grid.nx), np.arange(grid.ny), np.arange(grid.nz),
                                                 indexing="ij")]
    bnd = (ia == 0) | (ia == grid.nx - 1) | (ib == 0) | (ib == grid.ny - 1) | (ic == 0) | (ic == grid.nz - 1)
    fixed = (3 * np.nonzero(bnd)[0][:, None] + np.arange(3)).ravel()
    U, R, info = fl.solve_dirichlet(K, fixed, u_nodes[fixed], f=f, grid=grid, method="nd")
    return U[:, 0]


def v4():
    u_terms, (Lx, Ly, Lz) = mms_problem()
    out = {}
    families = {
        "uniform": lambda m: (np.linspace(0, Lx, 2 * m + 1), np.linspace(0, Ly, 2 * m + 1), np.linspace(0, Lz, m + 1)),
        "graded": lambda m: tuple(fl.subdivide(b, m) for b in (
            fl.graded_breaks(0, Lx, [], [Lx], 0.1, 1.0, 1.6), fl.graded_breaks(0, Ly, [], [0.0], 0.1, 1.0, 1.6),
            fl.graded_breaks(0, Lz, [], [0.0, Lz], 0.1, 0.6, 1.6))),
    }
    for fam, mk in families.items():
        for form, nu in (("full", 0.45), ("mixed", 0.45), ("full", 0.499), ("mixed", 0.499), ("mixed", 0.49999)):
            rows = []
            for m in (1, 2, 4, 8):
                g = fl.Grid3D(*mk(m))
                U = mms_solve(g, u_terms, nu, form)
                l2, h1 = mms_errors(g, U, u_terms)
                hmax = max(np.max(mm.h) for mm in g.m1)
                rows.append(dict(m=m, n_dof=g.ndof, hmax=float(hmax), L2=l2, H1semi=h1))
            for k in range(1, len(rows)):
                r0, r1 = rows[k - 1], rows[k]
                q = np.log(r1["m"] / r0["m"])
                r1["rate_L2"] = float(np.log(r0["L2"] / r1["L2"]) / q)
                r1["rate_H1"] = float(np.log(r0["H1semi"] / r1["H1semi"]) / q)
            out["%s/%s/nu%g" % (fam, form, nu)] = rows
            log("V4 %-8s %-5s nu=%-7g" % (fam, form, nu),
                " ".join("m=%d L2=%.2e(%s) H1=%.2e(%s)" % (r["m"], r["L2"], "%.2f" % r["rate_L2"] if "rate_L2" in r else "-",
                                                         r["H1semi"], "%.2f" % r["rate_H1"] if "rate_H1" in r else "-")
                         for r in rows))
    RES["V4_mms"] = out


# ------------------------------------------------------------------------------------------------ V5
def v5():
    out = {}
    for case, nu, form in (("C1", 0.45, "full"), ("C3", 0.49, "mixed"), ("C2", 0.499, "mixed")):
        p = dict(hmin=0.25, hmax=4.0, hzmax=0.75)
        bx, by, bz = cases.mesh(case, 1, p)
        g = fl.Grid3D(bx, by, bz)
        K = fl.stiffness(g, 1.0, nu, form)
        fixed, vals, plat = cases.boundary_conditions(g, case)
        U1, R1, i1 = fl.solve_dirichlet(K, fixed, vals, grid=g, method="nd")
        U2, R2, i2 = fl.solve_dirichlet(K, fixed, vals, grid=g, method="superlu")
        out["%s/%s/nu%.3f" % (case, form, nu)] = dict(
            n_dof=g.ndof, rel_diff_U=float(np.linalg.norm(U1 - U2) / np.linalg.norm(U2)),
            residual_nd=i1["rel_residual_free"], residual_superlu=i2["rel_residual_free"])
    RES["V5_solver_nd_vs_superlu"] = out
    log("V5", out)


# ------------------------------------------------------------------------------------------------ V6
def v6():
    out = {}
    p = dict(hmin=0.3, hmax=4.0, hzmax=0.75)
    for case in ("C1", "C1f", "C2", "C3"):
        full = case + "_fullmodel"
        cases.CASES[full] = dict(cases.CASES[case], symx=None, symy=None)
        cases.MESH[full] = cases.MESH[case]
        for nu, form in ((0.45, "full"), (0.499, "mixed")):
            r1 = cases.run(case, nu, form, 1, p=p, verbose=False)
            r2 = cases.run(full, nu, form, 1, p=p, verbose=False)
            F1, F2 = np.array(r1["F_platen_N_per_mm"]), np.array(r2["F_platen_N_per_mm"])
            T1, T2 = np.array(r1["taxel_force_N_per_mm"]), np.array(r2["taxel_force_N_per_mm"])
            D1, D2 = np.array(r1["taxel_mean_disp_per_delta"]), np.array(r2["taxel_mean_disp_per_delta"])
            out["%s/%s/nu%.3f" % (case, form, nu)] = dict(
                F_reduced=F1.tolist(), F_full=F2.tolist(),
                max_abs_diff_F_rel=float(np.max(np.abs(F1 - F2)) / np.max(np.abs(F2))),
                max_abs_diff_taxel_force_rel=float(np.max(np.abs(T1 - T2)) / np.max(np.abs(T2))),
                max_abs_diff_taxel_disp=float(np.max(np.abs(D1 - D2))),
                n_dof_reduced=r1["n_dof"], n_dof_full=r2["n_dof"])
        del cases.CASES[full], cases.MESH[full]
    RES["V6_symmetry_reduction"] = out
    log("V6", {k: "%.1e" % v["max_abs_diff_F_rel"] for k, v in out.items()})


# ------------------------------------------------------------------------------------------------ V7
def v7():
    out = {}
    for hmin in (0.2, 0.02, 0.002):
        b = fl.graded_breaks(0.0, 25.5, [5.1, 15.3], [25.5], hmin, 3.0, 1.5)
        h = np.diff(b)
        out["hmin%g" % hmin] = dict(first_elem_at_singular=float(h[-1]), requested_hmin=hmin, max_elem=float(h.max()),
                                    max_neighbour_ratio=float(np.max(np.maximum(h[1:] / h[:-1], h[:-1] / h[1:]))),
                                    n_elem=int(h.size))
    RES["V7_graded_breaks"] = out
    log("V7", out)



# ------------------------------------------------------------------------------------------------ V4b
def v4b():
    """Solenoidal MMS: u = curl psi (div u = 0 exactly), f = -mu lap u (lambda-free).  A locking-free method
    must give errors independent of lambda; the pure displacement form may lock as nu -> 1/2."""
    psi = [[(1.0, ((0.9, 0.3), (1.1, 0.2), (1.3, 0.1)))],
           [(0.7, ((1.2, 0.5), (0.8, 0.1), (1.6, 0.4)))],
           [(0.5, ((1.4, 0.2), (1.0, 0.7), (0.9, 0.3)))]]
    neg = lambda t: (-t[0], t[1])
    u_terms = [[fl.sep_diff(t, 1) for t in psi[2]] + [neg(fl.sep_diff(t, 2)) for t in psi[1]],
               [fl.sep_diff(t, 2) for t in psi[0]] + [neg(fl.sep_diff(t, 0)) for t in psi[2]],
               [fl.sep_diff(t, 0) for t in psi[1]] + [neg(fl.sep_diff(t, 1)) for t in psi[0]]]
    Lx, Ly, Lz = 3.0, 2.0, 1.5
    out = {}
    families = {
        "uniform": lambda m: (np.linspace(0, Lx, 2 * m + 1), np.linspace(0, Ly, 2 * m + 1), np.linspace(0, Lz, m + 1)),
        "graded": lambda m: tuple(fl.subdivide(b, m) for b in (
            fl.graded_breaks(0, Lx, [], [Lx], 0.1, 1.0, 1.6), fl.graded_breaks(0, Ly, [], [0.0], 0.1, 1.0, 1.6),
            fl.graded_breaks(0, Lz, [], [0.0, Lz], 0.1, 0.6, 1.6))),
    }
    for fam, mk in families.items():
        for form, nu in (("full", 0.45), ("mixed", 0.45), ("full", 0.499), ("mixed", 0.499),
                         ("full", 0.49999), ("mixed", 0.49999)):
            rows = []
            for m in (1, 2, 4, 8):
                g = fl.Grid3D(*mk(m))
                mu = fl.shear_modulus(1.0, nu)
                f = [[(-mu * c, fn) for j in range(3) for t in u_terms[i]
                      for c, fn in (fl.sep_diff(fl.sep_diff(t, j), j),)] for i in range(3)]
                K = fl.stiffness(g, 1.0, nu, form)
                fv = fl.sep_load(g, f)
                Xn, Yn, Zn = g.m1[0].x, g.m1[1].x, g.m1[2].x
                u_nodes = np.stack([fl.sep_eval(u_terms[i], Xn, Yn, Zn).ravel() for i in range(3)], axis=1).ravel()
                ia, ib, ic = [a.ravel() for a in np.meshgrid(np.arange(g.nx), np.arange(g.ny), np.arange(g.nz),
                                                             indexing="ij")]
                bnd = (ia == 0) | (ia == g.nx - 1) | (ib == 0) | (ib == g.ny - 1) | (ic == 0) | (ic == g.nz - 1)
                fixed = (3 * np.nonzero(bnd)[0][:, None] + np.arange(3)).ravel()
                U, R, info = fl.solve_dirichlet(K, fixed, u_nodes[fixed], f=fv, grid=g, method="nd")
                l2, h1 = mms_errors(g, U[:, 0], u_terms)
                rows.append(dict(m=m, n_dof=g.ndof, L2=l2, H1semi=h1))
            for k in range(1, len(rows)):
                r0, r1 = rows[k - 1], rows[k]
                q = np.log(r1["m"] / r0["m"])
                r1["rate_L2"] = float(np.log(r0["L2"] / r1["L2"]) / q)
                r1["rate_H1"] = float(np.log(r0["H1semi"] / r1["H1semi"]) / q)
            out["%s/%s/nu%g" % (fam, form, nu)] = rows
            log("V4b %-8s %-5s nu=%-7g" % (fam, form, nu),
                " ".join("m=%d L2=%.2e(%s) H1=%.2e(%s)" % (r["m"], r["L2"], "%.2f" % r["rate_L2"] if "rate_L2" in r else "-",
                                                         r["H1semi"], "%.2f" % r["rate_H1"] if "rate_H1" in r else "-")
                         for r in rows))
    RES["V4b_mms_solenoidal"] = out

if __name__ == "__main__":
    which = sys.argv[1:] or ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]
    t0 = time.time()
    try:
        RES.update(json.load(open("verify_fem.json")))
    except Exception:
        pass
    for w in which:
        globals()[w]()
        json.dump(RES, open("verify_fem.json", "w"), indent=1)
    log("done in %.1fs" % (time.time() - t0))
