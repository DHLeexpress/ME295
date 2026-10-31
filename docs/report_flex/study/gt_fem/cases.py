"""cases.py -- benchmark cases G1, C1, C1f, C2, C3, C6 on the 51 x 71 x 1.5 mm PDMS pad (fem_layer.py).

Frame (same as the PDMS frame of tactile.py): origin at the centre of the free face, x across (|x| <= 25.5 mm),
y along the finger (|y| <= 35.5 mm), z through the thickness, +z INTO the layer: free face z = 0, bonded face
z = t = 1.5 mm.  The bonded face is clamped (u = 0); the four lateral faces are traction-free (the pocket walls
of the blade are not modelled, as in the MuJoCo flex, which pins only the back face).
Loads are imposed displacements delta = 1 (mm) of a rigid platen/punch on the free face (+z = compression),
so every output is per unit imposed displacement.  Internal units mm, N, MPa; converted to SI on output.
Forces reported = force exerted BY the platen ON the layer (= contact force pressing into the skin).
"""
import time
import numpy as np
import fem_layer as fl

A, B, T = 25.5, 35.5, 1.5                     # half-width (x), half-length (y), thickness (mm)
E_MOD = 1.0                                   # MPa
XT = -25.5 + 10.2 * np.arange(6)              # taxel boundaries in x (5 columns, i)
YT = -35.5 + (71.0 / 7.0) * np.arange(8)      # taxel boundaries in y (7 rows, j)
Y_C3 = YT[4]                                  # +5.0714 mm: C3 platen covers y >= Y_C3 (rows j = 4..6)

# contact rectangles in full-face coordinates (x0, x1, y0, y1)
CASES = {
    "G1":  dict(rect=(-A, A, -B, B), bc="noslip", load="z", symx="sym", symy="sym", lateral="roller",
                desc="laterally confined layer (rollers on the 4 lateral faces), uniform compression"),
    "C1":  dict(rect=(-A, A, -B, B), bc="noslip", load="z", symx="sym", symy="sym",
                desc="rigid platen over the whole free face, no-slip, normal displacement"),
    "C1f": dict(rect=(-A, A, -B, B), bc="frictionless", load="z", symx="sym", symy="sym",
                desc="rigid platen over the whole free face, frictionless, normal displacement"),
    "C2":  dict(rect=(-A, A, -B, B), bc="noslip", load="y", symx="sym", symy="anti",
                desc="rigid platen over the whole free face, no-slip, tangential displacement along +y"),
    "C3":  dict(rect=(-A, A, Y_C3, B), bc="noslip", load="z", symx="sym", symy=None,
                desc="rigid platen over y >= +5.0714 mm (taxel rows 4-6), no-slip, normal displacement"),
    "C6":  dict(rect=(XT[2], XT[3], YT[3], YT[4]), bc="noslip", load="z", symx="sym", symy="sym",
                desc="rigid square punch exactly over the centre taxel (10.2 x 10.143 mm), no-slip, normal"),
    "C6f": dict(rect=(XT[2], XT[3], YT[3], YT[4]), bc="frictionless", load="z", symx="sym", symy="sym",
                desc="rigid square punch exactly over the centre taxel, frictionless, normal"),
}
LOAD_DIR = {"z": 2, "y": 1, "x": 0}


def domain(case):
    c = CASES[case]
    x0 = 0.0 if c["symx"] else -A
    y0 = 0.0 if c["symy"] else -B
    return x0, y0


# ----------------------------------------------------------------------------------------------- meshes
# Base meshes (level m = 1); level m = every element split into m (nested, geometrically similar).
MESH = {
    # hmin: element size at the singular lines (in-plane and through the thickness), g: growth factor,
    # hmax: largest in-plane size, hzmax: largest size through the thickness.
    "C1":  dict(hmin=0.08, g=1.5, hmax=3.0, hzmax=0.5, zsing=(0.0, T)),
    "C3":  dict(hmin=0.08, g=1.5, hmax=3.0, hzmax=0.5, zsing=(0.0, T)),
    "C6":  dict(hmin=0.08, g=1.5, hmax=3.0, hzmax=0.5, zsing=(0.0,)),
}
MESH["G1"] = MESH["C1f"] = MESH["C2"] = MESH["C1"]
MESH["C6f"] = MESH["C6"]


def singular_lines(case):
    """In-plane coordinates of the singular lines (pad edges that are loaded, contact edges)."""
    c = CASES[case]
    x0, y0 = domain(case)
    x0r, x1r, y0r, y1r = c["rect"]
    base = case.split("_")[0]          # (verify_fem.py V6 uses names like "C3_fullmodel")
    if base in ("C6", "C6f"):          # punch edges only (the pad edges carry ~nothing: 13 t away)
        sx, sy = [x1r] + ([x0r] if x0 < 0 else []), [y1r] + ([y0r] if y0 < 0 else [])
    else:
        sx = [A] + ([-A] if x0 < 0 else [])
        sy = [B] + ([-B] if y0 < 0 else [])
        if base == "C3":
            sy = sy + [Y_C3]
    return sx, sy


def base_breaks(case, p=None):
    p = dict(MESH[case], **(p or {}))
    x0, y0 = domain(case)
    sx, sy = singular_lines(case)
    rect = CASES[case]["rect"]
    lx = [v for v in list(XT) + [rect[0], rect[1]] if x0 < v < A]
    ly = [v for v in list(YT) + [rect[2], rect[3]] if y0 < v < B]
    bx = fl.graded_breaks(x0, A, lx, sx, p["hmin"], p["hmax"], p["g"])
    by = fl.graded_breaks(y0, B, ly, sy, p["hmin"], p["hmax"], p["g"])
    bz = fl.graded_breaks(0.0, T, [], list(p["zsing"]), p["hmin"], p["hzmax"], p["g"])
    return bx, by, bz


def mesh(case, m, p=None):
    bx, by, bz = base_breaks(case, p)
    return fl.subdivide(bx, m), fl.subdivide(by, m), fl.subdivide(bz, m)


# ----------------------------------------------------------------------------------------------- BCs
def boundary_conditions(grid, case, delta=1.0):
    """Fixed DOFs and their values; platen node indices (free face, inside the contact rectangle)."""
    c = CASES[case]
    mx, my, mz = grid.m1
    ia, ib, ic = [a.ravel() for a in np.meshgrid(np.arange(grid.nx), np.arange(grid.ny), np.arange(grid.nz),
                                                 indexing="ij")]
    X, Y = mx.x[ia], my.x[ib]
    node = grid.node(ia, ib, ic)
    tol = 1e-9
    # constraints are collected per set; duplicates must agree (checked below)
    dofs, vals = [], []

    def vfix(nodes, comp, val):
        dofs.append(3 * nodes.astype(np.int64) + comp)
        vals.append(np.full(nodes.size, float(val)))

    back = node[ic == grid.nz - 1]
    for k in range(3):
        vfix(back, k, 0.0)
    x0r, x1r, y0r, y1r = c["rect"]
    on = (ic == 0) & (X >= x0r - tol) & (X <= x1r + tol) & (Y >= y0r - tol) & (Y <= y1r + tol)
    plat = node[on]
    dvec = np.zeros(3)
    dvec[LOAD_DIR[c["load"]]] = delta
    comps = (0, 1, 2) if c["bc"] == "noslip" else (2,)
    for k in comps:
        vfix(plat, k, dvec[k])
    if c["bc"] != "noslip" and c["load"] != "z":
        raise ValueError("a frictionless platen cannot impose a tangential displacement")
    if c["symx"] == "sym":
        vfix(node[ia == 0], 0, 0.0)
    if c["symy"] == "sym":
        vfix(node[ib == 0], 1, 0.0)
    elif c["symy"] == "anti":
        vfix(node[ib == 0], 0, 0.0)
        vfix(node[ib == 0], 2, 0.0)
    if c.get("lateral") == "roller":
        vfix(node[ia == grid.nx - 1], 0, 0.0)
        vfix(node[ib == grid.ny - 1], 1, 0.0)
        if c["symx"] is None:
            vfix(node[ia == 0], 0, 0.0)
        if c["symy"] is None:
            vfix(node[ib == 0], 1, 0.0)
    d = np.concatenate(dofs)
    v = np.concatenate(vals)
    order = np.argsort(d, kind="stable")
    d, v = d[order], v[order]
    first = np.r_[True, d[1:] != d[:-1]]
    vfirst = v[first]
    if np.any(np.abs(v - vfirst[np.cumsum(first) - 1]) > 1e-14):
        raise ValueError("inconsistent constraints")
    return d[first], vfirst, plat


# ----------------------------------------------------------------------------------------------- unfolding
def parities(case):
    """Parity of (u_x, u_y, u_z) (and of nodal forces) under the mirror about x = 0 and about y = 0."""
    c = CASES[case]
    px = (-1.0, 1.0, 1.0) if c["symx"] == "sym" else None
    if c["symy"] == "sym":
        py = (1.0, -1.0, 1.0)
    elif c["symy"] == "anti":
        py = (-1.0, 1.0, -1.0)
    else:
        py = None
    return px, py


def face_fields(grid, case, U, R, ic):
    """Displacement and nodal-force fields on the face z-index ic, unfolded to the full 51 x 71 face."""
    Uf = U.reshape(grid.nx, grid.ny, grid.nz, 3)[:, :, ic, :].copy()
    Rf = R.reshape(grid.nx, grid.ny, grid.nz, 3)[:, :, ic, :].copy()
    bx, by = grid.m1[0].breaks, grid.m1[1].breaks
    px, py = parities(case)
    if px is not None:
        Uf, Rf = fl.unfold_face(Uf, 0, px, False), fl.unfold_face(Rf, 0, px, True)
        bx = fl.unfold_breaks(bx)
    if py is not None:
        Uf, Rf = fl.unfold_face(Uf, 1, py, False), fl.unfold_face(Rf, 1, py, True)
        by = fl.unfold_breaks(by)
    return bx, by, Uf, Rf


def multiplicity(case):
    c = CASES[case]
    return (2 if c["symx"] else 1) * (2 if c["symy"] else 1)


# ----------------------------------------------------------------------------------------------- one run
def run(case, nu, form, m, p=None, method="nd", threads=None, delta=1.0, keep_fields=False, verbose=True):
    t0 = time.time()
    bx, by, bz = mesh(case, m, p)
    grid = fl.Grid3D(bx, by, bz)
    K = fl.stiffness(grid, E_MOD, nu, form)
    t_asm = time.time() - t0
    fixed, vals, plat = boundary_conditions(grid, case, delta)
    U, R, info = fl.solve_dirichlet(K, fixed, vals, grid=grid, method=method, threads=threads)
    U, R = U[:, 0], R[:, 0]
    energy = 0.5 * float(U @ (K @ U))
    del K
    c = CASES[case]
    out = dict(case=case, nu=nu, form=form, m=m, mesh_params=dict(MESH[case], **(p or {})),
               n_elem=[grid.m1[0].ne, grid.m1[1].ne, grid.m1[2].ne], n_dof=grid.ndof,
               hmin_mm=float(min(np.min(mm.h) for mm in grid.m1)), t_assemble_s=round(t_asm, 2), solver=info)
    mult = multiplicity(case)
    out["K_energy_N_per_mm"] = 2.0 * energy * mult / delta ** 2
    # free face
    bxf, byf, Uf, Rf = face_fields(grid, case, U, R, 0)
    F_tot = Rf.reshape(-1, 3).sum(0) / delta
    out["F_platen_N_per_mm"] = F_tot.tolist()
    ff = fl.FaceField(bxf, byf, Uf / delta, Rf / delta)
    ff.traction(c["rect"])
    out["taxel_force_N_per_mm"] = ff.taxel_forces(XT, YT).transpose(1, 0, 2).tolist()      # [j][i][xyz]
    out["taxel_mean_disp_per_delta"] = ff.taxel_mean_disp(XT, YT).transpose(1, 0, 2).tolist()
    # bonded face: force exerted by the layer ON the blade = -reaction there
    bxb, byb, Ub, Rb = face_fields(grid, case, U, R, grid.nz - 1)
    fb = fl.FaceField(bxb, byb, Ub / delta, -Rb / delta)
    fb.traction((-A, A, -B, B))
    out["F_on_blade_N_per_mm"] = (-Rb.reshape(-1, 3).sum(0) / delta).tolist()
    out["taxel_force_on_blade_N_per_mm"] = fb.taxel_forces(XT, YT).transpose(1, 0, 2).tolist()
    # profiles on the free face (nodes): u(x) at y = 0 and u(y) at x = 0; traction on the platen part
    jy0 = int(np.argmin(np.abs(ff.my.x)))
    ix0 = int(np.argmin(np.abs(ff.mx.x)))
    out["profile_x_mm"] = ff.mx.x.tolist()
    out["profile_y_mm"] = ff.my.x.tolist()
    out["u_free_at_y0_per_delta"] = ff.U[:, jy0, :].tolist()       # [ix][xyz]
    out["u_free_at_x0_per_delta"] = ff.U[ix0, :, :].tolist()       # [iy][xyz]
    out["t_free_at_y0_MPa_per_mm"] = ff.T[:, jy0, :].tolist()
    out["t_free_at_x0_MPa_per_mm"] = ff.T[ix0, :, :].tolist()
    out["t_blade_at_y0_MPa_per_mm"] = fb.T[:, int(np.argmin(np.abs(fb.my.x))), :].tolist()
    out["t_blade_at_x0_MPa_per_mm"] = fb.T[int(np.argmin(np.abs(fb.mx.x))), :, :].tolist()
    out["t_total_s"] = round(time.time() - t0, 2)
    if keep_fields:
        out["_fields"] = dict(grid=grid, U=U, R=R, ff=ff, fb=fb)
    if verbose:
        k = LOAD_DIR[c["load"]]
        print("%-4s nu=%.3f %-5s m=%d ndof=%8d  F=%s N/mm  K_E=%.6f  res=%.1e  t=%.1fs (factor %.1fs)" % (
            case, nu, form, m, grid.ndof, np.array2string(np.asarray(F_tot), precision=6),
            out["K_energy_N_per_mm"], info["rel_residual_free"], out["t_total_s"], info["t_factor_s"]),
            flush=True)
    return out
