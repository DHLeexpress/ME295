# One benchmark model: build the pad flex in MuJoCo, extract K by FD of qfrc_spring, cross-check it,
# solve the static cases (G1, C1, C1f, C2, C3, C6, C4) with exact BCs, and the conditioning study of
# M + h^2 K (pinned back face). Usage: python run_model.py '<json config>' outdir
import sys, os, json, time, traceback
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

T_START = time.time()


def log(msg):
    print(f"[{time.time()-T_START:8.1f}s] {msg}", file=sys.stderr, flush=True)


DELTA = 1.0  # linear problems: unit prescribed displacement, K = F / DELTA


def bc_sets(P):
    g = P["g"]; cnt = P["cnt"]; X = P["X"]
    back = np.where(g[:, 2] == cnt[2] - 1)[0]
    face = np.where(g[:, 2] == 0)[0]
    xs = np.where((g[:, 0] == 0) | (g[:, 0] == cnt[0] - 1))[0]
    ys = np.where((g[:, 1] == 0) | (g[:, 1] == cnt[1] - 1))[0]
    return back, face, xs, ys


def fixed_from(spec):
    """spec: list of (point ids, component, value); later entries override earlier ones."""
    dct = {}
    for ids, k, v in spec:
        for i in ids:
            dct[3 * int(i) + k] = v
    fx = np.array(sorted(dct)); return fx, np.array([dct[i] for i in fx])


def loaded_weights(P, S, region_pts):
    """Wc[a,T]: free-face shape-function integrals over taxel T restricted to surface elements whose
    points are all in region_pts (the loaded area); used to attribute nodal reactions to taxels."""
    X = P["X"]; n = P["n"]
    inreg = np.zeros(n, bool); inreg[region_pts] = True
    W = np.zeros((n, NTX * NTY))
    if S["kind"] == "tri":
        for tri in S["conn"]:
            if not inreg[tri].all():
                continue
            p = X[tri][:, :2]
            area = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
            cxy = p.mean(0); i, j = taxel_of(cxy[0], cxy[1]); W[tri, i * NTY + j] += area / 3
    else:
        o = S["order"]; w1 = np.array([0.5, 0.5]) if o == 1 else np.array([1 / 6, 2 / 3, 1 / 6])
        w2 = np.outer(w1, w1).ravel()
        for q in S["conn"]:
            if not inreg[q].all():
                continue
            p = X[q][:, :2]; dx = np.ptp(p[:, 0]); dy = np.ptp(p[:, 1])
            cxy = p.mean(0); i, j = taxel_of(cxy[0], cxy[1]); W[q, i * NTY + j] += w2 * dx * dy
    return W


def taxel_maps(P, u, R, face, Wfull, Wload):
    Az = np.zeros(P["n"]); Az[face] = R[3 * face + 2]
    s = Wload.sum(1)
    frac = np.divide(Wload, s[:, None], out=np.zeros_like(Wload), where=s[:, None] > 0)
    unassigned = float(np.abs(Az[s == 0]).sum() / max(np.abs(Az).sum(), 1e-300))
    F_T = (Az[:, None] * frac).sum(0)                       # N per m of DELTA
    uz = np.zeros(P["n"]); uz[face] = u[3 * face + 2]
    ubar = (uz[:, None] * Wfull).sum(0) / (TX * TY)
    # taxel arrays indexed [i (x), j (y)]
    return dict(force_N_per_um=(F_T * 1e-6).reshape(NTX, NTY), mean_uz_over_delta=(ubar / DELTA).reshape(NTX, NTY),
                reaction_unassigned_frac=unassigned)


def static_cases(c, P, K, el):
    dd = describe(c); nu = c["nu"]
    back, face, xs, ys = bc_sets(P)
    X = P["X"]; x, y = X[:, 0], X[:, 1]
    S = surface(P, el)
    Wfull = surface_integrals(P, S)
    out = {}
    y3 = -LY / 2 + 4 * TY                       # +5.0714 mm, start of taxel row 4
    ctr = face[(np.abs(x[face]) <= TX / 2 + TOL) & (np.abs(y[face]) <= TY / 2 + TOL)]
    half = face[y[face] >= y3 - TOL]
    cases = {
        "G1": [(back, 0, 0.), (back, 1, 0.), (back, 2, 0.), (xs, 0, 0.), (ys, 1, 0.), (face, 2, DELTA)],
        "C1": [(back, 0, 0.), (back, 1, 0.), (back, 2, 0.), (face, 0, 0.), (face, 1, 0.), (face, 2, DELTA)],
        "C1f": [(back, 0, 0.), (back, 1, 0.), (back, 2, 0.), (face, 2, DELTA)],
        "C2": [(back, 0, 0.), (back, 1, 0.), (back, 2, 0.), (face, 0, 0.), (face, 1, DELTA), (face, 2, 0.)],
        "C3": [(back, 0, 0.), (back, 1, 0.), (back, 2, 0.), (half, 0, 0.), (half, 1, 0.), (half, 2, DELTA)],
        "C6ns": [(back, 0, 0.), (back, 1, 0.), (back, 2, 0.), (ctr, 0, 0.), (ctr, 1, 0.), (ctr, 2, DELTA)],
        "C6f": [(back, 0, 0.), (back, 1, 0.), (back, 2, 0.), (ctr, 2, DELTA)],
    }
    region = {"G1": face, "C1": face, "C1f": face, "C2": face, "C3": half, "C6ns": ctr, "C6f": ctr}
    for a in (5e-3, 2.5e-3):
        disc = face[x[face] ** 2 + y[face] ** 2 <= (a * (1 + 1e-9)) ** 2]
        tag = f"C4n_a{a*1e3:g}mm"
        if disc.size:
            cases[tag] = [(back, 0, 0.), (back, 1, 0.), (back, 2, 0.), (disc, 2, DELTA)]; region[tag] = disc
        else:
            out[tag] = dict(K_N_per_um=None, note="no free-face point inside the punch radius", n_constrained_points=0)
    for name, spec in cases.items():
        t0 = time.time(); log(f"case {name}")
        fx, uv = fixed_from(spec)
        u, R, resid = solve_dirichlet(K, fx, uv)
        comp = {"C2": 1}.get(name, 2)
        reg = region[name]
        F = R[3 * reg + comp].sum()
        Fback = R[3 * back + comp].sum()
        rec = dict(K_N_per_um=F / DELTA * 1e-6, n_constrained_points=int(reg.size),
                   equilibrium_rel=float(abs(F + Fback) / abs(F)), free_dof_residual_rel=resid,
                   tangential_reaction_rel=float(np.abs([R[3 * reg + k].sum() for k in range(3) if k != comp]).max() / abs(F)),
                   solve_time_s=time.time() - t0)
        if name in ("C1", "C1f", "C3", "C6ns", "C6f", "G1"):
            Wl = loaded_weights(P, S, reg)
            rec["taxel"] = taxel_maps(P, u, R, face, Wfull, Wl)
        if name == "C2":
            uy = np.zeros(P["n"]); uy[face] = u[3 * face + 1]
        out[name] = rec
    # UP: uniform pressure on the free face (tangential traction 0), back face pinned; K = F / mean(u_z)
    t0 = time.time()
    fx, uv = fixed_from([(back, 0, 0.), (back, 1, 0.), (back, 2, 0.)])
    fvec = np.zeros(3 * P["n"]); wsum = Wfull.sum(1)
    fvec[3 * face + 2] = wsum[face] / AREA            # total force 1 N
    N = 3 * P["n"]; mask = np.zeros(N, bool); mask[fx] = True; fr = np.where(~mask)[0]
    u = np.zeros(N); u[fr] = spla.splu(K[fr][:, fr].tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)).solve(fvec[fr])
    uz = u[3 * face + 2]
    ubar_area = float((uz * wsum[face]).sum() / AREA)
    ti, tj = taxel_of(x[face], y[face])            # rig2.py-style: plain vertex mean per taxel, floor binning
    tm = np.zeros((NTX, NTY)); tc = np.zeros((NTX, NTY)); np.add.at(tm, (ti, tj), uz); np.add.at(tc, (ti, tj), 1)
    ubar_vert = float((tm / np.maximum(tc, 1)).mean())
    out["UP"] = dict(K_area_mean_N_per_um=1.0 / ubar_area * 1e-6, K_vertex_taxel_mean_N_per_um=1.0 / ubar_vert * 1e-6,
                     max_over_mean_uz=float(uz.max() / ubar_area), solve_time_s=time.time() - t0)
    out["G1"]["exact_N_per_um"] = exact_G1(nu) * 1e-6
    out["G1"]["ratio_to_exact"] = out["G1"]["K_N_per_um"] / out["G1"]["exact_N_per_um"]
    out["C1"]["upper_bound_MA_over_t_N_per_um"] = exact_G1(nu) * 1e-6
    out["C2"]["GA_over_t_N_per_um"] = shear_GA_t(nu) * 1e-6
    # C4 through the surface interpolation: penalty at fine sample points inside the disc
    lam, mu, Mmod = lame(E_TARGET, nu)
    fx, uv = fixed_from([(back, 0, 0.), (back, 1, 0.), (back, 2, 0.)])
    for a in (5e-3, 2.5e-3):
        t0 = time.time()
        smp = surface_samples(P, S, lambda qx, qy: qx ** 2 + qy ** 2 <= a ** 2, target=a / 50)
        ids, vals, wts = smp
        rec = dict(n_samples=int(wts.size), sampled_area_over_pi_a2=float(wts.sum() / (np.pi * a * a)))
        for ratio in (1e6, 1e7):
            kappa = ratio * Mmod / (T / (P["cnt"][2] - 1))
            u, F, gap = solve_penalty(K, fx, uv, ids, vals, wts, DELTA, kappa)
            rec[f"K_N_per_um_ratio{ratio:.0e}"] = F / DELTA * 1e-6
            rec[f"max_gap_rel_ratio{ratio:.0e}"] = gap
        rec["K_N_per_um"] = rec["K_N_per_um_ratio1e+07"]
        rec["penalty_sensitivity_rel"] = abs(rec["K_N_per_um_ratio1e+07"] / rec["K_N_per_um_ratio1e+06"] - 1)
        rec["solve_time_s"] = time.time() - t0
        out[f"C4p_a{a*1e3:g}mm"] = rec
    return out


# ------------------------------------------------------------------------------------------------------
def pcg_mujoco(A, Pinv, b, tol=1e-8, maxit=50000, tmax=900.0):
    """mj_effSolve's PCG: x0=0, stop when r'r < tol^2 b'b. Returns (iterations, relres history at 100/1000)."""
    x = np.zeros_like(b); r = b.copy(); bn = b @ b; tl = tol * tol * bn
    z = Pinv @ r; p = z.copy(); rz = r @ z
    rel100 = rel1000 = None; t0 = time.time()
    for it in range(maxit):
        Ap = A @ p; pAp = p @ Ap
        if pAp <= 0:
            return dict(iterations=None, breakdown_at=it + 1, rel100=rel100, rel1000=rel1000)
        al = rz / pAp; x += al * p; r -= al * Ap
        rr = r @ r
        if it + 1 == 100: rel100 = float(np.sqrt(rr / bn))
        if it + 1 == 1000: rel1000 = float(np.sqrt(rr / bn))
        if rr < tl:
            return dict(iterations=it + 1, rel100=rel100 if rel100 is not None else float(np.sqrt(rr / bn)),
                        rel1000=rel1000 if rel1000 is not None else float(np.sqrt(rr / bn)), final_rel=float(np.sqrt(rr / bn)))
        z = Pinv @ r; rzn = r @ z; p = z + (rzn / rz) * p; rz = rzn
        if time.time() - t0 > tmax:
            break
    return dict(iterations=None, capped_at=it + 1, final_rel=float(np.sqrt(rr / bn)), rel100=rel100, rel1000=rel1000)


def eig_ext(A, which):
    """Extreme eigenvalue of a symmetric sparse matrix: Lanczos for the largest, shift-invert at 0 (SuperLU with
    a symmetric ordering) for the smallest."""
    try:
        if which == "max":
            return float(spla.eigsh(A, k=1, which="LA", tol=1e-6, ncv=40, return_eigenvectors=False, maxiter=50000)[0])
        lu = spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        op = spla.LinearOperator(A.shape, matvec=lu.solve, dtype=float)
        mu = spla.eigsh(op, k=1, which="LA", tol=1e-9, ncv=20, return_eigenvectors=False, maxiter=5000)[0]
        return float(1.0 / mu)
    except Exception as e:
        log(f"eig_ext {which} failed: {e!r}")
        return None


def conditioning(c, P, K):
    out = {}
    mp, dp, tl = load(c, pinned=True, gravity=True)
    Pp = points(mp, dp)
    key = {tuple(np.round(x * 1e9).astype(int)): i for i, x in enumerate(P["X"])}
    sel = np.where(Pp["dofadr"] >= 0)[0]
    order = np.argsort(Pp["dofadr"][sel]); sel = sel[order]
    my = np.array([key[tuple(np.round(Pp["X"][i] * 1e9).astype(int))] for i in sel])
    perm_my = (3 * my[:, None] + np.arange(3)).ravel()
    perm_mj = (Pp["dofadr"][sel][:, None] + np.arange(3)).ravel()
    assert np.all(perm_mj == np.arange(mp.nv)), "pinned model has non-flex dofs"
    Kpp = K[perm_my][:, perm_my].tocsr()
    Kmj = efm_stiffness(mp, dp) / H_STEP ** 2
    out["efmK_vs_fd_rel"] = float(sp.linalg.norm(Kmj - Kpp) / sp.linalg.norm(Kpp))
    mvec = np.repeat(Pp["mass"][sel], 3)
    # check: M from MuJoCo is diagonal = body masses
    v = np.random.default_rng(1).standard_normal(mp.nv); res = np.zeros(mp.nv)
    mujoco.mj_mulM(mp, dp, res, v)
    out["M_diag_check_rel"] = float(np.abs(res - mvec * v).max() / np.abs(mvec * v).max())
    h = H_STEP
    A = (sp.diags(mvec) + h * h * Kpp).tocsr()
    n = A.shape[0]
    out.update(nv=int(n), mass_min_kg=float(mvec.min()), mass_max_kg=float(mvec.max()), mass_total_free_kg=float(mvec.sum() / 3))
    t0 = time.time(); log("eigs")
    out["lam_max_A"] = eig_ext(A, "max"); out["lam_min_A"] = eig_ext(A, "min")
    if out["lam_max_A"] and out["lam_min_A"]:
        out["cond_A"] = out["lam_max_A"] / out["lam_min_A"]
    out["lam_max_K_N_per_m"] = eig_ext(Kpp, "max"); out["lam_min_K_N_per_m"] = eig_ext(Kpp, "min")
    if out["lam_max_K_N_per_m"] and out["lam_min_K_N_per_m"]:
        out["cond_K"] = out["lam_max_K_N_per_m"] / out["lam_min_K_N_per_m"]
    Dm = sp.diags(1 / np.sqrt(mvec)); Ks = (Dm @ Kpp @ Dm).tocsr()
    w2max = eig_ext(Ks, "max"); w2min = eig_ext(Ks, "min")
    if w2max and w2min:
        out.update(omega_max_rad_s=float(np.sqrt(w2max)), omega_min_rad_s=float(np.sqrt(w2min)),
                   h_omega_max=float(h * np.sqrt(w2max)), h_omega_min=float(h * np.sqrt(w2min)))
    # block-Jacobi (3x3 vertex blocks of A, as mj_effSolve) preconditioner
    nb = n // 3
    Ad = A.todok() if n < 3000 else None
    blocks = np.zeros((nb, 3, 3))
    Acoo = A.tocoo()
    same = (Acoo.row // 3) == (Acoo.col // 3)
    blocks[Acoo.row[same] // 3, Acoo.row[same] % 3, Acoo.col[same] % 3] = Acoo.data[same]
    Lb = np.linalg.cholesky(blocks)
    Linv = np.linalg.inv(Lb)
    Binv = np.linalg.inv(blocks)
    ii = (np.arange(nb)[:, None, None] * 3 + np.arange(3)[None, :, None]).repeat(3, 2)
    jj = (np.arange(nb)[:, None, None] * 3 + np.arange(3)[None, None, :]).repeat(3, 1)
    LinvM = sp.csr_matrix((Linv.ravel(), (ii.ravel(), jj.ravel())), shape=(n, n))
    Pinv = sp.csr_matrix((Binv.ravel(), (ii.ravel(), jj.ravel())), shape=(n, n))
    Sm = (LinvM @ A @ LinvM.T).tocsr()
    out["lam_max_PA"] = eig_ext(Sm, "max"); out["lam_min_PA"] = eig_ext(Sm, "min")
    if out["lam_max_PA"] and out["lam_min_PA"]:
        out["cond_PA"] = out["lam_max_PA"] / out["lam_min_PA"]
        out["pcg_iter_estimate"] = 0.5 * np.sqrt(out["cond_PA"]) * np.log(2 / 1e-8)
    out["eig_time_s"] = time.time() - t0
    # PCG (mj_effSolve emulation) on three right-hand sides
    b_grav = np.array(dp.qfrc_smooth)        # gravity: -m g on the z dofs
    rng = np.random.default_rng(0); b_rand = rng.standard_normal(n)
    Xs = Pp["X"][sel]; face = np.where(np.isclose(Xs[:, 2], Xs[:, 2].min()))[0]
    ic = face[np.argmin(Xs[face, 0] ** 2 + Xs[face, 1] ** 2)]
    b_pt = np.zeros(n); b_pt[3 * ic + 2] = 1.0
    t0 = time.time(); log("pcg")
    out["pcg"] = {nm: pcg_mujoco(A, Pinv, b) for nm, b in (("gravity", b_grav), ("random", b_rand), ("point_z_centre", b_pt))}
    out["pcg_time_s"] = time.time() - t0
    # same PCG with the mass matrix only as preconditioner: what mj_effSolve does when the flex is not
    # assembled into efm_K (mj_flexSimple false, e.g. any joint upstream of the pinned body): no 3x3 stiffness blocks
    Minv = sp.diags(1.0 / mvec).tocsr()
    t0 = time.time(); log("pcg mass")
    out["pcg_mass_prec"] = {nm: pcg_mujoco(A, Minv, b, maxit=20000, tmax=600.0)
                            for nm, b in (("gravity", b_grav), ("random", b_rand), ("point_z_centre", b_pt))}
    if w2max and w2min:
        out["cond_MinvA"] = (1 + h * h * w2max) / (1 + h * h * w2min)
        out["cond_MinvA_rigid_mode"] = 1 + h * h * w2max       # with an unconstrained rigid mode (omega_min = 0)
        out["pcg_iter_estimate_mass_prec"] = 0.5 * np.sqrt(out["cond_MinvA"]) * np.log(2 / 1e-8)
    out["pcg_mass_prec_time_s"] = time.time() - t0
    # MuJoCo's own PCG (mj_effSolve inside mj_forward, gravity rhs): does it converge within N iterations?
    Ng = out["pcg"]["gravity"].get("iterations")
    mj = {}
    tests = [100] + ([Ng - 1, Ng] if Ng else [])
    log("mujoco pcg check")
    for N in tests:
        if N is None or N < 1:
            continue
        mp.opt.iterations = int(N)
        d2 = mujoco.MjData(mp)
        mujoco.mj_forward(mp, d2)
        mj[str(N)] = int(d2.warning[mujoco.mjtWarning.mjWARN_INERTIA].number) == 0
    out["mujoco_converged_within_N"] = mj
    if Ng:
        out["mujoco_pcg_matches_emulation"] = bool(mj.get(str(Ng), False) and not mj.get(str(Ng - 1), True)) if Ng > 1 else mj.get(str(Ng))
    return out


def run(c, outdir):
    rid = cfg_id(c)
    res = dict(id=rid, config=describe(c), mujoco_version=mujoco.__version__, status="ok")
    t00 = time.time()
    try:
        log("compile"); m, d, tl = load(c, pinned=False); log(f"compiled in {tl:.1f}s, nv={m.nv}")
        P = points(m, d); el = elements(m, P)
        res["model"] = dict(n_points=int(P["n"]), grid_points=P["cnt"].tolist(), nv_unpinned=int(m.nv),
                            n_elements=int(el.shape[0]), compile_time_s=tl)
        K, info = fd_stiffness(m, d, P, el); log(f"FD done {info['fd_time_s']:.1f}s nnz={K.nnz}")
        info.update(k_checks(K, P["X"]))
        dd = res["config"]
        if P["interp"]:
            Ko = k_interp_own(P, el, dd["young_in_Pa"], dd["poisson_in"])
            info["fd_vs_own_hex_assembly_rel"] = float(sp.linalg.norm(K - Ko) / sp.linalg.norm(Ko))
            # and against the TARGET material (interpolated flexes take E, nu directly)
        else:
            Ko = k_full_from_model(m, P, el)
            info["fd_vs_flex_stiffness_assembly_rel"] = float(sp.linalg.norm(K - Ko) / sp.linalg.norm(Ko))
        info["nnz_K"] = int(K.nnz)
        res["fd"] = info
        del m, d
        res["cases"] = static_cases(c, P, K, el)
        log("conditioning"); res["conditioning"] = conditioning(c, P, K); log("done")
    except Exception as e:
        res["status"] = "error"; res["error"] = repr(e); res["traceback"] = traceback.format_exc()
    res["wall_time_s"] = time.time() - t00
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, rid + ".json"), "w") as f:
        json.dump(to_json(res), f, indent=1)
    return res


if __name__ == "__main__":
    c = json.loads(sys.argv[1]); outdir = sys.argv[2]
    r = run(c, outdir)
    print(r["id"], r["status"], "%.1f s" % r["wall_time_s"], r.get("error", ""))
