"""postprocess.py -- Richardson extrapolation, results.json, plots (reads runs/, runs_c4/, ../gt_transform if present).

Richardson: levels m (h ~ 1/m, nested).  With three levels (m1 < m2 < m3) the observed order p solves
(Q1 - Q2)/(Q2 - Q3) = (m1^-p - m2^-p)/(m2^-p - m3^-p); Q* = Q3 - (Q2 - Q3) m3^-p / (m2^-p - m3^-p).
Error bar (GCI-type, Roache, safety factor 1.25): err = 1.25 |Q* - Q3|, floored at 1e-12 |Q*|.
If the three values are not monotone (or p outside [0.5, 10]): Q* = Q3, err = 2 |Q3 - Q2|, flag "non-monotone".
Best value per (case, nu): full form if available (nu 0.45, 0.49), else mixed; the error bar is
max(GCI, |Q*_full - Q*_mixed|) when both forms exist (two different discretisations, same limit).
"""
import os, json, glob
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
A, B, T = 25.5, 35.5, 1.5
AREA = 51.0 * 71.0
XT = -25.5 + 10.2 * np.arange(6)
YT = -35.5 + (71.0 / 7.0) * np.arange(8)
LOADC = {"C1": 2, "C1f": 2, "C2": 1, "C3": 2, "C6": 2, "C6f": 2, "G1": 2}
NUS = (0.45, 0.49, 0.499)


def moduli(nu, E=1.0):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    return lam, mu, lam + 2 * mu, lam + 2 * mu / 3          # lam, G, M, K


def richardson(ms, qs):
    ms, qs = np.asarray(ms, float), np.asarray(qs, float)
    if len(ms) < 2:
        return dict(value=float(qs[-1]), err=float("nan"), p=None, flag="single level")
    if len(ms) == 2:
        return dict(value=float(qs[-1]), err=float(abs(qs[1] - qs[0])), p=None, flag="two levels only (finest value, err = abs(Q2-Q1))")
    m1, m2, m3 = ms[-3:]; q1, q2, q3 = qs[-3:]
    d1, d2 = q1 - q2, q2 - q3
    scale = max(abs(q3), 1e-300)
    if abs(d2) < 1e-13 * scale and abs(d1) < 1e-13 * scale:
        return dict(value=float(q3), err=float(1e-12 * scale), p=None, flag="converged to round-off")
    if d1 * d2 <= 0:
        return dict(value=float(q3), err=float(2 * max(abs(d2), 1e-12 * scale)), p=None, flag="non-monotone")
    f = lambda p: (m1 ** -p - m2 ** -p) / (m2 ** -p - m3 ** -p) - d1 / d2
    try:
        p = brentq(f, 0.05, 30.0)
    except ValueError:
        return dict(value=float(q3), err=float(2 * abs(d2)), p=None, flag="no order found")
    qs_ = q3 - d2 * m3 ** -p / (m2 ** -p - m3 ** -p)
    flag = "ok" if 0.5 <= p <= 10 else "order outside [0.5,10]"
    if flag != "ok":
        return dict(value=float(q3), err=float(2 * abs(d2)), p=float(p), flag=flag)
    return dict(value=float(qs_), err=float(max(1.25 * abs(qs_ - q3), 1e-12 * abs(qs_))), p=float(p), flag=flag)


def load_runs():
    R = {}
    for f in glob.glob(os.path.join(HERE, "runs", "*.json")):
        d = json.load(open(f))
        R.setdefault((d["case"], d["form"], round(d["nu"], 3)), {})[d["m"]] = d
    return R


def pressure_method_K(nu, a=A, b=B, t=T, nterm=4001, E=1.0):
    """Classical compressible pressure method (rigid bonded faces): lap p - lam^2 p = -12 G eps/t^2, p = 0 on the
    rectangle edges, sigma_zz = -p (Kelly & Van Engelen, PEER 2015/03, sec. 2.1, with rigid reinforcement).
    Series in cos(n pi x / 2a) (own derivation, checked against a finite-difference solve below).
    Returns K = F/delta in N/mm for the 2a x 2b pad."""
    lam_, G, M, Kb = moduli(nu, E)
    l2 = 12 * G / (Kb * t ** 2)
    c = 12 * G / t ** 2 / t                                   # per unit delta (eps = delta/t)
    n = np.arange(1, 2 * nterm, 2, dtype=float)
    al = n * np.pi / (2 * a)
    k = np.sqrt(al ** 2 + l2)
    return float(np.sum(8 * c / (n * np.pi * al * k ** 2) * (2 * b - 2 * np.tanh(k * b) / k)))


def pressure_method_fd(nu, h=0.05, E=1.0):
    """Finite-difference check of pressure_method_K (5-point Laplacian, quarter domain with symmetry)."""
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    lam_, G, M, Kb = moduli(nu, E)
    l2 = 12 * G / (Kb * T ** 2); c = 12 * G / T ** 3
    nx, ny = int(round(A / h)), int(round(B / h))
    def lap1(n):                                   # nodes 0..n-1 (x = 0..A-h), Neumann at 0, Dirichlet at A
        L = sp.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]).tolil()
        L[0, 1] = 2.0
        return L.tocsr() / h ** 2
    Lx, Ly = lap1(nx), lap1(ny)
    Lap = sp.kron(Lx, sp.eye(ny)) + sp.kron(sp.eye(nx), Ly)
    p = spla.spsolve((Lap - l2 * sp.eye(nx * ny)).tocsc(), -c * np.ones(nx * ny))
    wx = np.full(nx, h); wx[0] = h / 2; wy = np.full(ny, h); wy[0] = h / 2
    return float(4 * np.einsum("i,j,ij->", wx, wy, p.reshape(nx, ny)))


def gt_transform_results():
    """Return the gt_transform results file if it exists (any *.json except validate.json)."""
    d = os.path.join(HERE, "..", "gt_transform")
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f) in ("validate.json",):
            continue
        try:
            out[os.path.basename(f)] = json.load(open(f))
        except Exception:
            pass
    return out


def main():
    R = load_runs()
    res = dict(
        description="Ground-truth FEM for the finite 51 x 71 x 1.5 mm bonded PDMS pad (E = 1 MPa, linear elastic). "
                    "Imposed displacement delta of a rigid platen/punch; all forces per unit delta. "
                    "Frame: origin at the centre of the free face, x across (|x|<=25.5 mm), y along the finger "
                    "(|y|<=35.5 mm), +z into the layer (free face z=0, bonded face z=1.5 mm). "
                    "Force = force exerted by the platen ON the layer (+z = pressing into the skin). "
                    "Taxel arrays are [j][i] (j = row along y, 0..6 from y=-35.5; i = column along x, 0..4 from x=-25.5).",
        accuracy_rule="value = Richardson extrapolation over nested levels m (h ~ 1/m); err = 1.25|Q* - Q_finest| "
                      "(GCI), max'd with |full - mixed| when both forms exist. Taxel/profile arrays: finest level, "
                      "with per-entry err = |Q_m3 - Q_m2| * max(1, 1.25/(r^p - 1)) using p of the global stiffness.",
        scripts=dict(fem="fem_layer.py", cases="cases.py", sweep="run_cases.py", post="postprocess.py",
                     verify="verify_fem.py -> verify_fem.json", c4="axi_fem.py, run_c4.py -> runs_c4/, c4_verify.json"),
        cases={})
    for case in ("C1", "C1f", "C2", "C3", "C6", "C6f"):
        cres = {}
        for nu in NUS:
            per_form = {}
            for form in ("full", "mixed"):
                runs = R.get((case, form, nu))
                if not runs:
                    continue
                ms = sorted(runs)
                lc = LOADC[case]
                Kq = [runs[m]["F_platen_N_per_mm"][lc] / 1e3 for m in ms]
                KE = [runs[m]["K_energy_N_per_mm"] / 1e3 for m in ms]
                rK = richardson(ms, Kq)
                comps = [richardson(ms, [runs[m]["F_platen_N_per_mm"][c] / 1e3 for m in ms]) for c in range(3)]
                per_form[form] = dict(levels=ms, n_dof=[runs[m]["n_dof"] for m in ms],
                                      K_N_per_um_levels=Kq, K_energy_N_per_um_levels=KE, K=rK, F_xyz=comps,
                                      t_total_s=[runs[m]["t_total_s"] for m in ms],
                                      maxrss_GB=[runs[m].get("maxrss_GB") for m in ms])
            if not per_form:
                continue
            best = "full" if "full" in per_form else "mixed"
            pf = per_form[best]
            K_err = pf["K"]["err"]
            cross = None
            if "full" in per_form and "mixed" in per_form:
                cross = abs(per_form["full"]["K"]["value"] - per_form["mixed"]["K"]["value"])
                K_err = max(K_err, cross)
            runs = R[(case, best, nu)]
            ms = sorted(runs)
            fin = runs[ms[-1]]
            p = pf["K"]["p"]
            fac = max(1.0, 1.25 / ((ms[-1] / ms[-2]) ** p - 1)) if (p and len(ms) >= 2) else 1.0

            def arr(key, scale=1.0):
                a3 = np.asarray(fin[key]) * scale
                if len(ms) >= 2:
                    a2 = np.asarray(runs[ms[-2]][key]) * scale
                    return a3.tolist(), (np.abs(a3 - a2) * fac).tolist()
                return a3.tolist(), None

            tf, tfe = arr("taxel_force_N_per_mm", 1e-3)
            tb, tbe = arr("taxel_force_on_blade_N_per_mm", 1e-3)
            td, tde = arr("taxel_mean_disp_per_delta")
            # profile convergence on the coarsest-level nodes
            prof_err = {}
            if len(ms) >= 2:
                for key, ck in (("u_free_at_y0_per_delta", "profile_x_mm"), ("u_free_at_x0_per_delta", "profile_y_mm")):
                    x3 = np.asarray(fin[ck]); x2 = np.asarray(runs[ms[-2]][ck])
                    i3 = np.searchsorted(x3, x2); ok = (i3 < x3.size) & (np.abs(x3[np.minimum(i3, x3.size - 1)] - x2) < 1e-9)
                    d = np.asarray(fin[key])[i3[ok]] - np.asarray(runs[ms[-2]][key])[ok]
                    prof_err[key] = float(np.max(np.abs(d)) * fac)
            lam_, G, M, Kb = moduli(nu)
            entry = dict(
                best_form=best, K_N_per_um=pf["K"]["value"], K_err_N_per_um=K_err, K_p_obs=p, K_flag=pf["K"]["flag"],
                K_full_minus_mixed_N_per_um=cross,
                F_xyz_N_per_um=[c["value"] for c in pf["F_xyz"]], F_xyz_err_N_per_um=[c["err"] for c in pf["F_xyz"]],
                F_on_blade_xyz_N_per_um=(np.asarray(fin["F_on_blade_N_per_mm"]) / 1e3).tolist(),
                taxel_force_N_per_um=tf, taxel_force_err_N_per_um=tfe,
                taxel_force_on_blade_N_per_um=tb, taxel_force_on_blade_err_N_per_um=tbe,
                taxel_mean_disp_per_delta=td, taxel_mean_disp_err=tde,
                profile_x_mm=fin["profile_x_mm"], profile_y_mm=fin["profile_y_mm"],
                u_free_at_y0_per_delta=fin["u_free_at_y0_per_delta"], u_free_at_x0_per_delta=fin["u_free_at_x0_per_delta"],
                profile_err_max=prof_err,
                t_free_at_y0_MPa_per_mm=fin["t_free_at_y0_MPa_per_mm"], t_free_at_x0_MPa_per_mm=fin["t_free_at_x0_MPa_per_mm"],
                finest_level=ms[-1], finest_n_dof=fin["n_dof"], forms=per_form,
                upper_bound_MA_over_t_N_per_um=M * AREA / T / 1e3, GA_over_t_N_per_um=G * AREA / T / 1e3)
            cres["nu%.3f" % nu] = entry
        res["cases"][case] = cres
    # thin-layer (pressure method) for C1
    pm = {}
    for nu in NUS:
        Kpm = pressure_method_K(nu) / 1e3
        fd = pressure_method_fd(nu, h=0.025) / 1e3
        e = res["cases"]["C1"].get("nu%.3f" % nu)
        lam_, G, M, Kb = moduli(nu)
        pm["nu%.3f" % nu] = dict(K_pressure_method_N_per_um=Kpm, fd_check_N_per_um=fd,
                                 K_bulk_A_over_t_N_per_um=Kb * AREA / T / 1e3,
                                 K_confined_MA_over_t_N_per_um=M * AREA / T / 1e3,
                                 rel_err_vs_FEM=(Kpm / e["K_N_per_um"] - 1) if e else None,
                                 winkler_MA_over_t_rel_err_vs_FEM=(M * AREA / T / 1e3 / e["K_N_per_um"] - 1) if e else None)
    res["C1_pressure_method"] = dict(
        source="J.M. Kelly, N.C. Van Engelen, 'Single series solution for the rectangular fiber-reinforced elastomeric "
               "isolator compression modulus', PEER Report 2015/03, sec. 2.1 eqs. (2.1)-(2.11) (parabolic bulging, "
               "sigma_xx=sigma_yy=sigma_zz=-p, compressibility tr eps = -p/K), here with rigid bonded faces "
               "(u1 = v1 = 0); https://peer.berkeley.edu/sites/default/files/webpeer-2015-03_james_m._kelly_and_niel_c._van_engelen_.pdf",
        equation="lap p - (12 G/(K t^2)) p = -12 G (delta/t)/t^2, p = 0 on the pad edges, F = int p dA",
        values=pm)
    # C4 (axisymmetric)
    c4 = {}
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "runs_c4", "*.json"))]
    groups = {}
    for d in rows:
        groups.setdefault((d["a_mm"], d["R_out_mm"], d["bc"], d["form"], round(d["nu"], 3)), {})[d["m"]] = d
    for key, g in sorted(groups.items()):
        ms = sorted(g)
        r = richardson(ms, [g[m]["K_N_per_mm"] / 1e3 for m in ms])
        c4["a%.1f/R%.1f/%s/%s/nu%.3f" % key] = dict(K_N_per_um=r["value"], K_err=r["err"], p_obs=r["p"], flag=r["flag"],
                                                    levels=ms, K_levels=[g[m]["K_N_per_mm"] / 1e3 for m in ms],
                                                    n_dof=[g[m]["n_dof"] for m in ms])
    res["C4_axisymmetric"] = dict(note="disc of radius R_out (25.5 mm = inscribed circle of the pad; 51 mm for the "
                                       "finite-size check), clamped back face, free lateral face; axi_fem.py", runs=c4)
    best4 = {}
    for a in (5.0, 2.5):
        for bc in ("frictionless", "noslip"):
            for nu in NUS:
                kf = c4.get("a%.1f/R25.5/%s/full/nu%.3f" % (a, bc, nu))
                km = c4.get("a%.1f/R25.5/%s/mixed/nu%.3f" % (a, bc, nu))
                k51 = c4.get("a%.1f/R51.0/%s/%s/nu%.3f" % (a, bc, "full" if nu < 0.499 else "mixed", nu))
                b = kf or km
                if not b:
                    continue
                err = b["K_err"]
                if kf and km:
                    err = max(err, abs(kf["K_N_per_um"] - km["K_N_per_um"]))
                best4["a%.1f/%s/nu%.3f" % (a, bc, nu)] = dict(
                    K_N_per_um=b["K_N_per_um"], K_err_N_per_um=err, form="full" if kf else "mixed",
                    finite_size_rel_R25p5_vs_R51=(b["K_N_per_um"] / k51["K_N_per_um"] - 1) if k51 else None)
    res["C4_axisymmetric"]["best"] = best4
    res["gt_transform_files_found"] = list(gt_transform_results().keys())
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    return res



def compare_transform(res):
    """Compare C4 (axisymmetric FEM) and C6/C6f (3D finite-pad FEM) with the exact infinite-layer solutions of
    ../gt_transform/results.json (if present).  Adds res['comparison_gt_transform'] and res['code_verification']."""
    out = {}
    f = os.path.join(HERE, "..", "gt_transform", "results.json")
    if os.path.exists(f):
        tr = json.load(open(f))
        out["source"] = "../gt_transform/results.json (mtime %s)" % __import__("time").ctime(os.path.getmtime(f))
        c4 = {}
        for a in (5.0, 2.5):
            for bc, tbc in (("frictionless", "frictionless"), ("noslip", "bonded_noslip")):
                for nu in NUS:
                    mine = res["C4_axisymmetric"]["best"].get("a%.1f/%s/nu%.3f" % (a, bc, nu))
                    try:
                        t = tr["nu=%g" % nu]["C4"]["a=%gmm" % a][tbc]
                    except KeyError:
                        continue
                    if not mine:
                        continue
                    Kt = t["K_N_per_um"]; Kt_err = abs(t.get("rel_err_est", 0.0) * Kt)
                    c4["a%.1f/%s/nu%.3f" % (a, bc, nu)] = dict(
                        K_fem_axi_N_per_um=mine["K_N_per_um"], K_fem_err=mine["K_err_N_per_um"],
                        K_transform_N_per_um=Kt, K_transform_err=Kt_err, rel_diff=mine["K_N_per_um"] / Kt - 1,
                        diff_over_combined_err=abs(mine["K_N_per_um"] - Kt) / max(mine["K_err_N_per_um"] + Kt_err, 1e-300),
                        finite_size_rel_R25p5_vs_R51=mine["finite_size_rel_R25p5_vs_R51"])
        out["C4"] = c4
        c6 = {}
        for case, kind in (("C6", "noslip"), ("C6f", "frictionless")):
            for nu in NUS:
                e = res["cases"].get(case, {}).get("nu%.3f" % nu)
                try:
                    t = tr["nu=%g" % nu]["C6"][kind]
                except KeyError:
                    continue
                if not e:
                    continue
                Kt = t["K_N_per_um"]; Kt_err = abs(t["rel_err_est"] * Kt)
                d = dict(K_fem_N_per_um=e["K_N_per_um"], K_fem_err=e["K_err_N_per_um"], K_transform_N_per_um=Kt,
                         K_transform_err=Kt_err, rel_diff=e["K_N_per_um"] / Kt - 1,
                         diff_over_combined_err=abs(e["K_N_per_um"] - Kt) / (e["K_err_N_per_um"] + Kt_err))
                try:
                    tm = t["taxel_maps"]
                    bf = np.asarray(e["taxel_force_on_blade_N_per_um"]) / e["F_on_blade_xyz_N_per_um"][2]
                    bm = np.transpose(np.asarray(tm["base_reaction_fraction_of_total"]), (1, 0, 2))
                    uf = np.asarray(e["taxel_mean_disp_per_delta"])
                    um = np.transpose(np.asarray(tm["mean_surface_displacement_per_delta"]), (1, 0, 2))
                    d.update(base_force_fraction_maxabs_diff=float(np.max(np.abs(bf - bm))),
                             base_Fz_fraction_centre_fem_transform=[float(bf[3, 2, 2]), float(bm[3, 2, 2])],
                             taxel_mean_disp_per_delta_maxabs_diff=float(np.max(np.abs(uf - um))),
                             mean_uz_x_neighbour_fem_transform=[float(uf[3, 1, 2]), float(um[3, 1, 2])],
                             mean_uz_y_neighbour_fem_transform=[float(uf[4, 2, 2]), float(um[4, 2, 2])])
                except (KeyError, ValueError, IndexError) as ex:
                    d["map_compare_error"] = repr(ex)
                c6["%s/nu%.3f" % (case, nu)] = d
        out["C6"] = c6
    res["comparison_gt_transform"] = out
    v = json.load(open(os.path.join(HERE, "verify_fem.json")))
    cv = {}
    cv["V1_kron_vs_elementwise_max_rel_diff"] = max(v["V1_kron_vs_elementwise_max_rel_diff"].values())
    p = v["V2_patch_tests"]
    cv["P0_patch_max_interior_rel_err"] = max(x["max_interior_rel_err"] for x in p.values())
    cv["P0_patch_max_abs_energy_ratio_minus_1"] = max(abs(x["energy_ratio"] - 1) for x in p.values())
    cv["P0_n_tests"] = len(p)
    cv["G1_confined_max_abs_rel_err"] = max(abs(x["rel_err"]) for x in v["V3_G1_confined"].values())
    cv["V5_nd_vs_superlu_max_rel_diff_U"] = max(x["rel_diff_U"] for x in v["V5_solver_nd_vs_superlu"].values())
    cv["V6_half_vs_full_model_max_rel_diff_F"] = {k: x["max_abs_diff_F_rel"] for k, x in v["V6_symmetry_reduction"].items()}
    for key in ("V4_mms", "V4b_mms_solenoidal"):
        if key in v:
            cv[key + "_finest_rates"] = {k: dict(L2=r[-1].get("rate_L2"), H1=r[-1].get("rate_H1"), L2_err=r[-1]["L2"])
                                         for k, r in v[key].items()}
    try:
        c4v = json.load(open(os.path.join(HERE, "c4_verify.json")))
        cv["C4_axi_patch_max_nodal_err"] = max(x.get("max_nodal_err", 0.0) for x in c4v.values() if isinstance(x, dict))
    except Exception as ex:
        cv["C4_axi_verify_error"] = repr(ex)
    cv["source"] = "verify_fem.json (verify_fem.py), c4_verify.json (run_c4.py)"
    res["code_verification"] = cv
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    return res


def summary_md(res):
    L = ["| case | nu | K (N/um) | err (N/um) | p_obs | form | finest DOF | K_full - K_mixed | other F comps (N/um) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for case, c in res["cases"].items():
        for nu, e in c.items():
            lc = LOADC[case]
            oth = ", ".join("F%s=%.4g" % ("xyz"[k], e["F_xyz_N_per_um"][k]) for k in range(3)
                            if k != lc and abs(e["F_xyz_N_per_um"][k]) > 1e-9 * abs(e["K_N_per_um"]))
            L.append("| %s | %s | %.6g | %.1e | %s | %s | %d | %s | %s |" % (
                case, nu[2:], e["K_N_per_um"], e["K_err_N_per_um"], "%.2f" % e["K_p_obs"] if e["K_p_obs"] else e["K_flag"],
                e["best_form"], e["finest_n_dof"], "%.1e" % e["K_full_minus_mixed_N_per_um"]
                if e["K_full_minus_mixed_N_per_um"] is not None else "-", oth or "-"))
    L += ["", "| C1 nu | FEM K (N/um) | pressure method (N/um) | rel. err | K_bulk A/t | M A/t (upper bound; Winkler) | rel. err of M A/t |",
          "|---|---|---|---|---|---|---|"]
    for nu, v in res["C1_pressure_method"]["values"].items():
        e = res["cases"]["C1"].get(nu)
        L.append("| %s | %s | %.4f | %s | %.4f | %.4f | %s |" % (
            nu[2:], "%.6g" % e["K_N_per_um"] if e else "-", v["K_pressure_method_N_per_um"],
            "%+.2f%%" % (100 * v["rel_err_vs_FEM"]) if v["rel_err_vs_FEM"] is not None else "-",
            v["K_bulk_A_over_t_N_per_um"], v["K_confined_MA_over_t_N_per_um"],
            "%+.2f%%" % (100 * v["winkler_MA_over_t_rel_err_vs_FEM"]) if v["winkler_MA_over_t_rel_err_vs_FEM"] is not None else "-"))
    L += ["", "| C4 punch | K (N/um) | err (N/um) | form | finite-size effect K(R=25.5)/K(R=51) - 1 |", "|---|---|---|---|---|"]
    for k, v in res["C4_axisymmetric"]["best"].items():
        L.append("| %s | %.6g | %.1e | %s | %s |" % (k, v["K_N_per_um"], v["K_err_N_per_um"], v["form"],
                 "%.1e" % v["finite_size_rel_R25p5_vs_R51"] if v["finite_size_rel_R25p5_vs_R51"] is not None else "-"))
    open(os.path.join(HERE, "summary_table.md"), "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    res = compare_transform(main())
    summary_md(res)
    for case, c in res["cases"].items():
        for nu, e in c.items():
            print("%-4s %s K = %.6f +- %.1e N/um (p=%s, %s, form %s, finest dof %d)  bound MA/t=%.3f" % (
                case, nu, e["K_N_per_um"], e["K_err_N_per_um"], "%.2f" % e["K_p_obs"] if e["K_p_obs"] else "-",
                e["K_flag"], e["best_form"], e["finest_n_dof"], e["upper_bound_MA_over_t_N_per_um"]))
    for k, v in res["C1_pressure_method"]["values"].items():
        print("pressure method", k, v)
    for k, v in res["C4_axisymmetric"]["best"].items():
        print("C4", k, v)
