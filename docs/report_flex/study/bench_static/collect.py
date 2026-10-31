# Merge the per-model outputs into results.json (one record per model), add timing / upstream data and, when the
# ground-truth agents' files exist, error-vs-ground-truth columns; write CSV/markdown tables. Plots: plots.py.
import os, glob, json, math, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
CASES = ["G1", "C1", "C1f", "C2", "C3", "C6ns", "C6f", "C4p_a5mm", "C4p_a2.5mm", "C4n_a5mm", "C4n_a2.5mm", "UP"]


def kval(r, case):
    c = r.get("cases", {}).get(case)
    if c is None:
        return None
    if case == "UP":
        return c.get("K_area_mean_N_per_um")
    return c.get("K_N_per_um")


def load_records():
    recs = []
    for f in sorted(glob.glob(os.path.join(HERE, "out", "*.json"))):
        try:
            recs.append(json.load(open(f)))
        except Exception as e:
            print("bad file", f, e)
    return recs


def load_timing():
    t = {}
    f = os.path.join(HERE, "timing.jsonl")
    if os.path.exists(f):
        for line in open(f):
            try:
                r = json.loads(line); t[r["id"]] = r
            except Exception:
                pass
    return t


# ------------------------------------------------------------------------------------------------------
# ground truth (other agents); adapter tolerant to formats: we look for records carrying a case id, nu and a K
GT_CASE_ALIASES = {"C1": ["C1"], "C1f": ["C1f"], "C2": ["C2"], "C3": ["C3"], "C6ns": ["C6ns", "C6", "C6_noslip"],
                   "C6f": ["C6f", "C6_frictionless"], "C4p_a5mm": ["C4_a5", "C4_a5mm", "C4"], "G1": ["G1"],
                   "C4p_a2.5mm": ["C4_a2.5", "C4_a2.5mm"]}


def _walk(o, path=()):
    if isinstance(o, dict):
        yield path, o
        for k, v in o.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, path + (str(i),))


GT_FEM_MAP = {"C1": "C1", "C1f": "C1f", "C2": "C2", "C3": "C3", "C6": "C6ns", "C6f": "C6f"}
GT_TAXEL = {}   # (case, nu) -> dict(force=[i][j], disp=[i][j]) from gt_fem (their arrays are [j][i][xyz])


def load_gt_fem(out):
    f = os.path.join(STUDY, "gt_fem", "results.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    for gc, mine in GT_FEM_MAP.items():
        for nk, rec in d.get("cases", {}).get(gc, {}).items():
            nu = round(float(nk[2:]), 4)
            if rec.get("K_N_per_um") is None:
                continue
            e = rec.get("K_err_N_per_um"); e = None if (e is None or e != e) else e
            out[(mine, nu)] = (float(rec["K_N_per_um"]), "gt_fem:cases/%s/%s" % (gc, nk), e)
            if rec.get("taxel_force_N_per_um") is not None:
                GT_TAXEL[(mine, nu)] = dict(force=np.array(rec["taxel_force_N_per_um"])[..., 2].T,
                                            disp=np.array(rec["taxel_mean_disp_per_delta"])[..., 2].T)
    for k, rec in d.get("C4_axisymmetric", {}).get("best", {}).items():
        a, fr, nk = k.split("/")
        if fr != "frictionless":
            continue
        nu = round(float(nk[2:]), 4); am = "%gmm" % float(a[1:])
        for mine in ("C4p_a" + am, "C4n_a" + am):
            out[(mine, nu)] = (float(rec["K_N_per_um"]), "gt_fem:C4_axisymmetric/best/" + k, rec.get("K_err_N_per_um"))


def load_gt():
    """Returns {(case, nu): (K_N_per_um, source, err)}; gt_fem (finite-pad FEM) first, then gt_transform (generic)."""
    out = {}
    load_gt_fem(out)
    for name in ("gt_transform",):
        f = os.path.join(STUDY, name, "results.json")
        if not os.path.exists(f):
            continue
        try:
            data = json.load(open(f))
        except Exception as e:
            print("cannot read", f, e); continue
        for path, d in _walk(data):
            case = d.get("case") or d.get("case_id") or d.get("id")
            nu = d.get("nu") if "nu" in d else d.get("poisson")
            if case is None or nu is None:
                continue
            K = None
            for key in ("K_N_per_um", "K_N_um", "K_Npum", "K_n_N_per_um", "K_s_N_per_um", "K_punch_N_per_um", "K"):
                if key in d and isinstance(d[key], (int, float)):
                    K = float(d[key]); break
            if K is None:
                continue
            for mine, al in GT_CASE_ALIASES.items():
                if str(case) in al:
                    key = (mine, round(float(nu), 4))
                    # prefer gt_fem (refined FEM) over transform only if transform does not provide it
                    out.setdefault(key, (K, name + ":" + "/".join(path), None))
    return out


def main():
    recs = load_records()
    timing = load_timing()
    gt = load_gt()
    patch = {}
    if os.path.exists(os.path.join(HERE, "patch.json")):
        patch = {p["id"]: p for p in json.load(open(os.path.join(HERE, "patch.json")))}
    # internal reference: the finest quadratic flex (exact elasticity, 27-node hexes) per (case, nu)
    ref = {}
    for r in recs:
        c = r["config"]
        if r.get("status") != "ok" or c["variant"] != "quadratic":
            continue
        key = (c["nt"], c["cz"])
        for case in CASES:
            v = kval(r, case)
            if v is None:
                continue
            k = (case, c["nu"])
            if k not in ref or key > ref[k][1]:
                ref[k] = (v, key, r["id"])
    out = []
    for r in recs:
        c = r["config"]
        rec = dict(r)
        if r["id"] in timing:
            rec["timing"] = {k: v for k, v in timing[r["id"]].items() if k not in ("id", "config")}
        if r["id"] in patch:
            rec["patch_P0"] = {k: v for k, v in patch[r["id"]].items() if k not in ("id", "config")}
        ev = {}
        for case in CASES:
            v = kval(r, case)
            if v is None:
                continue
            k = (case, c["nu"])
            if k in gt:
                ev[case] = dict(K_gt_N_per_um=gt[k][0], gt_source=gt[k][1], gt_err_N_per_um=gt[k][2],
                                rel_error=v / gt[k][0] - 1)
                tx = r["cases"][case].get("taxel") if case in r.get("cases", {}) else None
                if tx and k in GT_TAXEL:
                    Fm = np.array(tx["force_N_per_um"]); Um = np.array(tx["mean_uz_over_delta"])
                    ev[case]["taxel_force_maxabs_err_N_per_um"] = float(np.abs(Fm - GT_TAXEL[k]["force"]).max())
                    ev[case]["taxel_disp_maxabs_err"] = float(np.abs(Um - GT_TAXEL[k]["disp"]).max())
            elif k in ref:
                ev.setdefault("_internal", {})[case] = dict(K_ref_N_per_um=ref[k][0], ref_model=ref[k][2],
                                                             rel_diff=v / ref[k][0] - 1)
        if case and c["nu"] is not None:
            g1 = r.get("cases", {}).get("G1")
            if g1:
                ev["G1_exact"] = dict(K_exact_N_per_um=g1["exact_N_per_um"], rel_error=g1["ratio_to_exact"] - 1)
        rec["error_vs_reference"] = ev
        out.append(rec)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    # flat CSV
    rows = []
    for r in out:
        c = r["config"]; cd = r.get("conditioning", {})
        row = dict(id=r["id"], variant=c["variant"], dof=c["dof"], nu=c["nu"], nt=c["nt"],
                   elem_dx_mm=round(c["elem_dx_mm"], 4), elem_dz_mm=round(c["elem_dz_mm"], 4),
                   elements_through_thickness=c["elements_through_thickness"],
                   aspect=round(c["aspect_dx_over_dz"], 3), young_in_Pa=c["young_in_Pa"], poisson_in=c["poisson_in"],
                   nv_pinned=cd.get("nv"), status=r.get("status"))
        for case in CASES:
            row["K_" + case + "_N_per_um"] = kval(r, case)
        row["cond_A"] = cd.get("cond_A"); row["cond_PA"] = cd.get("cond_PA")
        row["pcg_iters_blockjacobi_gravity"] = cd.get("pcg", {}).get("gravity", {}).get("iterations")
        row["pcg_iters_blockjacobi_random"] = cd.get("pcg", {}).get("random", {}).get("iterations")
        row["pcg_iters_massprec_gravity"] = cd.get("pcg_mass_prec", {}).get("gravity", {}).get("iterations")
        row["pcg_iters_massprec_random"] = cd.get("pcg_mass_prec", {}).get("random", {}).get("iterations")
        row["h_omega_max"] = cd.get("h_omega_max"); row["h_omega_min"] = cd.get("h_omega_min")
        row["mujoco_matches_pcg_emulation"] = cd.get("mujoco_pcg_matches_emulation")
        row["mujoco_converged_in_100"] = cd.get("mujoco_converged_within_N", {}).get("100")
        t = r.get("timing", {})
        row["compile_s"] = t.get("compile_s"); row["step_ms"] = 1e3 * t["step_s_median"] if t.get("step_s_median") else None
        row["fd_vs_assembly_rel"] = r.get("fd", {}).get("fd_vs_flex_stiffness_assembly_rel", r.get("fd", {}).get("fd_vs_own_hex_assembly_rel"))
        row["efmK_vs_fd_rel"] = cd.get("efmK_vs_fd_rel")
        for pk in ("uniaxial_strain_zz", "simple_shear_xy", "hydrostatic"):
            row["P0_" + pk + "_energy_ratio"] = r.get("patch_P0", {}).get(pk, {}).get("energy_ratio")
        for case in CASES:
            e = r.get("error_vs_reference", {}).get(case)
            row["err_" + case + "_vs_gt"] = e.get("rel_error") if e else None
        row["sym_rel"] = r.get("fd", {}).get("sym_maxabs_rel"); row["null_rel"] = r.get("fd", {}).get("nullspace_rel_max")
        rows.append(row)
    keys = list(rows[0].keys()) if rows else []
    with open(os.path.join(HERE, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(len(out), "records;", sum(r.get("status") == "ok" for r in out), "ok; GT entries:", len(gt))
    for k, v in sorted(gt.items()):
        print("  GT", k, v)
    for k, v in sorted(ref.items()):
        print("  internal ref", k, "%.5f" % v[0], v[2])


if __name__ == "__main__":
    main()
