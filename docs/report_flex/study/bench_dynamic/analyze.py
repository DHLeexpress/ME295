"""Aggregate runs/*.json (+ .npz) into results.json (one record per run, units in keys), tables and plots.

Usage: python analyze.py            (writes results.json, tables/*.md|csv, plots/*.png)
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A, B, T = 0.051, 0.071, 0.0015
GT_FILES = [os.path.join(HERE, "..", "gt_transform", "results.json"), os.path.join(HERE, "..", "gt_fem", "results.json")]


def moduli(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    return lam + 2 * G, G


def bounds(nu, E=1e6):
    M, G = moduli(E, nu)
    return dict(C1_upper_N_per_um=M * A * B / T * 1e-6, C2_upper_N_per_um=G * A * B / T * 1e-6)


def load_runs():
    recs = []
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "*.json"))):
        try:
            recs.append(json.load(open(f)))
        except Exception:  # noqa
            pass
    return recs


def ts_of(rid):
    p = os.path.join(HERE, "runs", rid + ".npz")
    return np.load(p) if os.path.exists(p) else None


def derive(r):
    """Derived metrics of one run (SI units, unit in each key)."""
    c = r["cfg"]
    out = {}
    cx, cy, cz = c["cells"]
    k = 2 if c["variant"] == "quadratic" else 1
    out["cell_dx_mm"] = 51.0 / cx
    out["cell_dy_mm"] = 71.0 / cy
    out["cell_dz_mm"] = 1.5 / cz
    out["node_spacing_dx_mm"] = 51.0 / (k * cx)
    ph = r.get("phases") or []
    if not ph:
        return out
    case = c["case"]

    def flat(p, comp=2, d_idx=2):
        dlt = p["ind_disp_m"][d_idx]
        F = p["F_mean_N"][comp]
        u = p["uz_foot_mean_m"]
        res = dict(delta_m=dlt, F_main_N=F, K_app_N_per_um=F / dlt * 1e-6 if dlt else None,
                   u_layer_foot_mean_m=u, u_layer_centre_m=p["uz_centre_m"],
                   K_layer_N_per_um=(F / u * 1e-6) if abs(u) > 1e-12 else None,
                   penetration_frac_foot=(1 - u / dlt) if dlt else None,
                   pen_max_m=p["pen_max_m"], pen_forceweighted_m=p["pen_forceweighted_m"],
                   chatter_rel=p["chatter_rel"], chatter_std_N=p["F_std_N"][comp],
                   F_peak_to_peak_N=p["F_max_N"][comp] - p["F_min_N"][comp],
                   settle_1pct_s=p["settle_1pct_s"], settle_5pct_s=p["settle_5pct_s"], drift_rel=p.get("drift_rel"),
                   ncon=p["ncon"], ncon_min=p["ncon_min"], ncon_max=p["ncon_max"], nefc=p["nefc"],
                   ncon_loaded=p["ncon_loaded"], niter_median=p["niter_median"], niter_max=p["niter_max"],
                   solver_nnz_median=p.get("solver_nnz_median"), pinned_weight_share=p.get("pinned_weight_share"),
                   F_vector_N=p["F_mean_N"])
        return res

    if case in ("C1", "C1f", "C4", "C6", "C6f"):
        out.update(flat(ph[-1]))
    if case in ("C6", "C6f"):
        p = ph[-1]
        tf = np.array(p["taxel_force_N"])[:, :, 2]
        tu = np.array(p["taxel_disp_m"])[:, :, 2]
        nb = [(3, 1), (3, 3), (2, 2), (4, 2)]          # (j, i) edge neighbours of the centre taxel (j=3, i=2)
        cn = [(2, 1), (2, 3), (4, 1), (4, 3)]          # corner neighbours
        tot = tf.sum()
        out["C6_centre_taxel_force_N"] = float(tf[3, 2])
        out["C6_centre_taxel_force_frac"] = float(tf[3, 2] / tot) if tot else None
        out["C6_edge_neighbour_force_frac_each"] = [float(tf[j, i] / tot) if tot else None for j, i in nb]
        out["C6_centre_taxel_uz_m"] = float(tu[3, 2])
        out["C6_edge_neighbour_uz_rel"] = [float(tu[j, i] / tu[3, 2]) if tu[3, 2] else None for j, i in nb]
        out["C6_corner_neighbour_uz_rel"] = [float(tu[j, i] / tu[3, 2]) if tu[3, 2] else None for j, i in cn]
    if case == "C2":
        pp = [p for p in ph if p["label"] == "press"]
        ps = [p for p in ph if p["label"] == "shear"]
        if pp:
            out["press"] = flat(pp[0])
        if pp and ps:
            p0, p1 = pp[0], ps[0]
            dy = p1["ind_disp_m"][1] - p0["ind_disp_m"][1]
            dF = p1["F_mean_N"][1] - p0["F_mean_N"][1]
            du = p1["uy_foot_mean_m"] - p0["uy_foot_mean_m"]
            out.update(delta_y_m=dy, Fy_shear_minus_press_N=dF, Ks_app_end_N_per_um=dF / dy * 1e-6,
                       uy_layer_shear_minus_press_m=du, slip_end_m=dy - du,
                       chatter_rel=p1["chatter_rel"], chatter_std_N=p1["F_std_N"][1],
                       settle_1pct_s=p1["settle_1pct_s"], settle_5pct_s=p1["settle_5pct_s"],
                       drift_rel=p1.get("drift_rel"), ncon=p1["ncon"], nefc=p1["nefc"],
                       niter_median=p1["niter_median"], niter_max=p1["niter_max"],
                       K_app_N_per_um=dF / dy * 1e-6)
            ts = ts_of(r["id"])
            if ts is not None and len(ts["F"]) >= p1["end_step"]:
                # early value: mean over the 10 steps after the shear ramp; creep: linear fit over the hold
                nh = int(round(c["hold_s"] / c["h_s"]))
                nr = int(round(c["ramp_s"] / c["h_s"]))
                e = p1["end_step"]
                s0 = e - nh
                early = ts["F"][s0:s0 + 10, 1].mean()
                Fy0 = p0["F_mean_N"][1]
                out["Ks_app_early_N_per_um"] = (early - Fy0) / dy * 1e-6
                tt = ts["t"][s0:e]
                yy = ts["F"][s0:e, 1]
                if len(tt) > 20:
                    slope = np.polyfit(tt, yy, 1)[0]
                    out["Fy_creep_rate_rel_per_s"] = float(slope / max(1e-12, abs(early - Fy0)))
                uy = ts["uy_foot"][s0:e]
                out["uy_layer_early_m"] = float(uy[:10].mean() - ts["uy_foot"][s0 - nr - 1])
    if case == "C5":
        lv = [p for p in ph if p["label"].startswith("lvl")]
        out["C5_delta_m"] = [p["ind_disp_m"][2] for p in lv]
        out["C5_F_N"] = [p["F_mean_N"][2] for p in lv]
        out["C5_contact_radius_max_m"] = [p["contact_radius_max_m"] for p in lv]
        out["C5_contact_radius_rms_fw_m"] = [p["contact_radius_rms_fw_m"] for p in lv]
        out["C5_u_centre_m"] = [p["uz_centre_m"] for p in lv]
        out["C5_ncon"] = [p["ncon"] for p in lv]
        if lv:
            p = lv[-1]
            out.update(flat(p))
            out["penetration_frac_centre"] = 1 - p["uz_centre_m"] / p["ind_disp_m"][2]
            out["F_at_30um_N"] = p["F_mean_N"][2] * 30e-6 / p["ind_disp_m"][2]
    return out


# ------------------------------------------------------------------------------------------- ground truth
def load_gt():
    """Return {(case, nu): {"K_N_per_um": ..., "src": ...}} from the gt_* results.json files, if present.
    The file formats are not fixed in advance: this reader accepts records with a case id, nu and a stiffness."""
    gt = {}
    for f in GT_FILES:
        if not os.path.exists(f):
            continue
        try:
            data = json.load(open(f))
        except Exception:  # noqa
            continue
        gt.update(parse_gt(data, os.path.relpath(f, HERE)))
    return gt


def parse_gt(data, src):
    """Best-effort parser (adapted after inspecting the files; see README)."""
    out = {}
    recs = data if isinstance(data, list) else data.get("records") or data.get("results") or []
    if isinstance(recs, dict):
        recs = [dict(v, case=k) if isinstance(v, dict) else dict(case=k, value=v) for k, v in recs.items()]
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        case = rec.get("case") or rec.get("case_id") or rec.get("id")
        nu = rec.get("nu") or rec.get("poisson")
        K = None
        for key in ("K_N_per_um", "K_n_N_per_um", "K_s_N_per_um", "K_punch_N_per_um", "stiffness_N_per_um",
                    "K_extrap_N_per_um", "K_best_N_per_um"):
            if rec.get(key) is not None:
                K = rec[key]
                break
        if case and nu is not None and K is not None:
            out[(str(case), round(float(nu), 4))] = dict(K_N_per_um=float(K), src=src, raw=rec)
    return out


def main():
    recs = load_runs()
    gt = load_gt()
    results = []
    for r in recs:
        d = derive(r) if r.get("status") in ("ok", "diverged") else {}
        c = r["cfg"]
        rec = dict(id=r["id"], status=r["status"], group=c.get("tag"), config=c,
                   young_in_Pa=r.get("young_in_Pa"), poisson_in=r.get("poisson_in"), vertex_grid=r.get("verts"),
                   nv=r.get("nv"), nflexvert=r.get("nflexvert"), nelem=r.get("nelem"),
                   nelem_active=r.get("nelem_active"), vertex_mass_kg=r.get("vertex_mass_kg"),
                   compile_s=r.get("compile_s"), us_per_step_median=r.get("us_per_step_median"),
                   us_per_step_p90=r.get("us_per_step_p90"), n_steps_in_contact=r.get("n_steps_in_contact"),
                   niter_median_all=r.get("niter_median_all"), frac_steps_at_iter_cap=r.get("frac_steps_at_iter_cap"),
                   probe_step_s=r.get("probe_step_s"), narena_bytes=r.get("narena_bytes"),
                   maxuse_arena_bytes=r.get("maxuse_arena_bytes"), maxrss_bytes=r.get("maxrss_bytes"),
                   maxuse_con=r.get("maxuse_con"), maxuse_efc=r.get("maxuse_efc"),
                   warnings=r.get("warnings"), text_warnings=r.get("text_warnings"),
                   diverged=r.get("diverged"), dominated_by_cells=r.get("dominated_by_cells"),
                   wall_total_s=r.get("wall_total_s"), foot_nverts=r.get("foot_nverts"),
                   foot_fallback_nearest=r.get("foot_fallback_nearest"),
                   centre_vertex_offset_m=r.get("centre_vertex_offset_m"))
        rec.update(d)
        rec["bounds"] = bounds(c.get("nu", 0.45))
        key = (c["case"], round(c.get("nu", 0.45), 4))
        if key in gt and rec.get("K_app_N_per_um") is not None:
            g = gt[key]["K_N_per_um"]
            rec["K_gt_N_per_um"] = g
            rec["K_gt_src"] = gt[key]["src"]
            rec["err_K_app_rel"] = rec["K_app_N_per_um"] / g - 1
            if rec.get("K_layer_N_per_um") is not None:
                rec["err_K_layer_rel"] = rec["K_layer_N_per_um"] / g - 1
        rec["phases"] = r.get("phases")
        results.append(rec)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f)
    print(f"{len(results)} records -> results.json; GT entries: {len(gt)}")
    return results, gt


if __name__ == "__main__":
    main()
