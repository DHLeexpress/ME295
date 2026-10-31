"""Total error of MuJoCo flex with real contacts vs ground truth, split into discretisation (static operator, same mesh)
and contact parts.

Inputs : ../bench_dynamic/results.json (1666 runs, rig.py), runs/*.json (this task, rig2.py), out/static_errors.csv
Outputs: out/dynamic_errors.csv (one row per run), out/config_errors.csv / .json (per configuration: placement mean,
         95 % CI over placements for local cases)
Definitions (per run, nu-matched GT from gt.py):
  K_app = F / indenter displacement (C2: dFy / dy)          err_total   = K_app / K_gt - 1
  K_static = MuJoCo's own operator, same mesh, exact BCs     err_disc    = K_static / K_gt - 1   (bench_static)
                                                             err_contact = K_app / K_static - 1
  C5: F at the final level (30 um) vs exact F(30 um); contact radius = max radius of loaded contacts vs exact a.
  C6: centre-taxel force fraction (exact 1), mean u_z/delta of the 4 edge neighbours (exact -0.035 / -0.047 at 0.45).
  Taxel maps: RMS over 35 taxels of the force fractions; u_z map RMS over the unloaded taxels (per delta).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import gt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "bench_dynamic"))
import analyze as bd_an  # noqa: E402  (bench_dynamic/analyze.py: derive())

GTCASE = {"C1": "C1", "C1f": "C1f", "C2": "C2", "C3": "C3", "C4": "C4", "C6": "C6", "C6f": "C6f"}
STATCASE = {"C1": "C1", "C1f": "C1f", "C2": "C2", "C3": "C3", "C4": "C4n_a5mm", "C6": "C6ns", "C6f": "C6f"}
LOCAL = ("C4", "C5", "C6", "C6f")


def static_lookup():
    s = pd.read_csv(os.path.join(HERE, "out", "static_errors.csv"))
    d = {}
    for _, r in s.iterrows():
        d[(r.variant, round(r.nu, 3), int(r.nt), int(r.nz_el))] = r
    return d


def load_records():
    out = []
    for r in json.load(open(os.path.join(ROOT, "bench_dynamic", "results.json"))):
        r["_src"] = "bench_dynamic"
        r.setdefault("cfg", r.get("config"))
        out.append(r)
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "*.json"))):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("cfg", {}).get("tag") == "test":
            continue
        r["_src"] = "compare"
        if r["cfg"].get("graded"):          # graded direct mesh: label by its finest size, no static counterpart
            gm = r["cfg"]["graded"]
            r["cfg"]["variant"] = f"graded_h{1e3 * gm['h0']:g}mm_nz{len(gm['z']) - 1}"
            r["cfg"]["cells"] = [1, 1, 1]
        if r.get("status") == "ok" and r.get("phases"):
            r.update(bd_an.derive(r))
            if r["cfg"]["case"] == "C3":
                p = r["phases"][-1]
                dlt = p["ind_disp_m"][2]
                F = p["F_mean_N"][2]
                u = p["uz_foot_mean_m"]
                r.update(K_app_N_per_um=F / dlt * 1e-6, K_layer_N_per_um=F / u * 1e-6 if abs(u) > 1e-12 else None,
                         penetration_frac_foot=1 - u / dlt, chatter_rel=p["chatter_rel"],
                         settle_1pct_s=p["settle_1pct_s"], ncon=p["ncon"], niter_median=p["niter_median"],
                         pinned_weight_share=p.get("pinned_weight_share"), F_vector_N=p["F_mean_N"])
        out.append(r)
    return out


def row_of(r, SL):
    c = r["cfg"]
    case = c["case"]
    nu = round(c["nu"], 3)
    cells = c["cells"]
    off = c.get("offset_m") or [0.0, 0.0]
    row = dict(src=r["_src"], id=r["id"], group=r.get("group") or c.get("tag"), status=r["status"],
               variant=c["variant"], cx=cells[0], cy=cells[1], cz=cells[2], case=case, contact=c["contact"],
               iterations=c["iterations"], tolerance=c.get("tolerance", 1e-8), solver=c.get("solver", "CG"),
               mass_scale=c.get("mass_scale", 1.0), split=c.get("split", 1), nu=nu, driver=c.get("driver", "servo"),
               indenter=c.get("indenter_type", "auto"), offx_mm=1e3 * off[0], offy_mm=1e3 * off[1],
               verts=str(c.get("verts")))
    row["config"] = (f"{row['variant']} {cells[0]}x{cells[1]}x{cells[2]} {row['contact']} it{row['iterations']}"
                     f" m{row['mass_scale']:g}" + (f" split{row['split']}" if row["split"] != 1 else "")
                     + (" mocap" if row["driver"] == "mocap" else "") + (" plane" if row["indenter"] == "plane" else "")
                     + (f" {row['solver']}" if row["solver"] != "CG" else "") + f" nu{nu:g}")
    if r["status"] != "ok" or not r.get("phases"):
        return row
    for k in ("us_per_step_median", "niter_median", "niter_median_all", "frac_steps_at_iter_cap", "settle_1pct_s",
              "settle_5pct_s", "chatter_rel", "ncon", "penetration_frac_foot", "pinned_weight_share", "K_app_N_per_um",
              "K_layer_N_per_um", "vertex_mass_kg", "compile_s", "slip_end_m"):
        v = r.get(k)
        row[k] = v if (v is None or np.isscalar(v)) else None
    row["n_mjWARN_INERTIA"] = (r.get("warnings") or {}).get("mjWARN_INERTIA", 0)
    # unphysical end state: mean free-face displacement under the indenter beyond the indenter displacement
    # (penetration share < -5 %) or > 1.5 x the indenter displacement = inverted / snapped-through layer
    pf = r.get("penetration_frac_foot")
    row["unphysical"] = bool(pf is not None and np.isscalar(pf) and (pf < -0.05))
    # snapped through / inverted: mean free-face displacement under the indenter > 3 x the indenter displacement
    row["snap"] = bool(pf is not None and np.isscalar(pf) and (pf < -2.0))
    # static counterpart
    nt = cells[0] // 5 if cells[0] % 5 == 0 else None
    st = SL.get((c["variant"], nu, nt, cells[2])) if nt else None
    if case in GTCASE:
        g = gt.get(GTCASE[case], nu)
        row["K_gt"] = g["K"]
        if row.get("K_app_N_per_um") is not None:
            row["err_total"] = row["K_app_N_per_um"] / g["K"] - 1
        if st is not None and f"K_{STATCASE[case]}" in st and pd.notna(st[f"K_{STATCASE[case]}"]):
            ks = st[f"K_{STATCASE[case]}"]
            row["K_static"] = ks
            row["err_disc"] = ks / g["K"] - 1
            if row.get("K_app_N_per_um") is not None:
                row["err_contact"] = row["K_app_N_per_um"] / ks - 1
    p = r["phases"][-1]
    dlt = p["ind_disp_m"][2]
    if case in ("C1", "C3", "C6", "C6f", "C1f") and "taxel_force_N" in p and abs(dlt) > 0:
        g = gt.get(GTCASE[case], nu)
        F = np.array(p["taxel_force_N"])[:, :, 2]
        U = np.array(p["taxel_disp_m"])[:, :, 2] / dlt
        if F.sum() > 0:
            row["Fmap_rms"] = float(np.sqrt(np.mean((F / F.sum() - g["F_map"]) ** 2)))
            if case in ("C6", "C6f"):
                row["centreF_frac"] = float(F[3, 2] / F.sum())
        if case == "C3":
            out = np.zeros((7, 5), bool)
            out[:4] = True
            row["uzmap_rms"] = float(np.sqrt(np.mean((U[out] - g["uz_map"][out]) ** 2)))
            row["C3_row3_uz"] = float(U[3].mean())
            row["C3_row3_uz_gt"] = float(g["uz_map"][3].mean())
        if case in ("C6", "C6f"):
            out = np.ones((7, 5), bool)
            out[3, 2] = False
            row["uzmap_rms"] = float(np.sqrt(np.mean((U[out] - g["uz_map"][out]) ** 2)))
            row["neigh_uz"] = gt.neigh_edge(U)
            row["neigh_uz_gt"] = g["neigh_edge_uz"]
            row["centre_uz"] = float(U[3, 2])
    if case == "C2":
        g = gt.get("C2", nu)
        row["K_gt"] = g["K"]
    if case == "C5" and r.get("C5_F_N"):
        R = "R25" if (c.get("sphere_R_m") or 0.01) > 0.02 else "R10"
        Fs, ds = np.array(r["C5_F_N"]), 1e6 * np.array(r["C5_delta_m"])
        Fg = np.array([gt.c5_F(d, nu, R)[0] for d in ds])
        row["C5_F30_N"] = float(Fs[-1] * 30 / ds[-1])
        row["C5_F30_gt_N"] = gt.c5_F(30, nu, R)[0]
        row["err_total"] = row["C5_F30_N"] / row["C5_F30_gt_N"] - 1
        row["C5_F_rms_rel"] = float(np.sqrt(np.mean((Fs / Fg - 1) ** 2)))
        row["C5_a_mm"] = 1e3 * r["C5_contact_radius_max_m"][-1]
        row["C5_a_gt_mm"] = gt.c5_F(ds[-1], nu, R)[1]
        row["err_a"] = row["C5_a_mm"] / row["C5_a_gt_mm"] - 1
        row["C5_ncon"] = r["C5_ncon"][-1]
    return row


def tci(x):
    """mean, half-width of the 95 % t-interval, n"""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    n = len(x)
    if n == 0:
        return None, None, 0
    if n == 1:
        return float(x[0]), None, 1
    from scipy.stats import t
    return float(x.mean()), float(t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)), n


def main():
    SL = static_lookup()
    rows = [row_of(r, SL) for r in load_records()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "out", "dynamic_errors.csv"), index=False)
    # per configuration (compare runs): aggregate placements
    cm = df[(df.src == "compare") & (df.status == "ok") & (~df.snap.fillna(False).astype(bool))]
    _ok = df[(df.src == "compare") & (df.status == "ok")]
    nunph = _ok.groupby(["config", "case"]).unphysical.sum()
    nsnap = _ok.groupby(["config", "case"]).snap.sum()
    agg = []
    metrics = ["err_total", "err_disc", "err_contact", "Fmap_rms", "uzmap_rms", "neigh_uz", "centreF_frac", "err_a",
               "C5_F_rms_rel", "chatter_rel", "us_per_step_median", "niter_median", "frac_steps_at_iter_cap",
               "settle_1pct_s", "penetration_frac_foot", "ncon"]
    for (cfg, case), g in cm.groupby(["config", "case"]):
        a = dict(config=cfg, case=case, n_runs=len(g), n_unphysical=int(nunph.get((cfg, case), 0)),
                 n_snap_excluded=int(nsnap.get((cfg, case), 0)))
        for m in metrics:
            if m in g:
                mu, hw, n = tci(g[m].values.astype(float))
                a[m] = mu
                if hw is not None:
                    a[m + "_ci95"] = hw
        g0 = g[(g.offx_mm == 0) & (g.offy_mm == 0)]
        if len(g0):
            for m in ("err_total", "neigh_uz", "neigh_uz_gt", "Fmap_rms", "uzmap_rms", "C3_row3_uz", "C3_row3_uz_gt"):
                if m in g0 and pd.notna(g0[m].iloc[0]):
                    a[m + "_centred"] = float(g0[m].iloc[0])
        agg.append(a)
    ag = pd.DataFrame(agg)
    ag.to_csv(os.path.join(HERE, "out", "config_errors.csv"), index=False)
    ag.to_json(os.path.join(HERE, "out", "config_errors.json"), orient="records", indent=1)
    n_bad = df[df.status != "ok"].groupby(["src", "status"]).size()
    print(len(df), "dynamic records;", "non-ok:", n_bad.to_dict())


if __name__ == "__main__":
    main()
