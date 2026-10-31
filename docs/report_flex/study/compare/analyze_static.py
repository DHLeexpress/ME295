"""Discretisation error of MuJoCo's flex operator (no contact): bench_static models vs the final ground truths.

Input : ../bench_static/results.json (295 models, MuJoCo's own K by FD of qfrc_spring, exact BCs), ../bench_static/timing.jsonl
        gt.py (final gt_fem / gt_transform values)
Output: out/static_errors.csv (one row per model), out/static_orders.json (observed convergence orders),
        out/static_summary.json
Error = K_model / K_gt - 1 (signed). Taxel-map metrics: F-map RMS = RMS over the 35 taxels of (F_taxel/F_total)_model
- (F_taxel/F_total)_gt (normal component; C2: y); uz-map RMS = RMS over taxels outside the loaded region of
(mean u_z/delta)_model - gt; C6 cross-talk = mean u_z/delta of the 4 edge neighbours of the centre taxel.
"""
import json
import os

import numpy as np
import pandas as pd

import gt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CASEMAP = {"C1": "C1", "C1f": "C1f", "C2": "C2", "C3": "C3", "C6ns": "C6", "C6f": "C6f",
           "C4n_a5mm": "C4", "C4p_a5mm": "C4", "C4n_a2.5mm": "C4_a2.5", "C4p_a2.5mm": "C4_a2.5"}
LOADED = {"C1": np.ones((7, 5), bool), "C1f": np.ones((7, 5), bool)}
m3 = np.zeros((7, 5), bool)
m3[4:, :] = True
LOADED["C3"] = m3
m6 = np.zeros((7, 5), bool)
m6[3, 2] = True
LOADED["C6ns"] = LOADED["C6f"] = m6


def load():
    recs = json.load(open(os.path.join(ROOT, "bench_static", "results.json")))
    timing = {}
    p = os.path.join(ROOT, "bench_static", "timing.jsonl")
    if os.path.exists(p):
        for ln in open(p):
            try:
                t = json.loads(ln)
            except Exception:
                continue
            if "id" in t:
                timing[t["id"]] = t
    return recs, timing


def main():
    recs, timing = load()
    rows = []
    for r in recs:
        if r.get("status") != "ok":
            continue
        c = r["config"]
        nu = c["nu"]
        row = dict(id=r["id"], variant=c["variant"], nu=nu, nt=c["nt"], dx_mm=c["elem_dx_mm"], dz_mm=c["elem_dz_mm"],
                   nz_el=c["elements_through_thickness"], node_dx_mm=c["node_dx_mm"], node_dz_mm=c["node_dz_mm"],
                   aspect=c["aspect_dx_over_dz"], ndof=r["model"]["nv_unpinned"])
        cases = r["cases"]
        for sc, gc in CASEMAP.items():
            if sc not in cases or cases[sc].get("K_N_per_um") is None:
                continue
            g = gt.get(gc, nu)
            K = cases[sc]["K_N_per_um"]
            row[f"K_{sc}"] = K
            row[f"err_{sc}"] = K / g["K"] - 1
            tx = cases[sc].get("taxel")
            if tx and "F_map" in g:
                F = np.array(tx["force_N_per_um"]).T            # [i][j] -> [j][i]
                U = np.array(tx["mean_uz_over_delta"]).T
                Fm = F / F.sum()
                row[f"Fmap_rms_{sc}"] = float(np.sqrt(np.mean((Fm - g["F_map"]) ** 2)))
                out = ~LOADED[sc]
                if out.any():
                    row[f"uzmap_rms_{sc}"] = float(np.sqrt(np.mean((U[out] - g["uz_map"][out]) ** 2)))
                if sc in ("C6ns", "C6f"):
                    row[f"neigh_uz_{sc}"] = gt.neigh_edge(U)
                    row[f"neigh_uz_gt_{sc}"] = g["neigh_edge_uz"]
                    row[f"centreF_{sc}"] = float(Fm[3, 2])
                if sc == "C3":
                    row["C3_row3_uz"] = float(U[3].mean())
                    row["C3_row3_uz_gt"] = float(g["uz_map"][3].mean())
                    row["C3_row4_Ffrac"] = float(Fm[4].sum())
                    row["C3_row4_Ffrac_gt"] = float(g["F_map"][4].sum())
        cond = r.get("conditioning", {})
        for k in ("pcg_iters_blockjacobi_gravity", "cond_A", "h_omega_max"):
            if k in cond:
                row[k] = cond[k]
        t = timing.get(r["id"])
        if t:
            for k in ("step_ms", "mj_step_ms_median", "step_s_median"):
                if k in t:
                    row["step_ms"] = t[k] * (1e3 if k.endswith("_s_median") else 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    # fill conditioning / timing from summary.csv where the record layout differs
    s = pd.read_csv(os.path.join(ROOT, "bench_static", "summary.csv")).set_index("id")
    for k in ("pcg_iters_blockjacobi_gravity", "pcg_iters_massprec_gravity", "cond_A", "cond_PA", "h_omega_max",
              "step_ms", "compile_s"):
        if k in s.columns:
            df[k] = df["id"].map(s[k])
    df.to_csv(os.path.join(HERE, "out", "static_errors.csv"), index=False)

    # observed convergence orders in the in-plane size at fixed through-thickness discretisation:
    # p from a least-squares fit of log|err| vs log(dx) over the 3 finest in-plane levels with a consistent sign
    orders = []
    for (var, nu, nz), g in df.groupby(["variant", "nu", "nz_el"]):
        g = g.sort_values("dx_mm")
        for sc in ["C1", "C1f", "C3", "C6ns", "C6f", "C4n_a5mm", "C4p_a5mm"]:
            col = f"err_{sc}"
            if col not in g or g[col].isna().all():
                continue
            gg = g[["dx_mm", col]].dropna()
            if len(gg) < 3:
                continue
            gg = gg.iloc[:3]
            e = gg[col].values
            if not (np.all(e > 0) or np.all(e < 0)):
                continue
            p = np.polyfit(np.log(gg["dx_mm"].values), np.log(np.abs(e)), 1)[0]
            orders.append(dict(variant=var, nu=nu, nz_el=int(nz), case=sc, p=float(p),
                               dx_range_mm=[float(gg.dx_mm.min()), float(gg.dx_mm.max())],
                               err_finest=float(e[0])))
    # two-parameter model e = a dx^p + b dz^q (relative residuals) over all models with >= 2 elements through the
    # thickness (nz = 1 is a different, uniaxial-strain-limited regime) -> separate in-plane and through-thickness orders
    from scipy.optimize import least_squares
    fits = []
    for (var, nu), g in df.groupby(["variant", "nu"]):
        g = g[g.nz_el >= (1 if var == "quadratic" else 2)]
        for sc in ["C1", "C1f", "C3", "C6ns", "C6f", "C4p_a5mm"]:
            col = f"err_{sc}"
            gg = g[["dx_mm", "dz_mm", col]].dropna()
            if len(gg) < 6 or not (gg[col] > 0).all():
                continue
            x, z, e = gg.dx_mm.values, gg.dz_mm.values, gg[col].values

            def res(th):
                return (np.exp(th[0]) * x ** th[1] + np.exp(th[2]) * z ** th[3]) / e - 1
            best = None
            for p0 in (0.5, 1.0, 2.0):
                for q0 in (0.5, 1.0, 2.0):
                    sol = least_squares(res, [np.log(e.mean() / 2), p0, np.log(e.mean() / 2), q0],
                                        bounds=([-30, 0, -30, 0], [10, 4, 10, 4]))
                    if best is None or sol.cost < best.cost:
                        best = sol
            th = best.x
            fits.append(dict(variant=var, nu=nu, case=sc, a=float(np.exp(th[0])), p_inplane=float(th[1]),
                             b=float(np.exp(th[2])), q_thickness=float(th[3]),
                             rms_rel_resid=float(np.sqrt(np.mean(best.fun ** 2))), n=int(len(gg)),
                             note="e = a dx_mm^p + b dz_mm^q; dx = cell size (quadratic: cell = 2 node spacings)"))
    json.dump(dict(fixed_nz_3finest=orders, two_param=fits), open(os.path.join(HERE, "out", "static_orders.json"), "w"),
              indent=1)
    for f in fits:
        if f["nu"] == 0.45:
            print(f"{f['variant']:10s} {f['case']:9s} p={f['p_inplane']:.2f} q={f['q_thickness']:.2f} "
                  f"a={f['a']:.3g} b={f['b']:.3g} resid={f['rms_rel_resid']:.2f} n={f['n']}")
    od = pd.DataFrame(orders)
    summ = {}
    for (var, sc), g in od.groupby(["variant", "case"]):
        summ[f"{var}/{sc}"] = dict(p_median=float(g.p.median()), p_min=float(g.p.min()), p_max=float(g.p.max()), n=len(g))
    json.dump(dict(orders_by_variant_case=summ, n_models=len(df)), open(os.path.join(HERE, "out", "static_summary.json"), "w"),
              indent=1)
    print(len(df), "models")
    for k, v in summ.items():
        print(f"{k:22s} p median {v['p_median']:.2f} [{v['p_min']:.2f}, {v['p_max']:.2f}] n={v['n']}")


if __name__ == "__main__":
    main()
