"""Check 1: do the two ground truths agree where they overlap (C4: 12 values, C6/C6f: K and taxel u_z maps)?
Output: out/check_gt.json (+ printed table)."""
import json
import os

import numpy as np

import gt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
best = gt.fem()["C4_axisymmetric"]["best"]
for a in ("5.0", "2.5"):
    for fr in ("frictionless", "noslip"):
        for nu in (0.45, 0.49, 0.499):
            f = best[f"a{a}/{fr}/nu{nu:.3f}"]
            case = "C4" + ("ns" if fr == "noslip" else "") + ("_a2.5" if a == "2.5" else "")
            t = gt.get(case, nu)
            d = f["K_N_per_um"] - t["K"]
            bar = np.hypot(f["K_err_N_per_um"], t["K_err"])
            rows.append(dict(case=case, nu=nu, K_fem=f["K_N_per_um"], K_fem_err=f["K_err_N_per_um"], K_tr=t["K"],
                             K_tr_err=t["K_err"], rel_diff=d / t["K"], diff_over_bar=abs(d) / bar))
for case in ("C6", "C6f"):
    for nu in (0.45, 0.49, 0.499):
        t = gt.get(case, nu)
        d = t["K_fem"] - t["K"]
        bar = np.hypot(t["K_fem_err"], t["K_err"])
        du = np.abs(t["uz_map_fem"] - t["uz_map"]).max()
        rows.append(dict(case=case, nu=nu, K_fem=t["K_fem"], K_fem_err=t["K_fem_err"], K_tr=t["K"], K_tr_err=t["K_err"],
                         rel_diff=d / t["K"], diff_over_bar=abs(d) / bar, uz_map_maxabs_diff_per_delta=float(du),
                         neigh_edge_uz_tr=t["neigh_edge_uz"], neigh_edge_uz_fem=gt.neigh_edge(t["uz_map_fem"])))
ok = all(r["diff_over_bar"] <= 1.0 for r in rows)
res = dict(rows=rows, all_within_combined_bar=ok,
           max_rel_diff_C4=max(abs(r["rel_diff"]) for r in rows if r["case"].startswith("C4")),
           max_rel_diff_C6=max(abs(r["rel_diff"]) for r in rows if r["case"].startswith("C6")),
           max_uz_map_diff=max(r.get("uz_map_maxabs_diff_per_delta", 0) for r in rows))
json.dump(res, open(os.path.join(HERE, "out", "check_gt.json"), "w"), indent=1)
for r in rows:
    print(f"{r['case']:9s} {r['nu']:.3f} fem {r['K_fem']:.6f}+-{r['K_fem_err']:.1e} tr {r['K_tr']:.6f}+-{r['K_tr_err']:.1e}"
          f" rel {r['rel_diff']:+.2e} |d|/bar {r['diff_over_bar']:.2f} {r.get('uz_map_maxabs_diff_per_delta', '')}")
print("all within bar:", ok, {k: v for k, v in res.items() if k != "rows"})
