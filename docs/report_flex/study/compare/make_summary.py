"""Collect the headline numbers into summary.json and print markdown tables (pasted into README.md).
Inputs: out/check_gt.json, out/static_errors.csv, out/static_orders.json, out/dynamic_errors.csv,
out/config_errors.csv, out/robot_grasp.jsonl, out/chosen.json."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
O = lambda f: os.path.join(HERE, "out", f)  # noqa: E731
CASES = ["C1", "C1f", "C2", "C3", "C4", "C5", "C6", "C6f"]


def pct(x, ci=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    s = f"{100 * x:+.1f}"
    if ci is not None and not (isinstance(ci, float) and np.isnan(ci)):
        s += f" ± {100 * ci:.1f}"
    return s


def main():
    chk = json.load(open(O("check_gt.json")))
    st = pd.read_csv(O("static_errors.csv"))
    orders = json.load(open(O("static_orders.json")))["two_param"]
    dy = pd.read_csv(O("dynamic_errors.csv"))
    ce = pd.read_csv(O("config_errors.csv"))
    ch = json.load(open(O("chosen.json")))
    S = dict(date="2026-10-31", mujoco="3.17.0", units="relative errors (fraction); stiffness N/um; time ms/step")
    S["gt_crosscheck"] = {k: v for k, v in chk.items() if k != "rows"}
    keep = ["full_raw_nt2_nz3", "full_corr_nt2_nz3", "full_corr_nt1_nz2", "full_corr_nt8_nz3", "full_corr_nt4_nz3",
            "quadratic_nt8_cz3", "quadratic_nt4_cz1", "trilinear_nt16_cz6", "full_corr_nt16_nz7"]
    sc = ["C1", "C1f", "C2", "C3", "C6ns", "C6f", "C4n_a5mm", "C4p_a5mm"]
    stat = {}
    for nu in (0.45, 0.49):
        for k in keep:
            r = st[st.id == f"{k}_nu{nu}"]
            if len(r):
                r = r.iloc[0]
                stat[f"{k}_nu{nu}"] = {c: float(r[f"err_{c}"]) for c in sc if pd.notna(r.get(f"err_{c}"))}
                stat[f"{k}_nu{nu}"]["neigh_uz_C6ns"] = float(r["neigh_uz_C6ns"]) if pd.notna(r.get("neigh_uz_C6ns")) else None
                stat[f"{k}_nu{nu}"]["step_ms"] = float(r["step_ms"]) if pd.notna(r.get("step_ms")) else None
    S["static_discretisation_error"] = stat
    S["static_convergence_orders_nu0.45"] = [o for o in orders if o["nu"] == 0.45]
    sd = dy[(dy.group == "sd") & (dy.status == "ok")]
    S["static_vs_dynamic_C1f"] = [dict(config=r.config, K_app=r.K_app_N_per_um, K_static=r.K_static,
                                       err_contact=r.err_contact) for r in sd.itertuples()]
    conf = {}
    for key, v in ch["figures"].items():
        for cfg, lab in v["configs"]:
            out = {}
            for c in CASES:
                r = ce[(ce.config == cfg) & (ce.case == c)]
                if not len(r):
                    continue
                r = r.iloc[0]
                d = {m: (float(r[m]) if m in r and pd.notna(r[m]) else None) for m in
                     ("err_total", "err_total_ci95", "err_disc", "err_contact", "chatter_rel", "us_per_step_median",
                      "niter_median", "frac_steps_at_iter_cap", "settle_1pct_s", "penetration_frac_foot", "Fmap_rms_centred",
                      "neigh_uz_centred", "neigh_uz_gt_centred", "err_a", "n_runs", "n_unphysical", "n_snap_excluded")}
                out[c] = d
            conf[cfg] = dict(label=lab, cases=out)
    S["configurations"] = conf
    rb = []
    if os.path.exists(O("robot_grasp.jsonl")):
        for ln in open(O("robot_grasp.jsonl")):
            r = json.loads(ln)
            hw = r.get("held_window", {})
            rb.append(dict(tag=r["o"]["tag"], badqacc=r.get("bad"), fatal=r.get("fatal"),
                           ms_per_step_all=r.get("ms_per_step_all"), ms_per_step_contact=r.get("ms_per_step_contact"),
                           held_N_left=hw.get("nL"), held_N_right=hw.get("nR"), rigid_blade_N=hw.get("rigid_n"),
                           compression_left_um=[1e6 * x for x in hw.get("compL", [])], penetration_mm=[1e3 * x for x in hw.get("pen", [])],
                           inverted_tets_end=r.get("inverted_tets_end"), box_lift_mm=r.get("box_lift_mm"),
                           max_normal_left_N=(r.get("max_over_run") or {}).get("nL")))
    S["robot_grasp_runs"] = rb
    S["choice"] = ch["choice"]
    json.dump(S, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    # markdown tables
    for key, v in ch["figures"].items():
        print(f"\n### {v['title']}\n")
        print("| configuration | " + " | ".join(CASES) + " | ms/step (range over cases) |")
        print("|---|" + "---|" * (len(CASES) + 1))
        for cfg, lab in v["configs"]:
            cs = conf[cfg]["cases"]
            cells = [pct(cs.get(c, {}).get("err_total"), cs.get(c, {}).get("err_total_ci95")) for c in CASES]
            ms = [cs.get(c, {}).get("us_per_step_median") for c in CASES]
            ms = [x for x in ms if x]
            mstr = f"{min(ms) / 1e3:.1f}-{max(ms) / 1e3:.1f}" if ms else "-"
            print(f"| {lab} | " + " | ".join(cells) + f" | {mstr} |")
        print("\n(decomposition: discretisation / contact)\n")
        print("| configuration | " + " | ".join(c for c in CASES if c != "C5") + " |")
        print("|---|" + "---|" * (len(CASES) - 1))
        for cfg, lab in v["configs"]:
            cs = conf[cfg]["cases"]
            print(f"| {lab} | " + " | ".join(f"{pct(cs.get(c, {}).get('err_disc'))} / {pct(cs.get(c, {}).get('err_contact'))}"
                                            for c in CASES if c != "C5") + " |")


if __name__ == "__main__":
    main()
