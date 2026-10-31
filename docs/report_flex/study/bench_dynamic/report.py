"""Tables and plots from results.json (run analyze.py first).

Usage: python report.py      -> tables/all_runs.csv, tables/*.md, plots/*.png
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # validated categorical order
MK = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3df"
A, B, T, E = 0.051, 0.071, 0.0015, 1e6
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})


def sel(**kw):
    out = []
    for r in R:
        c = r["config"]
        if all((tuple(c.get(k)) if k == "cells" else c.get(k)) == v for k, v in kw.items()):
            out.append(r)
    return out


def M_of(nu):
    return E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))


BOUND = {nu: M_of(nu) * A * B / T * 1e-6 for nu in (0.45, 0.49)}
FEM_C1 = 8.59   # main-session F2 quadratic-hex FEM (8.593 / 8.578), C1 no-slip nu=0.45 -- quoted, not re-run here

# ------------------------------------------------------------------ flat CSV of every run
KEYS = ["id", "status", "group", "variant", "cells", "case", "contact", "solver", "iterations", "split", "nu",
        "driver", "mass_scale", "nv", "nflexvert", "nelem", "vertex_mass_kg", "compile_s", "us_per_step_median",
        "us_per_step_p90", "niter_median", "niter_max", "frac_steps_at_iter_cap", "ncon", "nefc", "ncon_loaded",
        "delta_m", "F_main_N", "K_app_N_per_um", "u_layer_foot_mean_m", "K_layer_N_per_um", "penetration_frac_foot",
        "pen_max_m", "pinned_weight_share", "chatter_rel", "chatter_std_N", "settle_1pct_s", "settle_5pct_s",
        "drift_rel", "maxuse_arena_bytes", "maxrss_bytes", "probe_step_s", "K_gt_N_per_um", "err_K_app_rel",
        "err_K_layer_rel"]
with open(os.path.join(HERE, "tables", "all_runs.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(KEYS + ["Fx_N", "Fy_N", "Fz_N", "warnings"])
    for r in R:
        c = r["config"]
        row = []
        for k in KEYS:
            v = r.get(k, c.get(k))
            if k == "cells":
                v = "x".join(map(str, c["cells"]))
            row.append(v)
        Fv = r.get("F_vector_N") or [None] * 3
        row += list(Fv) + [";".join(f"{k}:{v}" for k, v in (r.get("warnings") or {}).items())]
        w.writerow(row)


def fmt(v, f="%.3g"):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    try:
        return f % v
    except TypeError:
        return str(v)


def md_table(name, rows, cols, title):
    with open(os.path.join(HERE, "tables", name), "w") as fh:
        fh.write(f"# {title}\n\n| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(fmt(c[1](r), c[2] if len(c) > 2 else "%.3g") for c in cols) + " |\n")


def ph(r, i=-1):
    p = r.get("phases") or []
    return p[i] if p else {}


def F_free_K(r):
    p = ph(r)
    if not p or not r.get("u_layer_foot_mean_m"):
        return None
    return p["F_to_free_dofs_N"][2] / r["u_layer_foot_mean_m"] * 1e-6


COLS = [("id", lambda r: r["id"][:8], "%s"), ("variant", lambda r: r["config"]["variant"], "%s"),
        ("cells", lambda r: "x".join(map(str, r["config"]["cells"])), "%s"), ("case", lambda r: r["config"]["case"], "%s"),
        ("contact", lambda r: r["config"]["contact"], "%s"), ("it", lambda r: r["config"]["iterations"], "%d"),
        ("status", lambda r: r["status"], "%s"),
        ("Fz N", lambda r: (r.get("F_vector_N") or [None] * 3)[2], "%.4g"),
        ("Fx N", lambda r: (r.get("F_vector_N") or [None] * 3)[0], "%.2g"),
        ("Fy N", lambda r: (r.get("F_vector_N") or [None] * 3)[1], "%.3g"),
        ("K_app N/um", lambda r: r.get("K_app_N_per_um"), "%.4g"),
        ("K_layer N/um", lambda r: r.get("K_layer_N_per_um"), "%.4g"),
        ("K_layer(free dofs) N/um", F_free_K, "%.4g"),
        ("pen. frac", lambda r: r.get("penetration_frac_foot"), "%.3f"),
        ("pen max um", lambda r: r.get("pen_max_m") and r["pen_max_m"] * 1e6, "%.3f"),
        ("ncon", lambda r: r.get("ncon"), "%d"), ("nefc", lambda r: r.get("nefc"), "%d"),
        ("CG it med", lambda r: r.get("niter_median"), "%.0f"),
        ("at cap", lambda r: r.get("frac_steps_at_iter_cap"), "%.2f"),
        ("chatter rel", lambda r: r.get("chatter_rel"), "%.1e"),
        ("settle1% s", lambda r: r.get("settle_1pct_s"), "%.3f"),
        ("us/step", lambda r: r.get("us_per_step_median"), "%.0f"),
        ("compile s", lambda r: r.get("compile_s"), "%.2f"),
        ("arena MB", lambda r: r.get("maxuse_arena_bytes") and r["maxuse_arena_bytes"] / 1e6, "%.1f"),
        ("err vs GT", lambda r: r.get("err_K_layer_rel"), "%+.3f")]


def key(r):
    c = r["config"]
    return (c["case"], c["variant"], c["contact"], c["cells"][0], c["cells"][2], c["iterations"], c["split"])


groups = {"main": "full flex F1-corrected, robot options (CG 100 it)", "conv": "same, CG 1000 it",
          "raw": "full flex UNcorrected", "tri": "trilinear", "quad": "quadratic",
          "split": "interpolated + split indenter (50-contact cap workaround)", "contact": "contact settings",
          "margin2": "corrected margin (flex only)", "newton": "Newton solver", "driver": "mocap indenter",
          "iter": "CG iteration budget", "nu49": "nu = 0.49", "timing": "timing re-measurement at low load"}
for g, t in groups.items():
    rows = sorted([r for r in R if r["group"] == g], key=key)
    extra = [("split", lambda r: r["config"]["split"], "%d"), ("m_scale", lambda r: r["config"]["mass_scale"], "%g"),
             ("solver", lambda r: r["config"]["solver"], "%s"), ("driver", lambda r: r["config"]["driver"], "%s"),
             ("nu", lambda r: r["config"]["nu"], "%g")]
    md_table(f"group_{g}.md", rows, COLS[:6] + extra + COLS[6:], f"group {g}: {t}")


def series(rows, xf, yf):
    pts = [(xf(r), yf(r)) for r in rows if r["status"] == "ok" and yf(r) is not None]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


DX = lambda r: 51.0 / r["config"]["cells"][0]  # noqa: E731


# ------------------------------------------------------------------ plot 1: C1 / C1f stiffness vs cell size
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
for col, (case, tag) in enumerate([("C1", "conv"), ("C1f", "main")]):
    for nz in range(1, 7):
        rows = [r for r in sel(case=case, variant="full_corr", contact="stiff", nu=0.45, iterations=1000 if tag == "conv" else 100)
                if r["group"] == tag and r["config"]["cells"][2] == nz]
        for rowi, (yf, lab) in enumerate([(lambda r: r.get("K_app_N_per_um"), "K_app = F / indenter disp."),
                                          (F_free_K, "K_layer = F(free dofs) / mean layer disp.")]):
            x, y = series(rows, DX, yf)
            ax[rowi, col].plot(x, y, color=COL[nz - 1], marker=MK[nz - 1], ms=6, label=f"{nz} el. thick")
            ax[rowi, col].set_title(f"{case} ({'no-slip, CG 1000 it' if case == 'C1' else 'frictionless, CG 100 it'}): {lab}",
                                    fontsize=9, color=INK)
    for rowi in range(2):
        a = ax[rowi, col]
        a.axhline(BOUND[0.45], color=INK2, ls="--", lw=1)
        a.text(10.1, BOUND[0.45] + 0.05, "upper bound M A/t = 9.157", ha="right", va="bottom", fontsize=8, color=INK2)
        if case == "C1":
            a.axhline(FEM_C1, color=INK, ls=":", lw=1)
            a.text(10.1, FEM_C1 - 0.05, "FEM ref 8.59 (F2)", ha="right", va="top", fontsize=8, color=INK)
        a.set_xscale("log")
        a.set_xticks([1.275, 1.7, 2.55, 5.1, 10.2])
        a.set_xticklabels(["1.275", "1.7", "2.55", "5.1", "10.2"])
        a.xaxis.set_minor_formatter(NullFormatter())
        a.invert_xaxis()
ax[0, 0].set_ylabel("N/um")
ax[1, 0].set_ylabel("N/um")
ax[1, 0].set_xlabel("in-plane cell size (mm)")
ax[1, 1].set_xlabel("in-plane cell size (mm)")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=8)
fig.suptitle("Full flex (F1-corrected), rigid platen, 2 um: what the indenter feels vs what the layer does", color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(HERE, "plots", "C1_stiffness_vs_mesh.png"), dpi=140)
plt.close(fig)

# ------------------------------------------------------------------ plot 2: cost, iterations, chatter
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for i, (case, it, tag) in enumerate([("C1f", 100, "main"), ("C1", 100, "main"), ("C1", 1000, "conv")]):
    rows = [r for r in sel(case=case, variant="full_corr", contact="stiff", nu=0.45, iterations=it) if r["group"] == tag]
    x, y = series(rows, lambda r: r["nv"], lambda r: r.get("us_per_step_median") and r["us_per_step_median"] / 1e3)
    ax[0].loglog(x, y, color=COL[i], marker=MK[i], ls="none", ms=6, label=f"{case}, CG {it} it")
    x, y = series(rows, lambda r: r["nv"], lambda r: r.get("niter_median"))
    ax[1].semilogx(x, y, color=COL[i], marker=MK[i], ls="none", ms=6, label=f"{case}, CG {it} it")
    x, y = series(rows, lambda r: r["nv"], lambda r: r.get("chatter_rel"))
    ax[2].loglog(x, y, color=COL[i], marker=MK[i], ls="none", ms=6, label=f"{case}, CG {it} it")
ax[0].axhline(1000, color=INK2, ls="--", lw=1)
ax[0].text(ax[0].get_xlim()[0] * 1.1, 1100, "1 s/step (stop rule)", fontsize=8, color=INK2)
for a, yl in zip(ax, ["ms per step (median, in contact)", "solver iterations (median, last 200 steps)",
                      "force chatter: std / mean, last 200 steps"]):
    a.set_xlabel("degrees of freedom nv")
    a.set_ylabel(yl)
ax[0].legend(fontsize=8)
fig.suptitle("Cost and convergence vs mesh size (full flex, F1-corrected, all 30 meshes)", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", "cost_iter_chatter.png"), dpi=140)
plt.close(fig)

# ------------------------------------------------------------------ plot 3: penetration fraction vs vertex mass
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, case in enumerate(["C1f", "C1", "C4", "C6f"]):
    rows = [r for r in sel(case=case, variant="full_corr", contact="stiff", nu=0.45, mass_scale=1.0)
            if r["group"] in ("main", "conv") and r["config"]["iterations"] == (1000 if case == "C1" else 100)]
    x, y = series(rows, lambda r: r.get("vertex_mass_kg"), lambda r: r.get("penetration_frac_foot"))
    ax[0].semilogx(np.array(x) * 1e6, y, color=COL[i], marker=MK[i], ls="none", ms=6, label=case)
ax[0].set_xlabel("free-vertex mass (mg)")
ax[0].set_ylabel("contact penetration / indenter displacement")
ax[0].legend(fontsize=8)
ax[0].set_title("stiff contact (solref 1e-4 1): the share lost in the contact", fontsize=9)
labels, vals = [], []
for cname, ms in [("stiff", 1.0), ("stiff", 10.0), ("stiff", 100.0), ("stiffest", 1.0), ("radius", 1.0),
                  ("margin_flex", 1.0), ("default", 1.0)]:
    rr = [r for r in sel(case="C1f", variant="full_corr", cells=(20, 28, 3), contact=cname, mass_scale=ms,
                         iterations=1000, nu=0.45, driver="servo", solver="CG")]
    if rr and rr[0].get("penetration_frac_foot") is not None:
        labels.append(f"{cname}" + (f" x{ms:g} mass" if ms != 1 else ""))
        vals.append(rr[0]["penetration_frac_foot"])
ax[1].barh(range(len(vals)), vals, color=COL[0], height=0.6)
for k, v in enumerate(vals):
    ax[1].text(v + 0.01, k, f"{v:.3f}", va="center", fontsize=8, color=INK)
ax[1].set_yticks(range(len(vals)))
ax[1].set_yticklabels(labels)
ax[1].set_xlim(0, 1.15)
ax[1].set_xlabel("penetration fraction (C1f, 20x28x3, 2 um)")
ax[1].set_title("contact setting", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", "penetration.png"), dpi=140)
plt.close(fig)

# ------------------------------------------------------------------ plot 4: interpolated flexes, cap and split
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, (var, s) in enumerate([("full_corr", 1), ("trilinear", 1), ("trilinear", 5), ("quadratic", 1), ("quadratic", 5)]):
    rows = [r for r in sel(case="C1f", variant=var, split=s, contact="stiff", nu=0.45, iterations=100)
            if r["group"] in ("main", "tri", "quad", "split") and r["config"]["cells"][2] in (1, 2, 3)
            and r["config"]["cells"][2] == {5: 1, 10: 2, 20: 3, 30: 3, 40: 3}[r["config"]["cells"][0]]]
    x, y = series(rows, DX, lambda r: r.get("K_app_N_per_um"))
    ax[0].plot(x, y, color=COL[i], marker=MK[i], ms=6, label=f"{var}, indenter split {s}x{s}")
    x, y = series(rows, DX, lambda r: r.get("ncon"))
    ax[1].plot(x, y, color=COL[i], marker=MK[i], ms=6, label=f"{var}, split {s}x{s}")
for a in ax:
    a.set_xscale("log")
    a.set_xticks([1.275, 1.7, 2.55, 5.1, 10.2])
    a.set_xticklabels(["1.275", "1.7", "2.55", "5.1", "10.2"])
    a.xaxis.set_minor_formatter(NullFormatter())
    a.invert_xaxis()
    a.set_xlabel("in-plane cell size (mm); meshes 5x7x1, 10x14x2, 20x28x3, 30x42x3, 40x56x3")
ax[0].set_ylabel("K_app (N/um), C1f")
ax[1].set_ylabel("contacts")
ax[1].set_yscale("log")
ax[0].legend(fontsize=7)
fig.suptitle("Interpolated flexes: 50 contacts per geom pair; splitting the platen multiplies the cap", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", "interp_cap_split.png"), dpi=140)
plt.close(fig)

# ------------------------------------------------------------------ plot 5: C6 taxel maps
rr = [r for r in sel(case="C6", variant="full_corr", cells=(20, 28, 3), contact="stiff", iterations=1000, nu=0.45)
      if r["group"] == "conv"]
if rr:
    p = ph(rr[0])
    tf = np.array(p["taxel_force_N"])[:, :, 2]
    tu = np.array(p["taxel_disp_m"])[:, :, 2] * 1e6
    nc = np.array(p["taxel_ncon"])
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.5))
    for a, data, lab in [(ax[0], tf, "taxel normal force (N)"), (ax[1], tu, "mean free-face uz per taxel (um)")]:
        im = a.imshow(data, origin="lower", cmap="Blues", extent=[-25.5, 25.5, -35.5, 35.5])
        for j in range(7):
            for i in range(5):
                txt = f"{data[j, i]:.3f}" + (f"\n{nc[j, i]} con" if a is ax[0] else "")
                a.text(-25.5 + 10.2 * (i + 0.5), -35.5 + 71 / 7 * (j + 0.5), txt, ha="center", va="center",
                       fontsize=7, color=INK if data[j, i] < 0.6 * data.max() else "white")
        a.set_title(lab, fontsize=9)
        a.set_xlabel("x (mm)")
        a.grid(False)
        fig.colorbar(im, ax=a, shrink=0.8)
    ax[0].set_ylabel("y (mm)")
    fig.suptitle("C6 no-slip square punch on the centre taxel, 5 um, full flex 20x28x3 (CG 1000 it)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plots", "C6_taxel_maps.png"), dpi=140)
    plt.close(fig)

# ------------------------------------------------------------------ plot 6: corrected vs uncorrected
fig, ax = plt.subplots(figsize=(6, 4))
for i, (var, lab) in enumerate([("full_corr", "F1-corrected inputs"), ("full_raw", "uncorrected (lambda doubled)")]):
    rows = [r for r in sel(case="C1f", variant=var, contact="stiff", nu=0.45, iterations=100)
            if r["group"] in ("main", "raw") and tuple(r["config"]["cells"]) in [(5, 7, 1), (10, 14, 2), (20, 28, 3), (40, 56, 4)]]
    x, y = series(rows, DX, F_free_K)
    ax.plot(x, y, color=COL[i], marker=MK[i], ms=6, label=lab)
ax.axhline(BOUND[0.45], color=INK2, ls="--", lw=1)
ax.text(10, BOUND[0.45] + 0.2, "M A/t = 9.157 (true upper bound)", fontsize=8, color=INK2, ha="right")
ax.set_xscale("log")
ax.set_xticks([1.275, 2.55, 5.1, 10.2])
ax.set_xticklabels(["1.275", "2.55", "5.1", "10.2"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.invert_xaxis()
ax.set_xlabel("in-plane cell size (mm); meshes 5x7x1, 10x14x2, 20x28x3, 40x56x4")
ax.set_ylabel("K_layer (N/um), C1f")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", "F1_bias.png"), dpi=140)
plt.close(fig)
print("tables/ and plots/ written")
