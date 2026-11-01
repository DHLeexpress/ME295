"""Figures for Lab Report 2 (week 2, v0): skin-model verification and the grip ablation.

Data come from files in the repository:
  docs/report_flex/study/compare/gt.py            exact / FEM ground truth (Winkler-gap panel)
  docs/report_flex/study/torsion/out_*.json       torsion test (grip panel)
  results/ablation_rev/results.json               40-run ablation (outcome grid)
The patch-test ratios and the through-thickness error budget are the published numbers of docs/report_flex
(flex_study.pdf Table 2; study/compare/README.md section 1, C1 at 20 x 28 cells).

    python reports/LR2_week2_v0/make_figures.py      -> reports/LR2_week2_v0/fig/*.pdf
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)

# reference palette (dataviz skill, light mode): categorical slots in fixed order, status colours, text inks
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontweight="bold", pad=6)


# ----------------------------------------------------------------------------------------------- figure: verify
def fig_verify():
    sys.path.insert(0, os.path.join(ROOT, "docs/report_flex/study/compare"))
    import gt  # noqa: E402  (exact transform/BEM and Q2-FEM ground truth, read-only)

    fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.35))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.93, bottom=0.10, wspace=0.34, hspace=0.62)

    # (a) patch test: stress ratio MuJoCo / exact for exact homogeneous strains (nu = 0.45)
    ax = axs[0, 0]
    states = ["uniaxial\nstrain", "simple\nshear", "hydrostatic"]
    raw, fixed = [1.818182, 1.000000, 1.931034], [1.0, 1.0, 1.0]
    x = np.arange(3)
    w = 0.32
    ax.bar(x - w / 2 - 0.01, raw, w, color=BLUE, label="young/poisson as given")
    ax.bar(x + w / 2 + 0.01, fixed, w, color=ORANGE, label="remapped inputs")
    ax.axhline(1.0, color=INK2, lw=0.8)
    for xi, v in zip(x, raw):
        if v > 1.01:
            ax.text(xi - w / 2, v + 0.04, f"{v:.2f}$\\times$", ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_xticks(x, states)
    ax.set_ylim(0, 2.75)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylabel("MuJoCo / exact stress")
    ax.legend(loc="upper center", frameon=False, ncol=2, handlelength=1.2, columnspacing=1.0)
    ax.grid(axis="x", visible=False)
    title(ax, "(a) Patch test: the full flex doubles Lamé λ")

    # (b) whole-face stiffness error vs elements through the 1.5 mm layer (C1, 20 x 28 cells)
    ax = axs[0, 1]
    n = np.arange(1, 7)
    disc = np.array([6.9, 3.2, 2.5, 2.3, 2.1, 2.1])
    cont = np.array([0.0, -15.1, -22.8, -28.5, -33.2, -37.1])
    tot = np.array([6.9, -12.4, -20.9, -27.0, -31.8, -35.8])
    ax.axhline(0, color=INK2, lw=0.8)
    for y, c, lab, mk in [(disc, BLUE, "elastic operator", "o"), (cont, ORANGE, "contact spring", "s"),
                          (tot, AQUA, "total (rig reading)", "D")]:
        ax.plot(n, y, color=c, marker=mk, ms=4.2, mec="white", mew=0.8, label=lab)
    ax.text(6.15, disc[-1], "elastic", color=INK, va="center", fontsize=7.5)
    ax.text(6.15, cont[-1] - 3.0, "contact", color=INK, va="center", fontsize=7.5)
    ax.text(6.15, tot[-1] + 3.0, "total", color=INK, va="center", fontsize=7.5)
    ax.set_xlim(0.7, 7.1)
    ax.set_ylim(-42, 12)
    ax.set_xticks(n)
    ax.set_xlabel("elements through the 1.5 mm layer")
    ax.set_ylabel("whole-face stiffness error (%)")
    title(ax, "(b) Overmeshing softens the contact")

    # (c) Winkler / exact stiffness vs Poisson ratio, four loads
    ax = axs[1, 0]
    nus = [0.45, 0.49, 0.499]
    cases = [("C1", "whole face", BLUE, "o"), ("C3", "load edge", ORANGE, "s"),
             ("C6", "one taxel", AQUA, "D"), ("C4", "5 mm punch", YELLOW, "^")]
    xs = np.arange(3)
    A, B, T = 0.051, 0.071, 0.0015
    for case, lab, c, mk in cases:
        r = []
        for nu in nus:
            M = gt.closed(nu)["M"]
            area = {"C1": A * B, "C3": A * (0.0355 - 0.00507), "C6": 0.0102 * (0.071 / 7), "C4": np.pi * 0.005 ** 2}[case]
            r.append(M * area / T * 1e-6 / gt.get(case, nu)["K"])
        ax.plot(xs, r, color=c, marker=mk, ms=4.2, mec="white", mew=0.8, label=lab)
        ax.text(2.1, r[-1], lab, color=INK, va="center", fontsize=7.5)
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 30], ["1", "2", "5", "10", "30"])
    ax.minorticks_off()
    ax.set_xticks(xs, ["$\\nu$ = 0.45\n(placeholder)", "$\\nu$ = 0.49", "$\\nu$ = 0.499\n(Sylgard-like)"])
    ax.set_xlim(-0.2, 2.75)
    ax.set_ylabel("Winkler / exact stiffness")
    title(ax, "(c) Cell (Winkler) model vs exact elasticity")

    # (d) torque that starts a grasped box turning (1 deg), ~24 N per pad
    ax = axs[1, 1]
    tdir = os.path.join(ROOT, "docs/report_flex/study/torsion")
    rows = [("cells (demo model)", "cells_0.2"), ("flex 6×8×2", "flex_0.2"),
            ("flex 11×15×2", "flex11x15x2_0.2"), ("flex 11×15×3", "flex11x15x3_0.2")]
    vals = [json.load(open(os.path.join(tdir, f"out_{tag}.json")))["torque_at_1deg"] for _, tag in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, 0.5, color=BLUE)
    exact = 1.11
    ax.axvline(exact, color=INK2, lw=0.9)
    ax.text(exact - 0.03, -0.72, "exact elasticity, 1.11", ha="right", va="center", color=INK, fontsize=7.5)
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.2f}", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.32)
    ax.set_ylim(-1.05, len(rows) - 0.45)
    ax.set_xlabel("twist at which the box starts to turn (N m)")
    ax.grid(axis="y", visible=False)
    title(ax, "(d) Grasp torsion at 24 N per pad")

    fig.savefig(os.path.join(OUT, "fig_verify.pdf"))
    plt.close(fig)
    return {"torsion": dict(zip([r[1] for r in rows], vals))}


# ---------------------------------------------------------------------------------------------- figure: ablation
def fig_ablation():
    runs = json.load(open(os.path.join(ROOT, "results/ablation_rev/results.json")))
    ctrls = [("skin", "skin controller (ours)"), ("fixed-125N", "fixed 125 N"), ("fixed-70N", "fixed 70 N"),
             ("fixed-40N", "fixed 40 N"), ("fixed-10N", "fixed 10 N")]
    masses = [0.4, 0.8, 1.6, 2.4]
    mus = [1.0, 0.5]
    look = {(r["controller"], r["mu"], r["mass"]): r for r in runs}
    style = {"shelved": (GOOD, "✓"), "misplaced": (WARN, "→"), "dropped": (CRIT, "✕")}

    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    fig.subplots_adjust(left=0.2, right=0.995, top=0.855, bottom=0.0)
    ax.set_xlim(-0.5, 8.35)
    ax.set_ylim(-1.3, 4.6)
    ax.axis("off")
    cw, ch = 0.92, 0.84
    for gi, mu in enumerate(mus):
        x0 = gi * 4.4
        hdr = "true μ = 1 (as the skin assumes)" if mu == 1.0 else "true μ = 0.5 (skin overestimates μ 2×)"
        ax.text(x0 + 1.5, 4.95, hdr, ha="center", va="bottom", color=INK, fontsize=8.5, fontweight="bold")
        for mi, m in enumerate(masses):
            ax.text(x0 + mi, 4.47, f"{m:g} kg", ha="center", va="bottom", color=INK2, fontsize=7.5)
    for ri, (key, lab) in enumerate(ctrls):
        yy = 3.9 - ri
        ax.text(-0.62, yy, lab, ha="right", va="center", color=INK, fontsize=8.5,
                fontweight="bold" if key == "skin" else "normal")
        for gi, mu in enumerate(mus):
            x0 = gi * 4.4
            n_ok = 0
            for mi, m in enumerate(masses):
                r = look[(key, mu, m)]
                col, glyph = style[r["outcome"]]
                n_ok += r["outcome"] == "shelved"
                ax.add_patch(FancyBboxPatch((x0 + mi - cw / 2, yy - ch / 2), cw, ch, boxstyle="round,pad=0,rounding_size=0.08",
                                            fc=col, ec="white", lw=1.5, alpha=0.92))
                tc = "white" if r["outcome"] != "misplaced" else INK
                if r["outcome"] == "shelved":
                    ax.text(x0 + mi - 0.27, yy, glyph, ha="center", va="center", color=tc, fontsize=7.5, family="DejaVu Sans")
                    ax.text(x0 + mi + 0.1, yy, f"{r['squeeze_hold']:.0f} N", ha="center", va="center", color=tc,
                            fontsize=8, fontweight="bold")
                else:
                    ax.text(x0 + mi, yy, glyph, ha="center", va="center", color=tc, fontsize=9, family="DejaVu Sans")
            ax.text(x0 + 3.62, yy, f"{n_ok}/4", ha="left", va="center", color=INK2, fontsize=8)
    # legend below the grid (icon + label: a status colour never carries meaning alone)
    items = [("shelved (number: squeeze per blade while held)", "shelved"), ("misplaced", "misplaced"), ("dropped", "dropped")]
    for (lab, out), x in zip(items, [0.1, 4.75, 6.25]):
        col, glyph = style[out]
        ax.add_patch(FancyBboxPatch((x - 0.17, -1.13), 0.34, 0.36, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=col, ec="white", lw=1))
        ax.text(x, -0.95, glyph, ha="center", va="center", color="white" if out != "misplaced" else INK, fontsize=7.5,
                family="DejaVu Sans")
        ax.text(x + 0.28, -0.95, lab, ha="left", va="center", color=INK, fontsize=7.5)
    fig.savefig(os.path.join(OUT, "fig_ablation.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    info = fig_verify()
    fig_ablation()
    print("wrote", sorted(os.listdir(OUT)), info)
