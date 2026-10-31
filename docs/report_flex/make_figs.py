"""Figures for flex_study.tex, re-plotted at print size from the result files of the other flexstudy tasks
(read only). Output: fig/*.pdf, fig/stills.jpg. Run: /home/dohyun/miniforge3/bin/python make_figs.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FS = os.path.join(HERE, "study")   # the curated copy of the flexstudy task directories
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED = "#0b0b0b", "#8a8984"
VAR = {"full_corr": (SER[0], "o", "full, F1-corrected"), "trilinear": (SER[1], "s", "trilinear"),
       "quadratic": (SER[2], "^", "quadratic"), "full_raw": (SER[3], "D", "full, uncorrected")}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.3,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.2, "lines.markersize": 3.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 300})

st = pd.read_csv(os.path.join(FS, "compare/out/static_errors.csv"))
dy = pd.read_csv(os.path.join(FS, "compare/out/dynamic_errors.csv.gz"), low_memory=False)
s45 = st[st.nu == 0.45]


def xticks(ax):
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(["1", "2", "5", "10"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def fig_conv():
    """static discretisation error vs in-plane cell size (2 elements thick; quadratic 1 cell)."""
    cases = [("C1", "(a) C1 whole-face platen, no-slip"), ("C3", "(b) C3 half platen (pad edge)"),
             ("C6ns", "(c) C6 centre-taxel punch, no-slip")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.25))
    for ax, (c, title) in zip(axs, cases):
        for var, (col, mk, lab) in VAR.items():
            nz = 1 if var == "quadratic" else 2
            g = s45[(s45.variant == var) & (s45.nz_el == nz)].sort_values("dx_mm")
            ax.loglog(g.dx_mm, 100 * g[f"err_{c}"].abs(), marker=mk, color=col, label=lab)
        ax.set_title(title)
        ax.set_xlabel("in-plane cell size (mm)")
        xticks(ax)
    axs[0].set_ylabel(r"$|K/K_{\rm exact}-1|$ (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, "conv.pdf"))
    plt.close(fig)


def fig_cost():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1))
    ax = axs[0]
    d = dy[(dy.src == "bench_dynamic") & (dy.group == "conv") & (dy.case == "C1") & (dy.cx == 20)
           & (dy.status == "ok")].sort_values("cz")
    ax.plot(d.cz, 100 * d.err_total, marker="o", color=SER[0], label="total (rig reading)")
    ax.plot(d.cz, 100 * d.err_disc, marker="s", color=SER[1], label="discretisation")
    ax.plot(d.cz, 100 * d.err_contact, marker="^", color=SER[2], label="contact")
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("elements through the 1.5 mm (20 x 28 cells)")
    ax.set_ylabel("C1 error (%)")
    ax.set_title("(a) C1 with contact, CG 1000")
    ax.legend(loc="center right")
    ax = axs[1]
    g = s45[s45.step_ms.notna() & s45.err_C3.notna()]
    for var, (col, mk, lab) in VAR.items():
        h = g[g.variant == var]
        ax.loglog(h.step_ms, 100 * h.err_C3.abs(), mk, color=col, label=lab, alpha=0.85, ms=3)
    o = np.argsort(g.step_ms.values)
    best, keep = np.inf, []
    for i in o:
        if g.err_C3.abs().values[i] < best:
            best = g.err_C3.abs().values[i]
            keep.append(i)
    ax.loglog(g.step_ms.values[keep], 100 * g.err_C3.abs().values[keep], color=INK, lw=0.8, ls="--",
              label="Pareto front")
    ax.set_xlabel("ms per step, pinned pad alone")
    ax.set_ylabel("|C3 discretisation error| (%)")
    ax.set_title("(b) static accuracy vs cost")
    ax.legend(loc="upper right", fontsize=5.8)
    ax = axs[2]
    g = s45[s45.variant == "full_corr"]
    for nz, col, mk in zip([1, 2, 3, 6], SER, "os^D"):
        h = g[g.nz_el == nz].sort_values("dx_mm")
        ax.semilogx(h.dx_mm, h.pcg_iters_blockjacobi_gravity, marker=mk, color=col, label=f"{nz} el. thick")
    ax.axhline(100, color=INK, lw=0.7, ls="--")
    ax.text(2.6, 104, "opt.iterations = 100", fontsize=6)
    ax.set_xlabel("in-plane cell size (mm)")
    ax.set_ylabel(r"PCG iterations on $M+h^2K$")
    ax.set_title("(c) flex solve, fixed pad")
    ax.legend(loc="lower left")
    xticks(ax)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "cost.pdf"))
    plt.close(fig)


def fig_crosstalk():
    r = json.load(open(os.path.join(FS, "gt_transform/results.json")))
    fig, axs = plt.subplots(2, 1, figsize=(3.35, 3.3))
    ax = axs[0]
    for nu, col in zip(["0.45", "0.49", "0.499"], SER):
        for bc, ls in (("frictionless", "-"), ("noslip", "--")):
            p = r[f"nu={nu}"]["C6"][bc]["profiles_per_delta"]
            x = 1e3 * np.array(p["x_m"])
            ax.plot(x, p["uz_x0"], ls, color=col, lw=1.0, label=f"$\\nu$={nu}, {bc}")
    for xb in (5.1, 15.3, 25.5):
        ax.axvline(xb, color=MUTED, lw=0.6)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xlim(0, 25.5)
    ax.set_xlabel("x (mm) along y = 0; grey: taxel borders")
    ax.set_ylabel(r"$u_z/\delta$ (+ = into layer)")
    ax.set_title("(a) exact layer under the C6 taxel punch")
    ax.set_ylim(-1.35, 1.12)
    ax.legend(fontsize=5.4, ncol=1, loc="lower right", handlelength=2.6, borderaxespad=0.2)
    ax = axs[1]
    for var in ("full_raw", "full_corr", "trilinear", "quadratic"):
        col, mk, lab = VAR[var]
        nz = 1 if var == "quadratic" else 2
        g = s45[(s45.variant == var) & (s45.nz_el == nz)].sort_values("dx_mm")
        ax.semilogx(g.dx_mm, g["neigh_uz_C6ns"], marker=mk, color=col, label=lab)
    ax.axhline(-0.035, color=INK, lw=0.8, ls="--")
    ax.set_ylim(-0.11, 0.55)
    ax.text(2.2, -0.05, "exact (GT) $-0.035$", fontsize=6, va="top")
    ax.set_xlabel("in-plane cell size (mm)")
    ax.set_ylabel(r"neighbour mean $u_z/\delta$")
    ax.set_title(r"(b) MuJoCo static operator, $\nu$=0.45, no-slip")
    ax.legend(fontsize=5.5, loc="upper left")
    xticks(ax)
    fig.tight_layout(pad=0.3, h_pad=0.6)
    fig.savefig(os.path.join(OUT, "crosstalk.pdf"))
    plt.close(fig)


def fig_stills():
    """crop the true-scale render and the x40 surface of 4 objects from video/stills.png (8 tiles of 960 x 540 px)."""
    im = plt.imread(os.path.join(FS, "video/stills.png"))
    names = ["(a) sphere R 7.7 mm, $F_z$ = 0.87 N", "(b) box edge 15 mm, $F_z$ = 4.05 N",
             "(c) coin R 10.6 mm, $F_z$ = 2.15 N", "(d) 8 mm cube, $F_z$ = 4.18 N"]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 2.3))
    for j, k in enumerate((0, 2, 6, 7)):            # sphere, box edge, coin, cube
        r, c = divmod(k, 2)
        y0, x0 = 540 * r, 960 * c
        rend = im[y0 + 62: y0 + 340, x0 + 12: x0 + 505]
        surf = im[y0 + 84: y0 + 330, x0 + 540: x0 + 920]
        a1, a2 = axs[j // 2, 2 * (j % 2)], axs[j // 2, 2 * (j % 2) + 1]
        a1.imshow(rend, interpolation="lanczos")
        a2.imshow(surf, interpolation="lanczos")
        for a_ in (a1, a2):
            a_.set_axis_off()
        a1.set_title(names[j], fontsize=7, pad=1.5, loc="left")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.0, wspace=0.02, hspace=0.16)
    fig.savefig(os.path.join(OUT, "stills.pdf"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_conv()
    fig_cost()
    fig_crosstalk()
    fig_stills()
    print("ok", sorted(os.listdir(OUT)))
