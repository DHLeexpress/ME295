"""plots.py -- PNG figures from results.json, runs/ and runs_c4/ (run postprocess.py first)."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import postprocess as pp

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)
C = {0.45: "#2a78d6", 0.49: "#eb6834", 0.499: "#1baf7a"}          # categorical slots 1-3, fixed per nu
LS = {"full": "-", "mixed": "--"}
MK = {"full": "o", "mixed": "s"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("divbr", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a9a", "#b52b2b"])
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2, "legend.frameon": False})
RES = json.load(open(os.path.join(HERE, "results.json")))
_trf = os.path.join(HERE, "..", "gt_transform", "results.json")
TR = json.load(open(_trf)) if os.path.exists(_trf) else None
TITLE = {"C1": "C1 full platen, no-slip (normal)", "C1f": "C1f full platen, frictionless (normal)",
         "C2": "C2 full platen, no-slip (shear y)", "C3": "C3 platen y >= 5.07 mm, no-slip",
         "C6": "C6 centre-taxel punch, no-slip", "C6f": "C6f centre-taxel punch, frictionless"}


def fig_convergence():
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, case in zip(axs.ravel(), TITLE):
        for nu_key, e in RES["cases"].get(case, {}).items():
            nu = float(nu_key[2:])
            Kst = e["K_N_per_um"]
            for form, f in e["forms"].items():
                d = np.abs(np.asarray(f["K_N_per_um_levels"]) - Kst) / Kst
                ok = d > 0
                ax.loglog(np.asarray(f["n_dof"])[ok], d[ok], LS[form], marker=MK[form], ms=6, color=C[nu],
                          label="%s, nu=%g" % (form, nu))
        ax.set_title(TITLE[case], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("DOF (symmetry-reduced model)")
        ax.set_ylabel("|K_h - K*| / K*")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Q2 FEM convergence of the stiffness (K* = Richardson value of the best form; "
                 "the best form's last point is ~0 by construction)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_convergence.png"), dpi=150)
    plt.close(fig)


def pm_profile(nu, x, y=0.0, nterm=2001):
    lam, G, M, Kb = pp.moduli(nu)
    a, b, t = pp.A, pp.B, pp.T
    l2 = 12 * G / (Kb * t ** 2); c = 12 * G / t ** 3
    n = np.arange(1, 2 * nterm, 2, dtype=float)
    al = n * np.pi / (2 * a); k = np.sqrt(al ** 2 + l2)
    cn = 4 * c * (-1) ** ((n - 1) / 2) / (n * np.pi)
    return np.sum((cn / k ** 2 * (1 - np.exp(-k * (b - abs(y))) * (1 + np.exp(-2 * k * abs(y))) / (1 + np.exp(-2 * k * b))))[None, :] * np.cos(np.outer(x, al)), 1)


def fig_c1_traction():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, nu in zip(axs, (0.45, 0.49, 0.499)):
        lam, G, M, Kb = pp.moduli(nu)
        for case, ls in (("C1", "-"), ("C1f", "--")):
            e = RES["cases"].get(case, {}).get("nu%.3f" % nu)
            if not e:
                continue
            x = np.asarray(e["profile_x_mm"]); tz = np.asarray(e["t_free_at_y0_MPa_per_mm"])[:, 2]
            ax.plot(x, tz, ls, color=C[nu] if case == "C1" else INK, lw=1.6, label="FEM %s (y = 0)" % case)
        xs = np.linspace(-pp.A, pp.A, 801)
        ax.plot(xs, pm_profile(nu, xs), ":", color="#7a4fc9", lw=2,
                label="pressure method (Kelly & Van Engelen 2015)")
        ax.axhline(M / pp.T, color=MUTED, lw=1, ls="-.", label="Winkler / confined M/t (comparison only)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("t_z per unit delta (MPa/mm = kPa/um)")
        ax.set_title("nu = %g" % nu, loc="left", fontsize=9, color=INK)
        ax.set_ylim(0, max(M / pp.T, np.percentile(tz, 99)) * 1.15)
    for ax in axs:
        ax.legend(fontsize=6.5, loc="lower center")
    fig.suptitle("C1/C1f: platen traction across the pad at y = 0 (edge singularities clipped)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_C1_traction.png"), dpi=150)
    plt.close(fig)


def fig_profiles():
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    for row, case in enumerate(("C3", "C6", "C6f")):
        for nu_key, e in RES["cases"].get(case, {}).items():
            nu = float(nu_key[2:])
            x = np.asarray(e["profile_x_mm"]); y = np.asarray(e["profile_y_mm"])
            axs[row, 0].plot(y, np.asarray(e["u_free_at_x0_per_delta"])[:, 2], color=C[nu], label="nu = %g" % nu)
            axs[row, 1].plot(x, np.asarray(e["u_free_at_y0_per_delta"])[:, 2], color=C[nu], label="nu = %g" % nu)
            if case in ("C6", "C6f") and TR is not None:
                try:
                    p = TR["nu=%g" % nu]["C6"]["noslip" if case == "C6" else "frictionless"]["profiles_per_delta"]
                    lab = "infinite layer, exact (gt_transform)" if nu == 0.45 else None
                    axs[row, 0].plot(np.asarray(p["y_m"]) * 1e3, p["uz_0y"], ":", color=INK, lw=1.2, label=lab)
                    axs[row, 1].plot(np.asarray(p["x_m"]) * 1e3, p["uz_x0"], ":", color=INK, lw=1.2, label=lab)
                except KeyError:
                    pass
        for ax, lab in ((axs[row, 0], "y (mm), at x = 0"), (axs[row, 1], "x (mm), at y = 0")):
            ax.set_xlabel(lab); ax.set_ylabel("u_z / delta (free face)")
            ax.set_title(TITLE[case], loc="left", fontsize=9, color=INK)
            for v in (pp.YT if "y" in lab else pp.XT):
                ax.axvline(v, color=GRID, lw=1, zorder=0)
        axs[row, 0].legend(fontsize=7)
    fig.suptitle("Free-face normal displacement profiles (thin vertical lines: taxel boundaries); "
                 "u_z = 1 under the platen/punch", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_profiles.png"), dpi=150)
    plt.close(fig)


def taxel_panel(ax, M, title, fmt, cmap, norm=None):
    M = np.asarray(M)
    im = ax.imshow(M, origin="lower", cmap=cmap, norm=norm, extent=(-25.5, 25.5, -35.5, 35.5), aspect="equal")
    for j in range(7):
        for i in range(5):
            xc = -25.5 + 10.2 * (i + 0.5); yc = -35.5 + 71 / 7 * (j + 0.5)
            v = M[j, i]
            rgb = im.cmap(im.norm(v))[:3]
            ax.text(xc, yc, fmt % v, ha="center", va="center", fontsize=6.5,
                    color="white" if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 0.5 else INK)
    ax.set_xticks(pp.XT); ax.set_yticks(pp.YT); ax.grid(False)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8, loc="left", color=INK)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8)


def fig_taxels(case, nu=0.45):
    e = RES["cases"].get(case, {}).get("nu%.3f" % nu)
    if not e:
        return
    tf = np.asarray(e["taxel_force_N_per_um"]); tb = np.asarray(e["taxel_force_on_blade_N_per_um"])
    td = np.asarray(e["taxel_mean_disp_per_delta"])
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    taxel_panel(axs[0], tf[:, :, 2] * 1e3, "contact F_z per taxel (mN/um)", "%.2f", SEQ)
    taxel_panel(axs[1], tb[:, :, 2] * 1e3, "force on blade F_z per taxel (mN/um)", "%.2f", SEQ)
    taxel_panel(axs[2], td[:, :, 2], "mean free-face u_z / delta per taxel", "%.3f", SEQ)
    v = np.max(np.abs(tb[:, :, 1])) * 1e3 or 1.0
    taxel_panel(axs[3], tb[:, :, 1] * 1e3, "force on blade F_y per taxel (mN/um)", "%.3f", DIV, TwoSlopeNorm(0, -v, v))
    fig.suptitle("%s, nu = %g (finest mesh, %s form, %d DOF); per unit platen displacement" % (
        TITLE[case], nu, e["best_form"], e["finest_n_dof"]), fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_taxels_%s.png" % case), dpi=150)
    plt.close(fig)


def fig_c4():
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "runs_c4", "*.json"))]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for d in rows:
        if d["R_out_mm"] != 25.5 or d["m"] != 8 or d["bc"] != "frictionless":
            continue
        if (d["form"] == "full") != (d["nu"] < 0.499):
            continue
        r = np.asarray(d["r_mm"])
        ax = axs[0] if d["a_mm"] == 5.0 else axs[1]
        ax.plot(r, d["uz_free_per_delta"], color=C[round(d["nu"], 3)], label="nu = %g (%s)" % (d["nu"], d["form"]))
    for ax, a in zip(axs, (5.0, 2.5)):
        ax.set_xlim(0, 4 * a)
        ax.set_xlabel("r (mm)"); ax.set_ylabel("u_z / delta (free face)")
        ax.set_title("C4 frictionless flat punch a = %g mm (axisymmetric Q2, R_out = 25.5 mm)" % a, fontsize=9,
                     loc="left", color=INK)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_C4_profiles.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_convergence(); fig_c1_traction(); fig_profiles(); fig_c4()
    for c in ("C3", "C6", "C6f", "C1", "C2"):
        fig_taxels(c)
    print("figures in", FIG)
