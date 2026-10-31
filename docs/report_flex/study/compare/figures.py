"""Figures fig_*.pdf/.png for the compare task. Inputs: out/static_errors.csv, out/static_orders.json,
out/dynamic_errors.csv, out/config_errors.csv. Run after analyze_static.py and analyze_dynamic.py."""
import json
import os

import matplotlib.ticker  # noqa: F401
import matplotlib
import numpy as np
import pandas as pd

import fig_style as S
import gt

plt = S.plt
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)
S.setup()
st = pd.read_csv(os.path.join(HERE, "out", "static_errors.csv"))
dy = pd.read_csv(os.path.join(HERE, "out", "dynamic_errors.csv"))
ce = pd.read_csv(os.path.join(HERE, "out", "config_errors.csv"))
fits = json.load(open(os.path.join(HERE, "out", "static_orders.json")))["two_param"]
s45 = st[st.nu == 0.45]


def fig_static_convergence():
    """|discretisation error| vs in-plane cell size, 2 elements through the thickness (quadratic: 1 cell)."""
    cases = [("C1", "C1 platen, no-slip"), ("C3", "C3 half platen (edge)"), ("C6ns", "C6 centre-taxel punch")]
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.8))
    for ax, (c, title) in zip(axs, cases):
        for var, (col, mk, lab) in S.VARIANT.items():
            nz = 1 if var == "quadratic" else 2
            g = s45[(s45.variant == var) & (s45.nz_el == nz)].sort_values("dx_mm")
            ax.loglog(g.dx_mm, 100 * g[f"err_{c}"].abs(), marker=mk, color=col, label=lab, ms=4)
        f = [x for x in fits if x["variant"] == "full_corr" and x["nu"] == 0.45 and x["case"] == c]
        if f:
            ax.text(0.04, 0.97, f"full: p = {f[0]['p_inplane']:.2f} (in-plane)\n      q = {f[0]['q_thickness']:.2f} (thickness)",
                    transform=ax.transAxes, fontsize=6.5, color=S.INK2, va="top")
        ax.set_title(title)
        ax.set_xlabel("in-plane cell size (mm)")
    axs[0].set_ylabel("|K / K_exact - 1|  (%)")
    ax = axs[3]
    for var in ("full_corr", "trilinear", "quadratic", "full_raw"):
        col, mk, lab = S.VARIANT[var]
        nz = 1 if var == "quadratic" else 2
        g = s45[(s45.variant == var) & (s45.nz_el == nz)].sort_values("dx_mm")
        ax.semilogx(g.dx_mm, g["neigh_uz_C6ns"], marker=mk, color=col, ms=4)
    ax.axhline(gt.get("C6", 0.45)["neigh_edge_uz"], color=S.INK, lw=1, ls="--")
    ax.text(2.0, gt.get("C6", 0.45)["neigh_edge_uz"] + 0.012, "exact -0.035", fontsize=6.5)
    ax.set_title("C6 cross-talk")
    ax.set_xlabel("in-plane cell size (mm)")
    ax.set_ylabel("edge-neighbour mean u_z / delta")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Static operator (exact BCs, no contact), nu = 0.45: discretisation error vs cell size", x=0.01,
                 ha="left", fontsize=9)
    fig.tight_layout()
    S.save(fig, os.path.join(FIG, "fig_static_convergence"))


def fig_thickness():
    """left: static error vs elements through the thickness (2.55 mm cells); right: dynamic C1 at 20x28xN."""
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.9))
    ax = axs[0]
    g = s45[(s45.variant == "full_corr") & (s45.nt == 4)].sort_values("nz_el")
    for (c, lab), col, mk in zip([("C1", "C1"), ("C3", "C3"), ("C6ns", "C6")], S.SERIES[:3], "os^"):
        ax.plot(g.nz_el, 100 * g[f"err_{c}"], marker=mk, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("elements through the 1.5 mm thickness")
    ax.set_ylabel("K / K_exact - 1  (%)")
    ax.set_title("Static operator, full (corrected), 2.55 mm cells")
    ax.legend()
    ax = axs[1]
    d = dy[(dy.src == "bench_dynamic") & (dy.group == "conv") & (dy.case == "C1") & (dy.cx == 20) & (dy.status == "ok")]
    d = d.sort_values("cz")
    ax.plot(d.cz, 100 * d.err_total, marker="o", color=S.SERIES[0], label="total (what the rig reads)")
    ax.plot(d.cz, 100 * d.err_disc, marker="s", color=S.SERIES[1], label="discretisation (static, same mesh)")
    ax.plot(d.cz, 100 * d.err_contact, marker="^", color=S.SERIES[2], label="contact (K_app / K_static - 1)")
    ax.axhline(0, color=S.MUTED, lw=0.8)
    ax.set_xlabel("elements through the thickness (20 x 28 cells)")
    ax.set_ylabel("error (%)")
    ax.set_title("With contact: C1, robot contact, CG 1000")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    S.save(fig, os.path.join(FIG, "fig_thickness"))


def pareto(x, y):
    o = np.argsort(x)
    best, keep = np.inf, []
    for i in o:
        if y[i] < best:
            best = y[i]
            keep.append(i)
    return np.array(keep)


def fig_pareto():
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0))
    ax = axs[0]
    g = s45[s45.step_ms.notna() & s45.err_C3.notna()]
    for var, (col, mk, lab) in S.VARIANT.items():
        h = g[g.variant == var]
        ax.loglog(h.step_ms, 100 * h.err_C3.abs(), mk, color=col, label=lab, ms=4, alpha=0.85)
    k = pareto(g.step_ms.values, g.err_C3.abs().values)
    ax.loglog(g.step_ms.values[k], 100 * g.err_C3.abs().values[k], color=S.INK, lw=1, ls="--", label="Pareto front")
    ax.set_xlabel("mj_step time, pinned pad alone (ms)")
    ax.set_ylabel("|C3 discretisation error| (%)")
    ax.set_title("Static operator: accuracy vs cost")
    ax.legend(fontsize=6.3)
    ax = axs[1]
    d = dy[(dy.status == "ok") & (dy.case == "C1f") & (dy.nu == 0.45) & (dy.driver == "servo") & dy.err_total.notna()
           & (dy.contact.isin(["stiff", "robotB"])) & (dy.mass_scale == 1) & (dy.split == 1) & (dy.solver == "CG")]
    for var in ("full_corr", "trilinear", "quadratic"):
        col, mk, lab = S.VARIANT[var]
        h = d[d.variant == var]
        ax.loglog(1e-3 * h.us_per_step_median, 100 * h.err_total.abs(), mk, color=col, label=lab, ms=3.5, alpha=0.8)
    # no Pareto line here: the lowest totals are cancellations of discretisation (+) and contact (-) errors
    ax.axvspan(1, 5, color=S.SERIES[3], alpha=0.12, lw=0)
    ax.text(1.1, 0.03, "robot budget 1-5 ms", fontsize=6.5, color=S.INK2, transform=ax.get_xaxis_transform())
    ax.set_xlabel("time per 0.5 ms step, rig with contact (ms)")
    ax.set_ylabel("|C1f total error| (%)")
    ax.set_title("With contact: C1f, true mass, solref 1e-4 or 1e-3")
    ax.legend(fontsize=6.3)
    fig.tight_layout()
    S.save(fig, os.path.join(FIG, "fig_pareto"))


def fig_solver():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9))
    ax = axs[0]
    g = s45[s45.variant == "full_corr"]
    for nz, col, mk in zip([1, 2, 3, 6], S.SERIES[:4], "os^D"):
        h = g[g.nz_el == nz].sort_values("dx_mm")
        ax.semilogx(h.dx_mm, h.pcg_iters_blockjacobi_gravity, marker=mk, color=col, label=f"{nz} el. thick")
    ax.axhline(100, color=S.INK, lw=0.8, ls="--")
    ax.text(3, 104, "opt.iterations = 100", fontsize=6.5)
    ax.set_xlabel("in-plane cell size (mm)")
    ax.set_ylabel("PCG iterations (M + h^2 K)")
    ax.set_title("Flex stiffness solve (pinned pad)")
    ax.legend(fontsize=6.5)
    ax = axs[1]
    d = dy[(dy.src == "bench_dynamic") & (dy.group.isin(["main", "conv"])) & (dy.case == "C1") & (dy.cz == 2)
           & (dy.status == "ok")]
    for it, col, mk in ((100, S.SERIES[0], "o"), (1000, S.SERIES[1], "s")):
        h = d[d.iterations == it].sort_values("cx")
        ax.semilogx(51 / h.cx, h.niter_median, marker=mk, color=col, label=f"CG budget {it}")
    ax.set_xlabel("in-plane cell size (mm)")
    ax.set_ylabel("constraint-solver iterations (median)")
    ax.set_title("C1 no-slip, 2 el. thick")
    ax.legend(fontsize=6.5)
    ax = axs[2]
    for c, col, mk in (("C1", S.SERIES[0], "o"), ("C1f", S.SERIES[1], "s"), ("C6", S.SERIES[2], "^")):
        h = dy[(dy.src == "bench_dynamic") & (dy.group == "conv") & (dy.case == c) & (dy.cz == 2) & (dy.status == "ok")]
        h = h.sort_values("cx")
        y = (1e3 * h.settle_1pct_s.astype(float)).fillna(200.0)   # None = never settled within the 200 ms hold
        ax.semilogx(51 / h.cx, y, marker=mk, color=col, label=c)
    ax.set_xlabel("in-plane cell size (mm)")
    ax.set_ylabel("settling time to 1 % (ms)")
    ax.set_title("After a 10 ms ramp, CG 1000 (200 = never)")
    ax.legend(fontsize=6.5)
    for a_ in axs:
        a_.set_xticks([1, 2, 5, 10])
        a_.set_xticklabels(["1", "2", "5", "10"])
        a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    S.save(fig, os.path.join(FIG, "fig_solver"))


def fig_contact():
    """static-vs-dynamic reconciliation (sd group): K_app / K_static for box (tet contacts) and plane (vertex contacts)."""
    d = dy[(dy.group == "sd") & (dy.status == "ok")].copy()
    d["setting"] = d.contact + np.where(d.mass_scale > 1, ", mass x100", "")
    order = ["stiff", "stiffest", "stiff, mass x100", "stiffest, mass x100"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for ax, mesh in zip(axs, ["10x14x2", "20x28x2", "20x28x3"]):
        cx, cy, cz = map(int, mesh.split("x"))
        h = d[(d.cx == cx) & (d.cz == cz)]
        x = np.arange(len(order))
        for k, (ind, col, lab) in enumerate((("auto", S.SERIES[0], "box (collides with tets)"),
                                             ("plane", S.SERIES[1], "plane (collides with vertices)"))):
            v = [h[(h.setting == o) & (h.indenter == ind)].err_contact.mean() for o in order]
            ax.bar(x + (k - 0.5) * 0.38, 100 * np.array(v), 0.36, color=col, label=lab, edgecolor="white", lw=1)
        ax.axhline(0, color=S.INK, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([o.replace(", ", "\n") for o in order], fontsize=6.5)
        ax.set_title(f"C1f, {mesh} cells")
    axs[0].set_ylabel("K_app / K_static(same mesh) - 1  (%)")
    axs[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    S.save(fig, os.path.join(FIG, "fig_contact"))


def fig_candidates(configs, fname, title):
    """per-case total error (placement mean, 95 % CI over 5 placements for local cases) of chosen configurations."""
    cases = ["C1", "C1f", "C2", "C3", "C4", "C6", "C6f", "C5"]
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    x = np.arange(len(cases))
    w = 0.8 / len(configs)
    for k, (cfg, lab) in enumerate(configs):
        col = S.SERIES[k]
        ys, es = [], []
        for c in cases:
            r = ce[(ce.config == cfg) & (ce.case == c)]
            ys.append(100 * r.err_total.iloc[0] if len(r) else np.nan)
            e = r["err_total_ci95"].iloc[0] if (len(r) and "err_total_ci95" in r and pd.notna(r["err_total_ci95"].iloc[0])) else 0
            es.append(100 * e)
        ax.errorbar(x + (k - (len(configs) - 1) / 2) * w, ys, yerr=es, fmt="os^Dv"[k % 5], color=col, label=lab,
                    capsize=2, ms=4, lw=1)
    ax.axhline(0, color=S.INK, lw=0.8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("K_app / K_exact - 1 (%)  [C5: F(30 um)]")
    ax.set_title(title)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    S.save(fig, os.path.join(FIG, fname))


if __name__ == "__main__":
    import sys
    fig_static_convergence()
    fig_thickness()
    fig_pareto()
    fig_solver()
    fig_contact()
    if os.path.exists(os.path.join(HERE, "out", "chosen.json")):
        ch = json.load(open(os.path.join(HERE, "out", "chosen.json")))
        for key, v in ch["figures"].items():
            fig_candidates([tuple(c) for c in v["configs"]], key, v["title"])
    print("figures written to", FIG)
