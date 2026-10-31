# Figures from results.json (run collect.py first). PNGs in plots/.
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
PL = os.path.join(HERE, "plots")
os.makedirs(PL, exist_ok=True)

# validated categorical slots (dataviz reference palette, light): fixed order, never cycled
COL = {"full_corr": "#2a78d6", "full_raw": "#eb6834", "trilinear": "#1baf7a", "quadratic": "#eda100"}
MRK = {"full_corr": "o", "full_raw": "s", "trilinear": "^", "quadratic": "D"}
LAB = {"full_corr": 'dof="full" (inputs corrected for F1)', "full_raw": 'dof="full" (uncorrected, as on the robot)',
       "trilinear": 'dof="trilinear" (cellcount)', "quadratic": 'dof="quadratic" (cellcount)'}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # ordinal ramp (step 250 ->)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.6, "lines.markersize": 5.5, "figure.dpi": 130})

R = json.load(open(os.path.join(HERE, "results.json")))
R = [r for r in R if r.get("status") == "ok"]
BY = {r["id"]: r for r in R}


def kval(r, case):
    c = r["cases"].get(case)
    if c is None:
        return None
    return c.get("K_area_mean_N_per_um") if case == "UP" else c.get("K_N_per_um")


def ref(case, nu):
    """(value, label): ground truth if the GT agents provided it, else the finest quadratic flex."""
    for r in R:
        e = r.get("error_vs_reference", {}).get(case)
        if r["config"]["nu"] == nu and e and "K_gt_N_per_um" in e:
            return e["K_gt_N_per_um"], "ground truth (" + e["gt_source"].split(":")[0] + ")"
    best = None
    for r in R:
        c = r["config"]
        if c["variant"] == "quadratic" and c["nu"] == nu and kval(r, case) is not None:
            n = r["model"]["n_points"]
            if best is None or n > best[0]:
                best = (n, kval(r, case), r["id"])
    return (best[1], "finest quadratic flex (" + best[2].replace("_nu" + str(nu), "") + ")") if best else (None, None)


def rid(v, nt, z, nu):
    return f"{v}_nt{nt}_{'nz' if v.startswith('full') else 'cz'}{z}_nu{nu}"


NTS = [1, 2, 4, 6, 8, 12, 16]
ZFINE = {"full_corr": 7, "full_raw": 7, "trilinear": 6, "quadratic": 3}
ZLIST = {"full_corr": [2, 3, 4, 5, 6, 7], "full_raw": [2, 3, 4, 5, 6, 7], "trilinear": [1, 2, 3, 4, 6], "quadratic": [1, 2, 3]}


def zintervals(v, z):
    return z - 1 if v.startswith("full") else (z if v == "trilinear" else 2 * z)


def fig_convergence(nu, cases, fname, title):
    fig, axes = plt.subplots(len(cases), 2, figsize=(10.5, 2.55 * len(cases)), squeeze=False)
    for i, (case, cname) in enumerate(cases):
        K0, lab0 = ref(case, nu)
        for j in range(2):
            ax = axes[i, j]
            for v in ("full_raw", "full_corr", "trilinear", "quadratic"):
                xs, ys = [], []
                if j == 0:
                    for nt in NTS:
                        r = BY.get(rid(v, nt, ZFINE[v], nu))
                        if r and kval(r, case) is not None:
                            xs.append(10.2 / nt); ys.append(kval(r, case) / K0)
                else:
                    ntf = max([nt for nt in NTS if any(BY.get(rid(v, nt, z, nu)) for z in ZLIST[v])] or [8])
                    ntf = min(ntf, 8)
                    for z in ZLIST[v]:
                        r = BY.get(rid(v, ntf, z, nu))
                        if r and kval(r, case) is not None:
                            xs.append(zintervals(v, z)); ys.append(kval(r, case) / K0)
                if xs:
                    ax.plot(xs, ys, color=COL[v], marker=MRK[v], label=LAB[v], markeredgecolor="white", markeredgewidth=0.8)
            ax.axhline(1.0, color=INK2, lw=1.0)
            ax.set_yscale("log")
            if j == 0:
                ax.set_xscale("log"); ax.invert_xaxis()
                ax.set_xticks([10.2, 5.1, 2.55, 1.7, 1.275, 0.85, 0.6375])
                ax.set_xticklabels(["10.2", "5.1", "2.55", "1.7", "1.28", "0.85", "0.64"])
                ax.set_xlabel("in-plane element size (mm)  [finest through-thickness of each variant]")
            else:
                ax.set_xlabel("node intervals through the 1.5 mm thickness  [in-plane 1.275 mm]")
            ax.set_ylabel(f"K / K_ref")
            yl = ax.get_ylim(); ax.set_ylim(min(yl[0], 0.95), max(yl[1], 1.05))
            from matplotlib.ticker import FuncFormatter, LogLocator
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}" if y in (0.9, 1.1, 1.2, 1.5, 2, 3, 5) else ""))
            if j == 0:
                ax.set_title(f"{cname}: K_ref = {K0:.4g} N/um, {lab0}", fontsize=8.5, color=INK, loc="left")
            else:
                ax.set_title(f"{case}: through-thickness refinement (same K_ref)", fontsize=8.5, color=INK, loc="left")
    axes[0, 0].legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.0, 1.55), ncol=2)
    fig.suptitle(title, fontsize=10.5, color=INK, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(PL, fname)); plt.close(fig)


def heat(ax, M, cmap, vmin, vmax, fmt, title):
    A = np.array(M).T     # [j (y), i (x)]
    ax.imshow(A, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=(-25.5, 25.5, -35.5, 35.5), aspect="equal")
    for i in range(5):
        for j in range(7):
            x = -25.5 + 10.2 * (i + 0.5); y = -35.5 + 71 / 7 * (j + 0.5)
            val = A[j, i]
            rel = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
            ax.text(x, y, fmt.format(val), ha="center", va="center", fontsize=6.3, color="white" if rel > 0.6 else INK)
    ax.set_xticks([-25.5, 0, 25.5]); ax.set_yticks([-35.5, 0, 35.5]); ax.grid(False)
    ax.set_title(title, fontsize=7.8, color=INK)
    ax.tick_params(labelsize=6.5)


def fig_taxels(nu, fname):
    seq = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    div = LinearSegmentedColormap.from_list("div", ["#d95926", "#f4c2a8", "#efeeea", "#9ec5f4", "#1c5cab"])
    models = [(rid("full_raw", 2, 3, nu), "robot mesh 11x15x3, full, uncorrected"),
              (rid("full_corr", 2, 3, nu), "11x15x3, full, corrected"),
              (rid("full_corr", 8, 7, nu), "41x57x7, full, corrected"),
              (None, "reference: finest quadratic")]
    best = None
    for r in R:
        c = r["config"]
        if c["variant"] == "quadratic" and c["nu"] == nu and (best is None or r["model"]["n_points"] > best["model"]["n_points"]):
            best = r
    models[3] = (best["id"], "finest quadratic " + best["id"].replace(f"_nu{nu}", ""))
    rows = [("C1", "force", "C1 force per taxel (N/um)"), ("C3", "force", "C3 force per taxel (N/um)"),
            ("C3", "disp", "C3 mean u_z / delta"), ("C6ns", "disp", "C6 (no-slip) mean u_z / delta")]
    fig, axes = plt.subplots(len(rows), len(models), figsize=(2.3 * len(models), 3.05 * len(rows)))
    for i, (case, kind, rt) in enumerate(rows):
        for j, (mid, mt) in enumerate(models):
            r = BY.get(mid); ax = axes[i, j]
            if r is None:
                ax.axis("off"); continue
            t = r["cases"][case]["taxel"]
            if kind == "force":
                M = np.array(t["force_N_per_um"]); vmax = max(1e-12, np.abs(M).max())
                heat(ax, M, seq, 0, vmax, "{:.3f}", f"{mt}\n{rt}; total {M.sum():.3f}")
            else:
                M = np.array(t["mean_uz_over_delta"])
                heat(ax, M, div, -1, 1, "{:.2f}", f"{mt}\n{rt}")
    fig.suptitle(f"Per-taxel maps, nu = {nu} (x across the blade, y along the finger; taxel rows j=0..6 bottom to top)",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(PL, fname)); plt.close(fig)


def fig_conditioning(fname):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4))
    for col, nu in enumerate((0.45, 0.49)):
        for row, key in enumerate(("pcg", "pcg_mass_prec")):
            ax = axes[row, col]
            for k, nz in enumerate([2, 3, 4, 5, 6, 7]):
                xs, ys = [], []
                for nt in NTS:
                    r = BY.get(rid("full_corr", nt, nz, nu))
                    if not r:
                        continue
                    it = r["conditioning"][key]["random"].get("iterations")
                    if it:
                        xs.append(10.2 / nt); ys.append(it)
                if xs:
                    ax.plot(xs, ys, color=BLUES[k], marker="o", markeredgecolor="white", markeredgewidth=0.8,
                            label=f"{nz-1} element{'s' if nz > 2 else ''} through thickness (dz = {1.5/(nz-1):.3g} mm)")
            r = BY.get(rid("full_raw", 2, 3, nu))
            if r:
                it = r["conditioning"][key]["random"].get("iterations")
                ax.plot([5.1], [it], marker="s", color=COL["full_raw"], ls="none", markersize=8, markeredgecolor="white",
                        label="robot 11x15x3, uncorrected")
            ax.axhline(100, color=INK2, lw=1.0, ls="--")
            ax.text(10.0, 100, "default opt.iterations = 100", fontsize=7, color=INK2, va="bottom")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
            ax.set_xticks([10.2, 5.1, 2.55, 1.7, 1.275, 0.85, 0.6375])
            ax.set_xticklabels(["10.2", "5.1", "2.55", "1.7", "1.28", "0.85", "0.64"])
            ax.set_xlabel("in-plane element size (mm)")
            ax.set_ylabel("PCG iterations to 1e-8 (random rhs)")
            pre = "3x3 vertex-block preconditioner\n(pad on a fixed body, as mj_effSolve)" if key == "pcg" else \
                "mass-only preconditioner\n(what MuJoCo uses when a joint is upstream)"
            ax.set_title(f'dof="full" corrected, nu={nu}: {pre}', fontsize=8.3, loc="left", color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=4, fontsize=7.5, bbox_to_anchor=(0.01, 0.0))
    # stiffness of the time step: h*omega_max and cond(M + h^2 K) vs node spacing through the thickness
    for row, key in enumerate(("h_omega_max", "cond_A")):
        ax = axes[row, 2]
        for v in ("full_raw", "full_corr", "trilinear", "quadratic"):
            for nu, fill in ((0.45, True), (0.49, False)):
                xs, ys = [], []
                for r in R:
                    c = r["config"]
                    if c["variant"] == v and c["nu"] == nu and r["conditioning"].get(key):
                        xs.append(c["node_dz_mm"]); ys.append(r["conditioning"][key])
                ax.plot(xs, ys, ls="none", marker=MRK[v], color=COL[v], markerfacecolor=COL[v] if fill else "white",
                        label=(LAB[v] if nu == 0.45 else None), markeredgewidth=0.9, alpha=0.9)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("node spacing through the thickness dz (mm)")
        from matplotlib.ticker import NullFormatter
        ax.set_xticks([1.5, 0.75, 0.5, 0.375, 0.25, 0.125]); ax.set_xticklabels(["1.5", "0.75", "0.5", "0.375", "0.25", "0.125"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        if key == "h_omega_max":
            zz = np.array([0.125, 1.5])
            for nu, ls in ((0.45, "-"), (0.49, ":")):
                cp = math.sqrt(lame(1e6, nu)[2] / 965.0)
                ax.plot(zz, 5e-4 * 2 * cp / (zz * 1e-3), color=INK2, lw=1, ls=ls, label=f"2 h c_p / dz, nu={nu}")
            ax.axhline(2.0, color=INK2, lw=1.0, ls="--"); ax.text(1.4, 2.1, "explicit stability limit h*omega = 2", fontsize=7, color=INK2)
            ax.set_ylabel("h * omega_max  (h = 0.5 ms)")
            ax.set_title("highest natural frequency x time step\n(filled nu=0.45, open nu=0.49)", fontsize=8.3, loc="left", color=INK)
            ax.legend(fontsize=6.5, loc="upper left")
        else:
            ax.set_ylabel("cond(M + h^2 K)")
            ax.set_title("condition number of the implicit metric (unpreconditioned)", fontsize=8.3, loc="left", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(PL, fname)); plt.close(fig)


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu)), E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))


def fig_tradeoff(nu, fname):
    cases = [("C1", "C1 full platen, no-slip"), ("C3", "C3 half platen (taxel rows 4-6)"), ("C6ns", "C6 centre-taxel punch, no-slip")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for j, (case, cn) in enumerate(cases):
        K0, lab0 = ref(case, nu)
        for v in ("full_raw", "full_corr", "trilinear", "quadratic"):
            pts = []
            for r in R:
                c = r["config"]
                if c["variant"] != v or c["nu"] != nu or kval(r, case) is None:
                    continue
                if r["id"] == ref(case, nu)[1]:
                    continue
                e = abs(kval(r, case) / K0 - 1)
                pts.append((r["conditioning"]["nv"], e, r["conditioning"]["pcg"]["random"].get("iterations") or np.nan))
            pts = np.array(sorted(pts))
            if len(pts) == 0:
                continue
            for i, (xcol, ax) in enumerate(((0, axes[0, j]), (2, axes[1, j]))):
                ax.plot(pts[:, xcol], np.maximum(pts[:, 1], 1e-5), ls="none", marker=MRK[v], color=COL[v], label=LAB[v],
                        markeredgecolor="white", markeredgewidth=0.6, alpha=0.9)
        for i, ax in enumerate((axes[0, j], axes[1, j])):
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_ylabel("|K / K_ref - 1|")
            ax.set_xlabel("degrees of freedom of the pinned pad" if i == 0 else "block-Jacobi PCG iterations to 1e-8 (random rhs)")
            ax.set_title(f"{cn}, nu={nu}", fontsize=8.3, loc="left", color=INK)
            if i == 1:
                ax.axvline(100, color=INK2, lw=1.0)
        axes[0, j].text(0.02, 0.02, f"K_ref = {K0:.4g} N/um: {lab0}", transform=axes[0, j].transAxes, fontsize=6.5, color=INK2)
    axes[0, 0].legend(fontsize=6.8, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    fig.tight_layout()
    fig.savefig(os.path.join(PL, fname)); plt.close(fig)


def fig_cost(fname):
    T = [r for r in R if r.get("timing")]
    if not T:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for v in ("full_raw", "full_corr", "trilinear", "quadratic"):
        xs = [r["timing"]["nv"] for r in T if r["config"]["variant"] == v]
        for ax, key, sc in ((axes[0], "step_s_median", 1e3), (axes[1], "compile_s", 1.0)):
            ys = [r["timing"][key] * sc for r in T if r["config"]["variant"] == v]
            ax.plot(xs, ys, ls="none", marker=MRK[v], color=COL[v], label=LAB[v], markeredgecolor="white", markeredgewidth=0.6)
    for ax, yl in ((axes[0], "mj_step wall time (ms), no contact"), (axes[1], "MuJoCo compile time (s)")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("degrees of freedom (pinned pad)"); ax.set_ylabel(yl)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_title("cost per step (pinned pad, gravity, opt.iterations 100)", fontsize=8.5, loc="left")
    axes[1].set_title("compile time (grows ~ quadratically with the number of flex bodies)", fontsize=8.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(PL, fname)); plt.close(fig)


if __name__ == "__main__":
    CASES = [("C1", "C1 rigid platen, whole face, no-slip"), ("C1f", "C1f same, frictionless"),
             ("C3", "C3 platen over taxel rows 4-6, no-slip"), ("C6ns", "C6 punch on the centre taxel, no-slip"),
             ("C6f", "C6 punch on the centre taxel, frictionless"), ("C4p_a5mm", "C4 circular punch a = 5 mm (penalty)"),
             ("UP", "uniform pressure, K = F / mean(u_z)")]
    for nu in (0.45, 0.49):
        fig_convergence(nu, CASES, f"convergence_nu{nu}.png",
                        f"Static stiffness of the pinned 51 x 71 x 1.5 mm pad vs discretisation (exact BCs, no contact), nu = {nu}")
        fig_taxels(nu, f"taxel_maps_nu{nu}.png")
        fig_tradeoff(nu, f"tradeoff_nu{nu}.png")
    fig_conditioning("conditioning.png")
    fig_cost("cost.png")
    print("plots written to", PL)
