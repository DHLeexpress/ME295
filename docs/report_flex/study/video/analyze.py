"""Physics checks over the production runs and the "+1,+2,+1" profile plot.

  python analyze.py            (RUNS env: run directory, default prod2)
Writes <RUNS>/checks.json, profile.png, prints a markdown table.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, os.environ.get("RUNS", "prod2"))


def taxel_weights(v0f):
    """Free-face vertex -> taxel averaging weights (boundary vertices shared), as in tactile.PDMSSkin."""
    W = np.zeros((sim.NYT * sim.NXT, len(v0f)))
    tol = 1e-9
    for k, (x, y) in enumerate(v0f[:, :2]):
        for i in range(sim.NXT):
            if -sim.A / 2 + i * sim.DXT - tol <= x <= -sim.A / 2 + (i + 1) * sim.DXT + tol:
                for j in range(sim.NYT):
                    if -sim.B / 2 + j * sim.DYT - tol <= y <= -sim.B / 2 + (j + 1) * sim.DYT + tol:
                        W[j * sim.NXT + i, k] = 1
    return W / np.maximum(W.sum(1, keepdims=True), 1)


def border_jump(U, vx, vy):
    """Largest |delta u_z| between grid neighbours whose connecting edge crosses a taxel border, and elsewhere.
    (Vertices exactly on a border belong to both sides; u is one C0 field, so no jump is possible by construction.)"""
    xs = np.linspace(-sim.A / 2, sim.A / 2, vx)
    ys = np.linspace(-sim.B / 2, sim.B / 2, vy)
    ix = np.floor((xs + sim.A / 2) / sim.DXT - 1e-9).clip(0, sim.NXT - 1)
    iy = np.floor((ys + sim.B / 2) / sim.DYT - 1e-9).clip(0, sim.NYT - 1)
    uz = U[..., 2]
    dx = np.abs(np.diff(uz, axis=0))
    dy = np.abs(np.diff(uz, axis=1))
    cx = (np.diff(ix) != 0)[:, None] & np.ones((1, vy), bool)
    cy = np.ones((vx, 1), bool) & (np.diff(iy) != 0)[None, :]
    cross = max(dx[cx].max(initial=0), dy[cy].max(initial=0))
    inside = max(dx[~cx].max(initial=0), dy[~cy].max(initial=0))
    return float(cross), float(inside)


def main():
    rows = []
    out = {}
    for k in range(8):
        jf = os.path.join(RUNS, f"seg{k}.json")
        if not os.path.exists(jf):
            continue
        r = json.load(open(jf))
        z = np.load(os.path.join(RUNS, f"seg{k}.npz"), allow_pickle=True)
        vx, vy = r["verts"][:2]
        F, ph = z["F"], z["phase"]
        # taxel-sum check on the saved frames (also checked every step in sim.py: taxsum_err_max_N)
        fr_err = max(float(np.abs(z["f_tax"][i].sum((0, 1)) - F[z["f_step"][i]]).max()) for i in range(len(z["f_step"])))
        # frame at the end of the hold (before the slide)
        hold_steps = np.where(ph == 2)[0]
        fi_hold = int(np.searchsorted(z["f_step"], hold_steps[-1], side="right") - 1)
        U = z["f_ufront"][fi_hold].reshape(vx, vy, 3).astype(float)
        jc, ji = border_jump(U, vx, vy)
        sl = (ph == 3) | (ph == 4)
        depth = z["depth"]
        over = float((z["uz_max"][sl] - depth[sl]).max()) if sl.any() else None
        hold = ph == 2
        pen_share = float(np.mean(z["pen_fw"][hold]) / np.mean(depth[hold]))
        lift_end = ph == 5
        res_uz = float(np.abs(z["f_ufront"][-1][:, 2]).max())       # residual after lift-off (last frame)
        rec = dict(seg=k, object=r["object"]["label"], F_target_N=r["object"]["F_target_N"],
                   press_stop=r["press_stop"], Fz_hold_N=r["Fz_hold_N"], Fz_max_N=r["Fz_max_N"],
                   Ft_max_N=r["Ft_max_N"], travel_max_um=r["depth_max_m"] * 1e6,
                   uz_max_um=r["uz_max_m"] * 1e6, uz_end_of_hold_um=float(U[..., 2].max() * 1e6),
                   pen_fw_hold_um=float(np.mean(z["pen_fw"][hold]) * 1e6), pen_share_hold=pen_share,
                   uz_minus_travel_slide_max_um=None if over is None else over * 1e6,
                   residual_uz_after_liftoff_um=res_uz * 1e6,
                   slide_Ft_over_Fz_max=r["slide_Ft_over_Fz_max"], ncon_max=r["ncon_max"],
                   niter_max=r["niter_max"], frac_steps_iter_cap=r["frac_steps_iter_cap"],
                   nan_step=r["nan_step"], taxsum_err_max_N=r["taxsum_err_max_N"], taxsum_err_frames_N=fr_err,
                   servo_vs_contact_err_hold_N=r["act_vs_contact_err_hold_N"], minJ_min=r.get("minJ_min"),
                   ninv_max=r.get("ninv_max"), border_jump_um=jc * 1e6, interior_jump_um=ji * 1e6,
                   wall_total_s=r["wall_total_s"], wall_median_ms=r["wall_median_ms"], warnings=r["warnings"])
        out[k] = rec
        rows.append(rec)
    json.dump(out, open(os.path.join(RUNS, "checks.json"), "w"), indent=1)
    hdr = ("| # | object | target N | stop | Fz hold N | Fz max N | Ft max N | travel um | max uz um | pen_fw hold um "
           "(share) | ncon max | iter-cap frac | min J | NaN | taxel-sum err N | servo-contact err N | wall s |")
    print(hdr)
    print("|---" * (hdr.count("|") - 1) + "|")
    for r in rows:
        print(f"| {r['seg']+1} | {r['object']} | {r['F_target_N']:.1f} | {r['press_stop']} | {r['Fz_hold_N']:.2f} | "
              f"{r['Fz_max_N']:.2f} | {r['Ft_max_N']:.2f} | {r['travel_max_um']:.0f} | {r['uz_max_um']:.0f} | "
              f"{r['pen_fw_hold_um']:.1f} ({r['pen_share_hold']:.2f}) | {r['ncon_max']} | {r['frac_steps_iter_cap']:.3f} | "
              f"{r['minJ_min']:.3f} | {r['nan_step']} | {max(r['taxsum_err_max_N'], r['taxsum_err_frames_N']):.1e} | "
              f"{r['servo_vs_contact_err_hold_N']:.1e} | {r['wall_total_s']:.0f} |")
    profile()


def profile(segs=(0, 7)):
    """u_z along a line through the deepest free-face vertex, across three taxels, at the end of the hold."""
    fig, axs = plt.subplots(len(segs), 1, figsize=(11, 4.2 * len(segs)))
    for ax, k in zip(np.atleast_1d(axs), segs):
        jf = os.path.join(RUNS, f"seg{k}.json")
        if not os.path.exists(jf):
            continue
        r = json.load(open(jf))
        z = np.load(os.path.join(RUNS, f"seg{k}.npz"), allow_pickle=True)
        vx, vy = r["verts"][:2]
        ph = z["phase"]
        hold_steps = np.where(ph == 2)[0]
        fi = int(np.searchsorted(z["f_step"], hold_steps[-1], side="right") - 1)
        U = z["f_ufront"][fi].reshape(vx, vy, 3).astype(float)
        tax = z["f_tax"][fi]
        v0f = z["v0front"]
        W = taxel_weights(v0f)
        utax = (W @ z["f_ufront"][fi].astype(float)).reshape(sim.NYT, sim.NXT, 3)
        i0, j0 = np.unravel_index(np.argmax(U[..., 2]), U[..., 2].shape)
        xs = np.linspace(-sim.A / 2, sim.A / 2, vx)
        ys = np.linspace(-sim.B / 2, sim.B / 2, vy)
        it = int(np.clip(np.floor((xs[i0] + sim.A / 2) / sim.DXT), 0, sim.NXT - 1))
        jt = int(np.clip(np.floor((ys[j0] + sim.B / 2) / sim.DYT), 0, sim.NYT - 1))
        jt = int(np.clip(jt, 1, sim.NYT - 2))          # three taxels along y: jt-1, jt, jt+1
        yl, yh = -sim.B / 2 + (jt - 1) * sim.DYT, -sim.B / 2 + (jt + 2) * sim.DYT
        sel = (ys >= yl - 1e-9) & (ys <= yh + 1e-9)
        ax.plot(ys[sel] * 1e3, U[i0, sel, 2] * 1e6, "o-", ms=3.5, color="C0",
                label=f"u_z of the free-face vertices on the grid line x = {xs[i0]*1e3:.2f} mm (exact nodal values)")
        for jj in (jt - 1, jt, jt + 1):
            a, b = -sim.B / 2 + jj * sim.DYT, -sim.B / 2 + (jj + 1) * sim.DYT
            ax.hlines(utax[jj, it, 2] * 1e6, a * 1e3, b * 1e3, color="C1", lw=3,
                      label="taxel-average u_z (PDMSSkin.deformation weights)" if jj == jt - 1 else None)
            ax.annotate(f"taxel ({it},{jj})\nF_n = {tax[jj, it, 2]:.3f} N\nmean u_z = {utax[jj, it, 2]*1e6:.1f} um",
                        xy=(0.5 * (a + b) * 1e3, 0.97), xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=9)
        for jj in range(jt - 1, jt + 3):
            ax.axvline((-sim.B / 2 + jj * sim.DYT) * 1e3, color="0.5", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(yl * 1e3 - 0.5, yh * 1e3 + 0.5)
        ax.set_ylim(min(-5, U[i0, sel, 2].min() * 1e6 * 1.3), U[i0, sel, 2].max() * 1e6 * 1.6)
        ax.set_xlabel("pad y (mm); dashed = taxel borders")
        ax.set_ylabel("u_z (um), + into the layer")
        ax.set_title(f"segment {k+1}: {r['object']['label']}, end of hold (t = {z['t'][z['f_step'][fi]]*1e3:.0f} ms), "
                     f"F_z = {z['F'][z['f_step'][fi], 2]:.2f} N", fontsize=10)
        ax.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Displacement is one continuous field across taxel borders; force is binned by contact point "
                 "(MuJoCo 3.17 flex, 40x56x3 cells)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "profile.png"), dpi=130)


if __name__ == "__main__":
    main()
