"""Render video frames (1920 x 1080 PNG) from runs/seg<k>.npz.

  python render.py prep                       -> runs/scales.json (common colour scales over all segments)
  python render.py title                      -> frames/title_XXXX.png (caption card)
  python render.py frames <seg> <i0> <i1>     -> frames/seg<k>_XXXX.png for frame indices i0 <= i < i1

Panels: (1) MuJoCo close-up (true scale, taxel borders drawn on the deformed surface), (2) free-face displacement
field, all three components exaggerated by EXAG, coloured by u_z (um), (3) 5 x 7 taxel normal force (N) with shear
arrows, (4) total normal and tangential contact force and indentation vs time.
"""
import glob
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import mujoco

import sim

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, os.environ.get("RUNS", "prod2"))
FR = os.path.join(HERE, os.environ.get("FRAMES", "frames"))
EXAG = float(os.environ.get("EXAG", "40"))
WIN_HX, WIN_HY = 0.0153, 0.0203        # half sizes (pad x, pad y) of the panel-2 close-up window (m)
from matplotlib.colors import LightSource
LS = LightSource(azdeg=200, altdeg=60)
DPI = 100
PH_NAMES = ["approach", "press", "hold", "slide", "hold", "lift-off"]
MAT = "MuJoCo 3.17 flex (dof=full, linear tets), linear elastic, E = 1 MPa, nu = 0.45, no material damping"


def load(k):
    z = np.load(os.path.join(RUNS, f"seg{k}.npz"), allow_pickle=True)
    rec = json.load(open(os.path.join(RUNS, f"seg{k}.json")))
    return z, rec


def prep():
    uzmin, uzmax, fmax, ftmax, Fz, Ft = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for f in sorted(glob.glob(os.path.join(RUNS, "seg*.npz"))):
        z = np.load(f, allow_pickle=True)
        u = z["f_ufront"]
        uzmin, uzmax = min(uzmin, float(u[..., 2].min())), max(uzmax, float(u[..., 2].max()))
        fmax = max(fmax, float(z["f_tax"][..., 2].max()))
        ftmax = max(ftmax, float(np.hypot(z["f_tax"][..., 0], z["f_tax"][..., 1]).max()))
        Fz = max(Fz, float(z["F"][:, 2].max()))
        Ft = max(Ft, float(np.hypot(z["F"][:, 0], z["F"][:, 1]).max()))
    sc = dict(uz_min_um=uzmin * 1e6, uz_max_um=uzmax * 1e6, taxel_Fz_max_N=fmax, taxel_Ft_max_N=ftmax,
              Fz_max_N=Fz, Ft_max_N=Ft, exaggeration=EXAG)
    json.dump(sc, open(os.path.join(RUNS, "scales.json"), "w"), indent=1)
    print(sc)


def bilinear(U, vx, vy, px, py):
    """Interpolate a vertex field U (vx, vy, c) at pad points (px, py)."""
    fx = np.clip((px + sim.A / 2) / (sim.A / (vx - 1)), 0, vx - 1 - 1e-9)
    fy = np.clip((py + sim.B / 2) / (sim.B / (vy - 1)), 0, vy - 1 - 1e-9)
    i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
    a, b = (fx - i)[:, None], (fy - j)[:, None]
    return (U[i, j] * (1 - a) * (1 - b) + U[i + 1, j] * a * (1 - b) + U[i, j + 1] * (1 - a) * b
            + U[i + 1, j + 1] * a * b)


def taxel_lines(n=120):
    """Polylines (pad frame, z = 0) of the taxel borders."""
    out = []
    for i in range(sim.NXT + 1):
        x = -sim.A / 2 + i * sim.DXT
        y = np.linspace(-sim.B / 2, sim.B / 2, n)
        out.append(np.c_[np.full(n, x), y])
    for j in range(sim.NYT + 1):
        y = -sim.B / 2 + j * sim.DYT
        x = np.linspace(-sim.A / 2, sim.A / 2, n)
        out.append(np.c_[x, np.full(n, y)])
    return out


class MjPanel:
    def __init__(self, k, rec, W=960, H=600):
        xml = open(os.path.join(RUNS, f"seg{k}.xml")).read()
        xml = xml.replace("<asset>", '<asset>\n  <texture type="skybox" builtin="gradient" rgb1="0.93 0.94 0.96" '
                          'rgb2="0.62 0.66 0.72" width="512" height="512"/>', 1)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.m.flex_rgba[0] = [0.98, 0.72, 0.42, 1.0]
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, height=H, width=W, max_geom=20000)
        o = rec["object"]
        x0, y0 = o["xy0"]
        sd = np.array(o["slide_dir"]) * o["slide_m"] / 2
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # pad frame -> world: (x, -y, -z); camera looks along world +x (image right = pad +y)
        self.cam.lookat[:] = [x0 + sd[0], -(y0 + sd[1]), -0.004]
        self.cam.distance = 0.040
        self.cam.azimuth = 0.0
        self.cam.elevation = -24.0
        self.opt = mujoco.MjvOption()
        self.vx, self.vy = rec["verts"][0], rec["verts"][1]
        self.lines = taxel_lines()

    def render(self, qpos, ufront):
        d = self.d
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_flex(self.m, d)
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        scn = self.r.scene
        U = ufront.reshape(self.vx, self.vy, 3).astype(float)
        for L in self.lines:
            u = bilinear(U, self.vx, self.vy, L[:, 0], L[:, 1])
            p = np.c_[L, np.zeros(len(L))] + u
            p[:, 2] -= 0.04e-3                            # 40 um toward the viewer (out of the free face)
            pw = p @ sim.RW
            for a, b in zip(pw[:-1], pw[1:]):
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                    np.array([0.1, 0.1, 0.1, 1.0], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.10e-3, a, b)
                scn.ngeom += 1
        return self.r.render()


def compose(k, i, z, rec, sc, mjp, out):
    o = rec["object"]
    t = z["t"]
    s = int(z["f_step"][i])
    ph = int(z["phase"][s])
    F = z["F"]
    vx, vy = rec["verts"][0], rec["verts"][1]
    U = z["f_ufront"][i].reshape(vx, vy, 3).astype(float)
    tax = z["f_tax"][i]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=DPI)
    fig.patch.set_facecolor("white")
    fig.text(0.01, 0.99, f"Segment {k+1}/8: {o['label']}", fontsize=19, weight="bold", va="top")
    fig.text(0.01, 0.952, MAT + f";  pad 51 x 71 x 1.5 mm, {rec['cells'][0]}x{rec['cells'][1]}x{rec['cells'][2]} "
             f"cells; back face bonded; mu = 1; target load {o['F_target_N']:.1f} N or indentation cap "
             f"{rec['depth_cap_m']*1e6:.0f} um", fontsize=11.5, va="top")
    fig.text(0.99, 0.99, f"t = {t[s]*1e3:6.1f} ms  ({PH_NAMES[ph]})", fontsize=18, ha="right", va="top",
             family="monospace")
    # (1) MuJoCo render
    img = mjp.render(z["f_qpos"][i], U)
    ax1 = fig.add_axes([0.005, 0.36, 0.49, 0.53])
    ax1.imshow(img)
    ax1.set_axis_off()
    ax1.set_title("(1) MuJoCo render, TRUE scale (dark lines: taxel borders drawn on the deformed face)", fontsize=12, pad=4)
    # (2) exaggerated surface
    ax2 = fig.add_axes([0.50, 0.33, 0.46, 0.58], projection="3d")
    # close-up window (pad frame) around the object's mid-slide position, true aspect ratio in all three axes
    cxw, cyw = np.array(o["xy0"]) + np.array(o["slide_dir"]) * o["slide_m"] / 2
    hx, hy = WIN_HX, WIN_HY
    cxw = float(np.clip(cxw, -sim.A / 2 + hx, sim.A / 2 - hx))
    cyw = float(np.clip(cyw, -sim.B / 2 + hy, sim.B / 2 - hy))
    xs = np.linspace(-sim.A / 2, sim.A / 2, vx)
    ys = np.linspace(-sim.B / 2, sim.B / 2, vy)
    ii = np.where(np.abs(xs - cxw) <= hx + 1e-9)[0]
    jj = np.where(np.abs(ys - cyw) <= hy + 1e-9)[0]
    Uw = U[ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1]
    Xp, Yp = np.meshgrid(xs[ii], ys[jj], indexing="ij")
    Xs = (Yp + EXAG * Uw[..., 1]) * 1e3                  # plot X = pad y (mm)
    Ys = (Xp + EXAG * Uw[..., 0]) * 1e3                  # plot Y = pad x (mm)
    Zs = -EXAG * Uw[..., 2] * 1e3                        # up = out of the free face
    norm = TwoSlopeNorm(vmin=min(-1.0, sc["uz_min_um"]), vcenter=0.0, vmax=sc["uz_max_um"])
    cmap = plt.get_cmap("RdBu_r")
    uzc = 0.25 * (Uw[:-1, :-1, 2] + Uw[1:, :-1, 2] + Uw[:-1, 1:, 2] + Uw[1:, 1:, 2]) * 1e6
    # Lambert shading of the exaggerated geometry on top of the u_z colours (so the shape reads as a surface)
    nrm = np.cross(np.stack([Xs[1:, :-1] - Xs[:-1, :-1], Ys[1:, :-1] - Ys[:-1, :-1], Zs[1:, :-1] - Zs[:-1, :-1]], -1),
                   np.stack([Xs[:-1, 1:] - Xs[:-1, :-1], Ys[:-1, 1:] - Ys[:-1, :-1], Zs[:-1, 1:] - Zs[:-1, :-1]], -1))
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
    nrm *= np.sign(nrm[..., 2:3] + 1e-12)
    lam = 0.55 + 0.45 * np.clip(nrm @ np.array([-0.35, -0.45, 0.82]), 0, 1)
    fc = cmap(norm(uzc))
    fc[..., :3] *= lam[..., None]
    ax2.plot_surface(Xs, Ys, Zs, facecolors=fc, rstride=1, cstride=1, linewidth=0.15,
                     edgecolor=(0, 0, 0, 0.12), antialiased=False, shade=False)
    for L in taxel_lines(240):
        sel = (np.abs(L[:, 0] - cxw) <= hx + 1e-9) & (np.abs(L[:, 1] - cyw) <= hy + 1e-9)
        if sel.sum() < 2:
            continue
        L = L[sel]
        u = bilinear(U, vx, vy, L[:, 0], L[:, 1])
        ax2.plot((L[:, 1] + EXAG * u[:, 1]) * 1e3, (L[:, 0] + EXAG * u[:, 0]) * 1e3, -EXAG * u[:, 2] * 1e3 + 0.08,
                 color="k", lw=1.2, zorder=10)
    zlo = -EXAG * sc["uz_max_um"] * 1e-3 * 1.03
    zhi = max(0.5, -EXAG * sc["uz_min_um"] * 1e-3 * 1.3)
    ax2.set_xlim((cyw - hy) * 1e3, (cyw + hy) * 1e3)
    ax2.set_ylim((cxw - hx) * 1e3, (cxw + hx) * 1e3)
    ax2.set_zlim(zlo, zhi)
    ax2.set_box_aspect((2 * hy * 1e3, 2 * hx * 1e3, zhi - zlo), zoom=1.25)
    ax2.view_init(elev=30, azim=-72)
    for pane in (ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane):
        pane.set_alpha(0.0)
    ax2.grid(False)
    ax2.set_zticks([])
    ax2.zaxis.line.set_color((0, 0, 0, 0))
    ax2.set_xlabel("pad y (mm)", labelpad=6)
    ax2.set_ylabel("pad x (mm)", labelpad=6)
    ax2.set_title(f"(2) close-up of the free face ({2*hy*1e3:.0f} x {2*hx*1e3:.0f} mm window), displacement "
                  f"(all 3 components) EXAGGERATED x{EXAG:.0f}, colour = true u_z\n"
                  f"max u_z now {U[..., 2].max()*1e6:.0f} um (shown as {U[..., 2].max()*EXAG*1e3:.1f} mm)",
                  fontsize=12, y=1.0)
    cax = fig.add_axes([0.972, 0.42, 0.008, 0.45])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("u_z (um), + = into the layer")
    # (3) taxel map: horizontal = pad y, vertical = pad x (same orientation as panels 1 and 2)
    ax3 = fig.add_axes([0.035, 0.05, 0.30, 0.27])
    fz = tax[..., 2]                                     # (NYT, NXT)
    ax3.imshow(fz.T, origin="lower", extent=[-35.5, 35.5, -25.5, 25.5], cmap="viridis",
               norm=Normalize(0, max(1e-6, sc["taxel_Fz_max_N"])), aspect="equal")
    cy = (-sim.B / 2 + (np.arange(sim.NYT) + 0.5) * sim.DYT) * 1e3
    cx = (-sim.A / 2 + (np.arange(sim.NXT) + 0.5) * sim.DXT) * 1e3
    CY, CX = np.meshgrid(cy, cx, indexing="ij")
    qs = 9.0 / max(1e-6, sc["taxel_Ft_max_N"])           # mm per N
    msk = np.hypot(tax[..., 0], tax[..., 1]) > 2e-3
    ax3.quiver(CY[msk], CX[msk], tax[..., 1][msk] * qs, tax[..., 0][msk] * qs, color="r", angles="xy", scale_units="xy", scale=1,
               width=0.008)
    for jj in range(sim.NYT):
        for ii in range(sim.NXT):
            v = fz[jj, ii]
            if abs(v) > 5e-3:
                ax3.text(cy[jj], cx[ii] - 3.2, f"{v:.2f}", ha="center", va="center", fontsize=9,
                         color="w" if v < 0.6 * sc["taxel_Fz_max_N"] else "k")
    for x in np.arange(sim.NYT + 1) * sim.DYT * 1e3 - 35.5:
        ax3.axvline(x, color="0.6", lw=0.5)
    for y in np.arange(sim.NXT + 1) * sim.DXT * 1e3 - 25.5:
        ax3.axhline(y, color="0.6", lw=0.5)
    ax3.set_xlabel("pad y (mm)")
    ax3.set_ylabel("pad x (mm)")
    Fs = F[s]
    ax3.set_title(f"(3) taxel normal force (N, colour/number), shear on the layer (red, {1/qs:.2f} N per mm)\n"
                  f"sum of taxels: Fz = {tax[..., 2].sum():.2f} N, Ft = {np.hypot(*tax[..., :2].sum((0, 1))):.2f} N"
                  f"  |  contacts: {int(z['ncon'][s])}", fontsize=11)
    cax3 = fig.add_axes([0.335, 0.05, 0.006, 0.27])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, sc["taxel_Fz_max_N"]), cmap="viridis"), cax=cax3,
                 label="taxel F_n (N)")
    # (4) time strip
    ax4 = fig.add_axes([0.42, 0.105, 0.53, 0.195])
    Ft = np.hypot(F[:, 0], F[:, 1])
    ax4.plot(t * 1e3, F[:, 2], color="C0", lw=1.6, label="normal F_z (N)")
    ax4.plot(t * 1e3, Ft, color="C3", lw=1.4, label="tangential |F_t| (N)")
    ax4.axvline(t[s] * 1e3, color="k", lw=1.0)
    ax4.plot([t[s] * 1e3], [F[s, 2]], "o", color="C0")
    ax4.plot([t[s] * 1e3], [Ft[s]], "o", color="C3")
    ax4.set_xlim(0, t[-1] * 1e3)
    ax4.set_ylim(min(-0.05, F[:, 2].min() * 1.1), max(F[:, 2].max(), Ft.max()) * 1.15 + 0.05)
    ax4.set_xlabel("simulated time (ms)", labelpad=1)
    ax4.set_ylabel("total contact force on the layer (N)")
    ax4b = ax4.twinx()
    dep = np.where(z["depth"] > 0, z["depth"], 0.0) * 1e6
    ax4b.plot(t * 1e3, dep, color="0.45", lw=1.0, ls="--", label="object travel past first touch (um)")
    ax4b.plot(t * 1e3, z["uz_max"] * 1e6, color="0.2", lw=0.6, alpha=0.8, label="max free-face u_z (um, incl. contact penetration)")
    ax4b.set_ylabel("um")
    ax4b.set_ylim(0, max(1.0, dep.max(), z["uz_max"].max() * 1e6) * 1.15)
    bounds = np.r_[0, np.cumsum(np.bincount(z["phase"], minlength=6))]
    for p in range(6):
        a, b = t[min(bounds[p], len(t) - 1)] * 1e3, t[min(bounds[p + 1], len(t)) - 1] * 1e3
        ax4.axvspan(a, b, color=("0.93" if p % 2 else "white"), zorder=-5)
        ax4.text(0.5 * (a + b), ax4.get_ylim()[1] * 0.95, PH_NAMES[p], ha="center", va="top", fontsize=9,
                 color="0.35")
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=4, fontsize=9, frameon=False)
    ax4.set_title(f"(4) now: F_z = {Fs[2]:.2f} N, |F_t| = {Ft[s]:.2f} N, travel {dep[s]:.0f} um, "
                  f"max u_z {z['uz_max'][s]*1e6:.0f} um, contact penetration (force-weighted) "
                  f"{z['pen_fw'][s]*1e6:.0f} um", fontsize=11)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def title():
    os.makedirs(FR, exist_ok=True)
    objs = sim.objects()
    recs = [json.load(open(os.path.join(RUNS, f"seg{k}.json"))) for k in range(len(objs))]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=DPI)
    fig.text(0.05, 0.9, "PDMS tactile layer under contact: MuJoCo 3.17 flex close-up", fontsize=30, weight="bold")
    lines = [
        MAT + ".",
        f"Pad 51 x 71 x 1.5 mm, back face bonded (pinned), {recs[0]['cells'][0]}x{recs[0]['cells'][1]}x"
        f"{recs[0]['cells'][2]} cells (1.275 x 1.268 x 0.5 mm); F1-corrected inputs young = "
        f"{recs[0]['young_in_Pa']:.0f} Pa, poisson = {recs[0]['poisson_in']:.6f}.",
        "Density 965 kg/m^3. No viscoelasticity, no Rayleigh damping: the only dissipation is the discrete integrator's.",
        f"Contact: solref {recs[0]['solref'][0]:g} 1, solimp 0.9999, mu = 1 (elliptic cone, impratio 100), CG solver "
        f"{recs[0]['iterations']} iterations, timestep 0.5 ms.",
        "Each object (rigid, servo-driven, pseudo-random from seed 20261030): approach, press until the target load "
        f"or a {recs[0]['depth_cap_m']*1e6:.0f} um indentation cap, hold, slide 2-4 mm, hold, lift off.",
        f"Displacements in panel (2) are exaggerated x{EXAG:.0f}; panel (1) is true scale. Playback "
        f"{(sim.FRAME_EVERY*sim.H*30):.2f} s of simulated time per s of video (quasi-static slow motion).",
        "Curved/sharp objects reach < 1 N at the 100 um cap: 5-20 N would need several hundred um (tets inverted at 250 um).",
        "Objects (position = pad x, y of the centre at the start; slide direction and distance; target load):",
    ]
    for r in recs:
        o = r["object"]
        ang = np.degrees(np.arctan2(o["slide_dir"][1], o["slide_dir"][0]))
        lines.append(f"  {r['seg']+1}. {o['label']}: at ({o['xy0'][0]*1e3:.1f}, {o['xy0'][1]*1e3:.1f}) mm, slide "
                     f"{o['slide_m']*1e3:.1f} mm at {ang:.0f} deg, target {o['F_target_N']:.1f} N -> reached "
                     f"{r['Fz_max_N']:.2f} N ({'target load' if r['press_stop'] == 'force' else 'stopped by the indentation cap'})")
    for n, L in enumerate(lines):
        fig.text(0.05, 0.82 - n * 0.042, L, fontsize=15)
    fig.savefig(os.path.join(FR, "title.png"), dpi=DPI)
    plt.close(fig)


def frames(k, i0, i1):
    os.makedirs(FR, exist_ok=True)
    z, rec = load(k)
    sc = json.load(open(os.path.join(RUNS, "scales.json")))
    mjp = MjPanel(k, rec)
    n = len(z["f_step"])
    for i in range(i0, min(i1, n)):
        compose(k, i, z, rec, sc, mjp, os.path.join(FR, f"seg{k}_{i:04d}.png"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "title":
        title()
    elif cmd == "frames":
        frames(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
