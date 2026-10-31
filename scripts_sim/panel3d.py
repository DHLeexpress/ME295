"""3D 'what the skin feels' panel in the ROBOT BASE FRAME: a drop-in for librarian_media.SensorPanel (560 x 1026).

Top: two matplotlib 3D views of the gripper, both in the UR10e base frame (= MuJoCo world, metres), centred on the
midpoint of the two PDMS pads (default +-60 mm), seen from the direction of the video's close-up camera:
  force view   each pad as its 5 x 7 cells at their true pose (the free face, z = -h in the PDMS frame), coloured by
               the normal force per cell (N); per-cell arrows = force of the book on that cell, R @ [shear_x,
               shear_y, normal] (N, fixed mm/N); the book cut to the view box (translucent wireframe); white arrow
               = book weight m g; yellow arrow = sum over both pads of the book's force (static hold: equals m g)
  deform view  the same cells coloured by compression (um); arrows = displacement of the free face R @ u (x gain)
Bottom: the four time series of SensorPanel (unchanged).

    panel = SensorPanel3D()
    rgb = panel.draw(rec, view=close_cam)   # rec: librarian_media.Recorder; view: mujoco.MjvCamera, (elev, azim), None

The view follows a MuJoCo free camera: matplotlib view_init(elev, azim) = (-elevation, azimuth + 180)
(mjv_to_mpl_view; verified against MuJoCo's own camera in test_panel3d.py).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from librarian_media import COLORS, SIDES, Log, lowpass  # noqa: E402

BG, AX_BG = "#101418", "#171c22"
INK, INK2, MUTED, ACCENT = "#e8e8e8", "#cfcfcf", "#8d96a0", "#f2c94c"
FORCE_ARROW, DEFORM_ARROW, SUM_ARROW, WEIGHT_ARROW = "#4fd1ff", "#ff9f43", "#f2c94c", "#f4f4f4"
TRIAD = {"x": "#ff6b6b", "y": "#7bd389", "z": "#6fa8ff"}
# fallback view (MuJoCo convention, deg): the close-up camera's at the start pose (tool along +y: 90 - 52 = 38)
FALLBACK_MJ_VIEW = (38.0, -30.0)
# matplotlib's normalised box side for box aspect (1, 1, 1), zoom 1 (Axes3D.set_box_aspect) and its eye distance
_MPL_BOX = 1.8294640721620434 * 25 / 24 / np.sqrt(3.0)
_MPL_DIST = 10.0


# ------------------------------------------------------------------ camera conversion
def mjv_to_mpl_view(azimuth: float, elevation: float) -> Tuple[float, float]:
    """MuJoCo free camera (azimuth, elevation in deg) -> matplotlib Axes3D.view_init(elev, azim), same view.

    MuJoCo looks along f = (cos el cos az, cos el sin az, sin el) from lookat - distance * f, up =
    (-sin el cos az, -sin el sin az, cos el). matplotlib puts the eye at centre + d (cos e cos a, cos e sin a, sin e),
    looking at the centre, z up. Same direction: (cos e cos a, cos e sin a, sin e) = -f -> e = -el, a = az + 180;
    the screen axes then agree too: right = (sin az, -cos az, 0), up as above (no roll in either).
    """
    elev = -float(elevation)
    azim = (float(azimuth) + 360.0) % 360.0 - 180.0          # az + 180, wrapped to [-180, 180)
    return elev, azim


def mpl_to_mjv_view(elev: float, azim: float) -> Tuple[float, float]:
    """Inverse of mjv_to_mpl_view: matplotlib (elev, azim) -> MuJoCo (azimuth, elevation)."""
    return (float(azim) + 360.0) % 360.0 - 180.0, -float(elev)


def view_axes(elev: float, azim: float):
    """matplotlib's screen axes for (elev, azim) in data space (equal box aspect): right u, up v, toward eye w."""
    e, a = np.radians(elev), np.radians(azim)
    w = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    u = np.array([-np.sin(a), np.cos(a), 0.0])
    v = np.cross(w, u)
    return u, v, w


def persp_focal_length(distance: float, half_size: float, zoom: float = 1.0) -> float:
    """matplotlib focal_length that puts the eye `distance` (m) from the box centre, as a MuJoCo camera at that
    distance from its lookat: matplotlib's eye sits _dist * f box-units away, the box side (2 half_size) being
    _MPL_BOX * zoom box-units."""
    return distance * _MPL_BOX * zoom / (_MPL_DIST * 2.0 * half_size)


# ------------------------------------------------------------------ geometry helpers
def clip_segment(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Liang-Barsky: the part of segment a-b inside the box [lo, hi], or None."""
    t0, t1 = 0.0, 1.0
    dv = b - a
    for k in range(3):
        if abs(dv[k]) < 1e-15:
            if a[k] < lo[k] or a[k] > hi[k]:
                return None
            continue
        ta, tb = (lo[k] - a[k]) / dv[k], (hi[k] - a[k]) / dv[k]
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return None
    return a + t0 * dv, a + t1 * dv


def clip_polygon(P: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: convex polygon (k, 3) clipped to the box [lo, hi]; (0, 3) if outside."""
    poly = list(P)
    for k in range(3):
        for bound, keep in ((lo[k], lambda x, b: x >= b), (hi[k], lambda x, b: x <= b)):
            if not poly:
                return np.zeros((0, 3))
            out = []
            for i, cur in enumerate(poly):
                prev = poly[i - 1]
                cin, pin = keep(cur[k], bound), keep(prev[k], bound)
                if cin != pin:
                    t = (bound - prev[k]) / (cur[k] - prev[k])
                    out.append(prev + t * (cur - prev))
                if cin:
                    out.append(cur)
            poly = out
    return np.array(poly) if poly else np.zeros((0, 3))


def arrow_segments(starts: np.ndarray, vecs: np.ndarray, w: np.ndarray, u: np.ndarray, head_frac: float = 0.3,
                   head_max: float = 0.005, angle_deg: float = 26.0) -> np.ndarray:
    """Shaft + two head strokes per arrow, heads opened in the screen plane (w = toward the eye): (3k, 2, 3)."""
    if len(starts) == 0:
        return np.zeros((0, 2, 3))
    L = np.linalg.norm(vecs, axis=1, keepdims=True)
    d = vecs / np.maximum(L, 1e-12)
    n = np.cross(w, d)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(nn > 1e-6, n / np.maximum(nn, 1e-12), u)          # arrow along the view axis: open along u
    hl = np.minimum(head_frac * L, head_max)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    tips = starts + vecs
    w1 = tips - hl * (c * d + s * n)
    w2 = tips - hl * (c * d - s * n)
    segs = np.stack([np.stack([starts, tips], 1), np.stack([tips, w1], 1), np.stack([tips, w2], 1)], 1)
    return segs.reshape(-1, 2, 3)


def cap_length(vecs: np.ndarray, cap: float):
    """Scale arrows longer than `cap` (m) down to `cap`; returns (vecs, number capped)."""
    L = np.linalg.norm(vecs, axis=-1, keepdims=True)
    k = np.minimum(1.0, cap / np.maximum(L, 1e-12))
    return vecs * k, int((L[..., 0] > cap).sum())


def book_geom(m, body: int) -> int:
    """The book's collision box: the geom named like the body, else its first box geom."""
    import mujoco
    name = m.body(body).name
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    if g >= 0:
        return g
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            return g
    raise ValueError(f"no box geom on body {name}")


_BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
_BOX_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def box_corners(c: np.ndarray, R: np.ndarray, half: np.ndarray) -> np.ndarray:
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return c + (s * half) @ R.T


# ------------------------------------------------------------------ pad state in the base frame
def pad_world(m, d, skin, fr) -> dict:
    """One pad in the base frame: cell centres on the free face, cell quads, outline, per-cell force of the book on
    the cell R @ [shear_x, shear_y, normal] (N) and the free face's displacement R @ u (m)."""
    body = skin.body
    R = d.xmat[body].reshape(3, 3)
    p = d.xpos[body]
    half = np.asarray(skin.half, float)
    xy = np.asarray(fr.taxel_xy, float)                                       # (ny, nx, 2)
    ny, nx = xy.shape[:2]
    face = p - half[2] * R[:, 2]                                              # free face: z = -h
    centres = face + xy[..., :1] * R[:, 0] + xy[..., 1:] * R[:, 1]            # (ny, nx, 3)
    hx = (2 * half[0] / nx) / 2
    hy = (2 * half[1] / ny) / 2
    ex, ey = hx * R[:, 0], hy * R[:, 1]
    quads = np.stack([centres - ex - ey, centres + ex - ey, centres + ex + ey, centres - ex + ey], -2)
    outline = face + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * half[:2] @ R[:, :2].T
    f_loc = np.concatenate([np.asarray(fr.shear, float), np.asarray(fr.normal, float)[..., None]], -1)
    return dict(R=R, p=p, half=half, face=face, centres=centres, quads=quads.reshape(-1, 4, 3), outline=outline,
                force=f_loc @ R.T, disp=np.asarray(fr.deformation, float) @ R.T, normal=np.asarray(fr.normal),
                comp=np.asarray(fr.deformation)[..., 2], fr=fr)


# ------------------------------------------------------------------ the panel
class SensorPanel3D:
    """'What the skin feels' in the robot base frame (two 3D views) + the four SensorPanel time series.

    force_max (N per cell) and deform_max (um) fix the colour scales; force_mm_per_n and deform_gain fix the arrow
    scales (constant over the video so frames compare); arrows longer than arrow_cap_mm are shortened to it (counted
    in the caption). half_size: minimum half side of the view box (m); it grows (10 mm steps) when the pads are
    further apart. proj: "ortho" (default; lengths read the same everywhere) or "persp" (eye at the camera's
    distance, as in the MuJoCo close-up)."""

    def __init__(self, width: int = 560, height: int = 1026, window: float = 12.0, q_star: float = 0.5,
                 force_max: float = 2.0, deform_max: float = 6.0, force_mm_per_n: float = 10.0,
                 deform_gain: float = 1000.0, half_size: float = 0.060, arrow_cap_mm: float = 38.0,
                 proj: str = "ortho", fallback_view: Tuple[float, float] = FALLBACK_MJ_VIEW, zoom: float = 1.0):
        self.W, self.H = width, height
        self.window, self.q_star = window, q_star
        self.force_mm_per_n, self.deform_gain = force_mm_per_n, deform_gain
        self.k_force = force_mm_per_n * 1e-3          # m of arrow per N
        self.half_min, self.cap = half_size, arrow_cap_mm * 1e-3
        self.proj, self.zoom = proj, zoom
        self.fallback = mjv_to_mpl_view(*fallback_view)
        self.last_view = None
        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BG)
        self.fig.text(0.5, 1 - 4 / height, "What the skin feels, in the robot base frame (5 x 7 cells, 20 Hz assumed)",
                      fontsize=10, ha="center", va="top", color=INK)
        self.norm_f = mcolors.Normalize(0.0, force_max)
        self.norm_d = mcolors.Normalize(0.0, deform_max)
        self.cmap_f, self.cmap_d = plt.get_cmap("magma"), plt.get_cmap("viridis")
        band, top0 = 322, 22
        self.views = {
            "force": self._make_view("force", top0, band, "Contact force per cell (contact solver)",
                                     self.cmap_f, self.norm_f, "normal force, N per cell", FORCE_ARROW),
            "deform": self._make_view("deform", top0 + band, band, "PDMS deformation per cell (elastic layer)",
                                      self.cmap_d, self.norm_d, "compression (um)", DEFORM_ARROW)}
        self._make_timeseries(top=top0 + 2 * band + 10)

    # -------------------------------------------------------------- layout
    def _rect(self, x, y, w, h):
        """Pixel box (x, y from the top-left) -> figure-fraction rect."""
        return [x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H]

    def _make_view(self, key, top, band, title, cmap, norm, cb_label, arrow_color):
        fig = self.fig
        fig.text(8 / self.W, 1 - (top + 4) / self.H, title, fontsize=8.2, color=ACCENT, ha="left", va="top")
        size = band - 8
        ax = fig.add_axes(self._rect(14, top + 8, size, size), projection="3d", computed_zorder=False)
        ax.set_facecolor(BG)
        ax.patch.set_alpha(0.0)
        ax.set_box_aspect((1, 1, 1), zoom=self.zoom)
        ax.set_proj_type("ortho")
        for axis, lab in ((ax.xaxis, "x (m)"), (ax.yaxis, "y (m)"), (ax.zaxis, "z (m)")):
            axis.set_pane_color(mcolors.to_rgba(AX_BG, 1.0))
            axis.pane.set_edgecolor("#3b434d")
            axis._axinfo["grid"].update(color=(0.55, 0.6, 0.66, 0.16), linewidth=0.5)
            axis.line.set_color("#56606b")
            axis.set_major_locator(MultipleLocator(0.04))
            axis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.set_label_text(lab, fontsize=6.5, color=INK2)
            axis.labelpad = -7
        ax.tick_params(axis="both", labelsize=5.6, colors=INK2, pad=-3)
        ax.disable_mouse_rotation()
        v = dict(ax=ax, key=key, capped=0)
        # draw order (no per-pixel depth in matplotlib): book faces < cells < pad outlines < book edges (x-ray,
        # dashed) < arrows < resultants < labels
        v["book_faces"] = Poly3DCollection([], facecolor=(0.72, 0.78, 0.88, 0.09),
                                           edgecolor=(0.86, 0.9, 0.95, 0.28), linewidths=0.6, zorder=1)
        v["cells"] = Poly3DCollection([], edgecolor=(0.55, 0.6, 0.66, 0.5), linewidths=0.35, zorder=3)
        v["outlines"] = Line3DCollection([], linewidths=1.6, zorder=4)
        v["book_edges"] = Line3DCollection([], colors=[(0.86, 0.9, 0.95, 0.8)], linewidths=0.8, linestyles=(0, (3, 2)),
                                           zorder=5)
        v["arrows"] = Line3DCollection([], colors=[arrow_color], linewidths=0.95, zorder=6)
        for k in ("book_faces", "cells", "outlines", "book_edges", "arrows"):
            ax.add_collection3d(v[k], autolim=False)
        v["book_lbl"] = ax.text(0, 0, 0, "book", fontsize=6, color=(0.82, 0.86, 0.92), zorder=9, ha="center",
                                va="center")
        if key == "force":
            v["weight"] = Line3DCollection([], colors=[WEIGHT_ARROW], linewidths=1.9, zorder=7)
            v["sum"] = Line3DCollection([], colors=[SUM_ARROW], linewidths=1.4, zorder=8)
            ax.add_collection3d(v["weight"], autolim=False)
            ax.add_collection3d(v["sum"], autolim=False)
            v["weight_lbl"] = ax.text(0, 0, 0, "m g", fontsize=6.2, color=WEIGHT_ARROW, zorder=9, ha="center",
                                      va="bottom")
            v["sum_lbl"] = ax.text(0, 0, 0, "Σ", fontsize=7, color=SUM_ARROW, zorder=9, ha="center", va="bottom")
        v["labels"] = {s: ax.text(0, 0, 0, s, fontsize=7, color="white", fontweight="bold", zorder=10,
                                  ha="center", va="bottom",
                                  bbox=dict(boxstyle="round,pad=0.18", fc=COLORS[s], ec="none", alpha=0.95))
                       for s in SIDES}
        # right column: readouts, triad, caption; colour bar at the far right
        x0 = 334
        v["read"] = []
        for i in range(5):
            y = top + 20 + 12 * i
            mk = fig.text(x0 / self.W, 1 - y / self.H, "", fontsize=6.8, color=INK, ha="left", va="top")
            tx = fig.text((x0 + 11) / self.W, 1 - y / self.H, "", fontsize=6.4, color=INK, ha="left", va="top",
                          family="DejaVu Sans Mono")
            v["read"].append((mk, tx))
        tax = fig.add_axes(self._rect(x0 + 52, top + 92, 84, 84))
        tax.set_xlim(-1.45, 1.45)
        tax.set_ylim(-1.45, 1.45)
        tax.set_aspect("equal")
        tax.axis("off")
        v["triad_title"] = tax.text(0, -1.45, "base frame", fontsize=5.8, color=MUTED, ha="center", va="bottom")
        v["triad"] = {c: (tax.plot([0, 0], [0, 0], color=TRIAD[c], lw=1.6, solid_capstyle="round")[0],
                          tax.text(0, 0, c, fontsize=7, color=TRIAD[c], ha="center", va="center", fontweight="bold"))
                      for c in "xyz"}
        tax.plot([0], [0], "o", color=INK2, ms=2)
        v["caption"] = fig.text(x0 / self.W, 1 - (top + 186) / self.H, "", fontsize=6.1, color=MUTED, ha="left",
                                va="top", linespacing=1.35)
        cax = fig.add_axes(self._rect(self.W - 40, top + 26, 8, band - 70))
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.locator = MultipleLocator(0.5 if key == "force" else 1.0)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=6, colors=INK2, length=2)
        cb.outline.set_edgecolor("#56606b")
        cb.set_label(cb_label, fontsize=6.4, color=INK2, labelpad=2)
        return v

    def _make_timeseries(self, top: int):
        fig = self.fig
        gs = fig.add_gridspec(4, 1, hspace=0.78, left=0.11, right=0.86, top=1 - (top + 16) / self.H, bottom=0.03)
        self.ax_n, self.ax_s, self.ax_q, self.ax_w = (fig.add_subplot(gs[i, 0]) for i in range(4))
        self.lines = {}
        for ax, title, ylab in [(self.ax_n, "normal force on each skin (sum of cells)", "N"),
                                (self.ax_s, "shear (friction) force on each skin", "N"),
                                (self.ax_q, "margin q (patch, assumed mu) & grip force register", "q"),
                                (self.ax_w, "arm F/T (book load) and book slip in the grasp", "N / mm")]:
            ax.set_facecolor(AX_BG)
            ax.tick_params(labelsize=6.5, colors=INK2)
            for sp in ax.spines.values():
                sp.set_color("#555")
            ax.grid(alpha=0.18)
            ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=3)
            ax.set_ylabel(ylab, fontsize=7.5, color=INK)
        for side in SIDES:
            self.lines[f"n_{side}"], = self.ax_n.plot([], [], color=COLORS[side], lw=1.5, label=side)
            self.lines[f"s_{side}"], = self.ax_s.plot([], [], color=COLORS[side], lw=1.5, label=side)
            self.lines[f"q_{side}"], = self.ax_q.plot([], [], color=COLORS[side], lw=1.5, label=f"q {side}")
        self.ax_q.axhline(self.q_star, color="#bbbbbb", lw=0.8, ls="--")
        self.ax_q.text(0.99, self.q_star, "q*", transform=self.ax_q.get_yaxis_transform(), color="#bbbbbb",
                       fontsize=7, ha="right", va="bottom")
        self.ax_f = self.ax_q.twinx()
        self.lines["force_reg"], = self.ax_f.plot([], [], color=ACCENT, lw=1.2, ls="-", label="force register")
        self.ax_f.set_ylim(-5, 260)
        self.ax_f.tick_params(labelsize=6.5, colors=ACCENT)
        self.lines["fz"], = self.ax_w.plot([], [], color="#9be7a1", lw=1.3, label="-Fz (N, 3 Hz low-pass)")
        self.lines["slip"], = self.ax_w.plot([], [], color="#ffb86c", lw=1.3, label="slip (mm)")
        self.lines["turn"], = self.ax_w.plot([], [], color="#c39bd3", lw=1.2, label="rotation (deg)")
        for ax in (self.ax_n, self.ax_s, self.ax_w):
            ax.legend(fontsize=6.5, loc="upper left", facecolor=BG, labelcolor=INK, framealpha=0.6, ncol=3)
        self.ax_q.legend(handles=[self.lines["q_left"], self.lines["q_right"], self.lines["force_reg"]], fontsize=6.5,
                         loc="lower left", facecolor=BG, labelcolor=INK, framealpha=0.6, ncol=3)
        self.ax_q.set_ylim(-0.1, 1.05)

    # -------------------------------------------------------------- per frame
    def resolve_view(self, view) -> Tuple[float, float, Optional[float], str]:
        """(elev, azim, camera distance or None, description) for view = a mujoco.MjvCamera (anything with
        azimuth / elevation, MuJoCo convention), a matplotlib (elev, azim) tuple, or None (fixed fallback)."""
        if view is None:
            az, el = mpl_to_mjv_view(*self.fallback)
            return (*self.fallback, None, f"fixed view (az {az:.0f}, el {el:.0f} deg)")
        if hasattr(view, "azimuth") and hasattr(view, "elevation"):
            elev, azim = mjv_to_mpl_view(view.azimuth, view.elevation)
            return elev, azim, float(getattr(view, "distance", 0.0)) or None, \
                f"close-up camera (az {view.azimuth:.0f}, el {view.elevation:.0f} deg)"
        elev, azim = (float(x) for x in view)
        return elev, azim, None, f"view elev {elev:.0f}, azim {azim:.0f} deg (matplotlib)"

    def scene(self, rec) -> dict:
        """Everything the 3D views draw, in the base frame (the tests use it for the physics checks)."""
        m, d = rec.m, rec.d
        pads = {}
        for side in SIDES:
            fr = rec.frames.get(side) if rec.frames else None
            k = rec.skins[side]
            skin = getattr(k, "skin", k)                 # SkinSensor wraps the PDMSSkin
            if fr is None:
                fr = k.read(d)
            pads[side] = pad_world(m, d, skin, fr)
        centre = 0.5 * (pads["left"]["face"] + pads["right"]["face"])
        corners = np.concatenate([pads[s]["outline"] for s in SIDES])
        need = np.abs(corners - centre).max() + 0.015
        half = max(self.half_min, float(np.ceil(need / 0.01) * 0.01))
        g = book_geom(m, rec.book)
        book = dict(c=d.geom_xpos[g].copy(), R=d.geom_xmat[g].reshape(3, 3).copy(), half=m.geom_size[g].copy(),
                    mass=float(m.body_mass[rec.book]), g=np.array(m.opt.gravity))   # body_mass: make_sim may edit it
        force = {s: pads[s]["force"].reshape(-1, 3).sum(0) for s in SIDES}                # book on each pad
        normal = {s: pads[s]["fr"].total_normal * pads[s]["R"][:, 2] for s in SIDES}       # its normal part
        shear = {s: force[s] - normal[s] for s in SIDES}                                   # its in-plane part
        return dict(pads=pads, centre=centre, half=half, book=book, force=force, normal=normal, shear=shear,
                    force_sum=force["left"] + force["right"], shear_sum=shear["left"] + shear["right"],
                    weight=book["mass"] * book["g"])

    def draw(self, rec, view=None) -> np.ndarray:
        sc = self.scene(rec)
        elev, azim, dist, desc = self.resolve_view(view)
        self.last_view = (elev, azim)
        u, v_up, w = view_axes(elev, azim)
        c, L = sc["centre"], sc["half"]
        lo, hi = c - L, c + L
        for key, v in self.views.items():
            ax = v["ax"]
            ax.set_xlim3d(lo[0], hi[0])
            ax.set_ylim3d(lo[1], hi[1])
            ax.set_zlim3d(lo[2], hi[2])
            ax.view_init(elev=elev, azim=azim)
            if self.proj == "persp" and dist:
                ax.set_proj_type("persp", focal_length=persp_focal_length(dist, L, self.zoom))
            else:
                ax.set_proj_type("ortho")
            self._draw_book(v, sc["book"], lo, hi)
            self._draw_pads(v, key, sc, w, u)
            self._draw_triad(v, u, v_up)
            for side in SIDES:
                pd = sc["pads"][side]
                R = pd["R"]
                top_z = np.abs(R[2, :2]) @ np.asarray(pd["half"][:2])            # pad's half height in world z
                pos = pd["face"] + 0.008 * R[:, 2] + np.array([0.0, 0.0, top_z + 0.006])
                v["labels"][side].set_position_3d(np.clip(pos, lo, hi))
            if key == "force":
                self._draw_resultants(v, sc, w, u, lo, hi)
        self._readouts(sc, desc)
        self._timeseries(rec.log)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def _draw_book(self, v, book, lo, hi):
        P = box_corners(book["c"], book["R"], book["half"])
        edges = [s for a, b in _BOX_EDGES if (s := clip_segment(P[a], P[b], lo, hi)) is not None]
        v["book_edges"].set_segments(edges)
        faces = [f for idx in _BOX_FACES if len(f := clip_polygon(P[list(idx)], lo, hi)) >= 3]
        v["book_faces"].set_verts(faces)
        if faces:                      # label the visible part: centre of the clipped book, near its top
            pts = np.concatenate(faces)
            mid = 0.5 * (pts.min(0) + pts.max(0))
            v["book_lbl"].set_position_3d((mid[0], mid[1], pts[:, 2].max() - 0.006))
            v["book_lbl"].set_text("book")
        else:
            v["book_lbl"].set_text("")

    def _draw_pads(self, v, key, sc, w, u):
        quads, cols, starts, vecs, capped = [], [], [], [], 0
        for side in SIDES:
            pd = sc["pads"][side]
            quads.append(pd["quads"])
            if key == "force":
                cols.append(self.cmap_f(self.norm_f(pd["normal"].ravel())))
                vec = pd["force"].reshape(-1, 3) * self.k_force
                on = pd["normal"].ravel() > 0.01
            else:
                cols.append(self.cmap_d(self.norm_d(1e6 * pd["comp"].ravel())))
                vec = pd["disp"].reshape(-1, 3) * self.deform_gain
                on = np.linalg.norm(pd["disp"].reshape(-1, 3), axis=1) > 0.3e-6
            vec, n = cap_length(vec[on], self.cap)
            capped += n
            starts.append(pd["centres"].reshape(-1, 3)[on])
            vecs.append(vec)
        v["cells"].set_verts(list(np.concatenate(quads)))
        v["cells"].set_facecolor(np.concatenate(cols))
        v["outlines"].set_segments([sc["pads"][s]["outline"] for s in SIDES])
        v["outlines"].set_color([COLORS[s] for s in SIDES])
        v["arrows"].set_segments(arrow_segments(np.concatenate(starts), np.concatenate(vecs), w, u, head_max=0.0035))
        v["capped"] = capped

    def _draw_resultants(self, v, sc, w, u, lo, hi):
        """Book weight m g (white, centred on the pads' mid-height) and the sum of the book's force on both pads
        (yellow, from the same start height), same mm/N, side by side to the screen-right of both pads (u: the
        camera's horizontal right vector), so they never cover a cell. At a static hold the two heads coincide."""
        pts = np.concatenate([sc["pads"][s]["outline"] for s in SIDES])
        c = sc["centre"]
        right = float(((pts - c) @ u).max())
        W, _ = cap_length(sc["weight"][None] * self.k_force, 1.6 * self.half_min)
        F, _ = cap_length(sc["force_sum"][None] * self.k_force, 1.6 * self.half_min)
        top = np.array([0.0, 0.0, 0.5 * np.linalg.norm(W[0])])
        w0 = c + (right + 0.010) * u + top
        f0 = c + (right + 0.019) * u + top
        w0, f0 = np.clip(w0, lo, hi), np.clip(f0, lo, hi)
        v["weight"].set_segments(arrow_segments(w0[None], W, w, u, head_max=0.005))
        v["sum"].set_segments(arrow_segments(f0[None], F, w, u, head_max=0.005) if np.linalg.norm(F) > 1e-4 else [])
        up = np.array([0.0, 0.0, 0.003])
        v["weight_lbl"].set_position_3d(np.clip(w0 + up, lo, hi))
        v["sum_lbl"].set_position_3d(np.clip(f0 + up, lo, hi))
        v["sum_lbl"].set_text("\u03a3" if np.linalg.norm(F) > 1e-4 else "")

    def _draw_triad(self, v, u, v_up):
        for k, cname in enumerate("xyz"):
            e = np.zeros(3)
            e[k] = 1.0
            p = np.array([e @ u, e @ v_up])                      # screen projection of the base axis
            line, lbl = v["triad"][cname]
            line.set_data([0, p[0]], [0, p[1]])
            n = np.linalg.norm(p)
            lbl.set_position(p / n * (n + 0.3) if n > 1e-6 else (0.0, 0.0))

    def _readouts(self, sc, desc):
        f, dv = self.views["force"], self.views["deform"]

        def put(view, i, marker, color, text):
            mk, tx = view["read"][i]
            mk.set_text(marker)
            mk.set_color(color)
            tx.set_text(text)

        put(f, 0, "", INK, "        N (N)  |T| (N)     q")
        put(dv, 0, "", INK, "        compr (um)  shear (um)")
        for i, side in enumerate(SIDES, start=1):
            fr = sc["pads"][side]["fr"]
            put(f, i, "\u25a0", COLORS[side], f"{side:6s} {fr.total_normal:5.1f}   {np.linalg.norm(fr.total_shear):5.2f}"
                f"  {fr.net_margin():+5.2f}")
            u = 1e6 * fr.deformation
            put(dv, i, "\u25a0", COLORS[side], f"{side:6s} {u[..., 2].max():6.1f}  "
                f"{np.linalg.norm(u[..., :2], axis=-1).max():9.1f}   (max)")
        F, Wt = sc["force_sum"], sc["weight"]
        put(f, 3, "\u2193", SUM_ARROW, f"\u03a3 pads ({F[0]:+.2f}, {F[1]:+.2f}, {F[2]:+.2f}) N")
        put(f, 4, "\u2193", WEIGHT_ARROW, f"m g    ({Wt[0]:+.2f}, {Wt[1]:+.2f}, {Wt[2]:+.2f}) N")
        put(dv, 3, "", INK, "")
        put(dv, 4, "", INK, "")
        for view in (f, dv):
            view["triad_title"].set_text("base frame")
        cap_f = f"\n{f['capped']} arrows capped at {self.cap * 1e3:.0f} mm" if f["capped"] else ""
        cap_d = f"\n{dv['capped']} arrows capped at {self.cap * 1e3:.0f} mm" if dv["capped"] else ""
        f["caption"].set_text(f"view: {desc}\ncyan: force of the book on each cell\n(normal + shear), "
                              f"{self.force_mm_per_n:g} mm per N\nΣ (yellow): sum over both pads;\nm g (white): "
                              f"book weight, same scale{cap_f}")
        dv["caption"].set_text(f"view: {desc}\norange: displacement of the free face\nper cell, x{self.deform_gain:g}"
                               f" (1 um = {self.deform_gain * 1e-3:g} mm)\ncolour: compression (into the layer)"
                               f"{cap_d}")

    def _timeseries(self, log: Log):
        rows = log.rows if log is not None else []
        if not rows:
            return
        t = log.column("t")
        t1 = t[-1]
        t0 = max(0.0, t1 - self.window)
        sel = t >= t0
        ts = t[sel]
        for side in SIDES:
            self.lines[f"n_{side}"].set_data(ts, log.column(f"{side}_normal")[sel])
            self.lines[f"s_{side}"].set_data(ts, log.column(f"{side}_shear")[sel])
            self.lines[f"q_{side}"].set_data(ts, np.clip(log.column(f"{side}_margin")[sel], -0.1, 1.05))
        self.lines["force_reg"].set_data(ts, log.column("gripper_force")[sel])
        self.lines["fz"].set_data(ts, -lowpass(log.column("fz"), t, 3.0)[sel])
        self.lines["slip"].set_data(ts, log.column("slip_mm")[sel])
        self.lines["turn"].set_data(ts, log.column("slip_deg")[sel])
        for ax in (self.ax_n, self.ax_s, self.ax_q, self.ax_w):
            ax.set_xlim(t0, max(t0 + self.window, t1))
        nmax = max(5.0, np.max([log.column(f"{s}_normal")[sel].max() for s in SIDES]) * 1.15)
        smax = max(1.0, np.max([log.column(f"{s}_shear")[sel].max() for s in SIDES]) * 1.25)
        wmax = max(2.0, np.max(np.abs(np.r_[log.column("fz")[sel], log.column("slip_mm")[sel],
                                            log.column("slip_deg")[sel]])) * 1.2)
        self.ax_n.set_ylim(0, nmax)
        self.ax_s.set_ylim(0, smax)
        self.ax_w.set_ylim(-wmax * 0.3, wmax)
