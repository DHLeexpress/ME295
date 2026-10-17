#!/usr/bin/env python3
"""Draw the project hero figure -> docs/figures/hero.png (+ vector hero.svg).

Original illustration built from matplotlib primitives, so it is reproducible
from the repo:  python scripts/make_hero.py
Left: (B) a parallel-jaw gripper with thin tactile-skin fingers pulls one
book out of a packed shelf. Right: (sense) the shear field, (A) the friction-cone
margin controller, (C) an LLM agent that calls touch as a tool.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

ROOT = Path(__file__).resolve().parents[1]
W, H, DPI = 1600, 800, 100
PT = 72.0 / DPI                      # points per pixel at the drawing DPI
SANS = ["Lato", "DejaVu Sans"]
MONO = ["DejaVu Sans Mono", "DejaVu Sans"]

INK, SLATE, MUTED, LINE = "#0F172A", "#475569", "#94A3B8", "#E2E8F0"
TEAL, TEAL_L, TEAL_XL = "#0D9488", "#14B8A6", "#CCFBF1"
BLUE, BLUE_XL = "#1D4ED8", "#DBEAFE"
VIOLET, VIOLET_XL = "#6D28D9", "#EDE9FE"
ORANGE, ORANGE_XL = "#C2410C", "#FFEDD5"
AMBER, CORAL = "#F59E0B", "#EF4444"
PAGES = "#F4ECDC"


def shade(color, k):
    return tuple(np.clip(np.array(to_rgb(color)) * k, 0, 1))


def tint(color, k):
    c = np.array(to_rgb(color))
    return tuple(c + (1 - c) * k)


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")
        self.z = 0

    def _z(self):
        self.z += 1
        return self.z

    def _add(self, artist, clip=None):
        artist.set_zorder(self._z())
        if hasattr(artist, "get_path") and not isinstance(artist, Line2D):
            self.ax.add_patch(artist)
        else:
            self.ax.add_artist(artist)
        if clip is not None:
            artist.set_clip_path(clip)
        return artist

    # primitives (pixel units, y down) ------------------------------------
    def poly(self, pts, fc, ec="none", lw=0.0, alpha=1.0, clip=None, ls="-"):
        return self._add(Polygon(np.asarray(pts, float), closed=True, fc=fc, ec=ec, lw=lw * PT,
                                 alpha=alpha, joinstyle="round", ls=ls), clip)

    def rect(self, x, y, w, h, fc, ec="none", lw=0.0, r=0.0, alpha=1.0, clip=None, ls="-"):
        if r > 0:
            p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                               fc=fc, ec=ec, lw=lw * PT, alpha=alpha, ls=ls)
        else:
            p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw * PT, alpha=alpha, ls=ls)
        return self._add(p, clip)

    def circle(self, cx, cy, r, fc, ec="none", lw=0.0, alpha=1.0, clip=None, ls="-"):
        return self._add(Circle((cx, cy), r, fc=fc, ec=ec, lw=lw * PT, alpha=alpha, ls=ls), clip)

    def ellipse(self, cx, cy, w, h, fc, ec="none", lw=0.0, alpha=1.0, clip=None, ls="-"):
        return self._add(Ellipse((cx, cy), w, h, fc=fc, ec=ec, lw=lw * PT, alpha=alpha, ls=ls), clip)

    def line(self, xs, ys, color, lw, alpha=1.0, ls="-", clip=None, cap="round"):
        return self._add(Line2D(xs, ys, color=color, lw=lw * PT, alpha=alpha, ls=ls,
                                solid_capstyle=cap, dash_capstyle=cap), clip)

    def arrow(self, x0, y0, x1, y1, color, lw=2.0, head=10.0, alpha=1.0, clip=None, ls="-",
              rad=0.0, outline=None):
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=1,head_width=0.6",
                            mutation_scale=head * PT, color=color, lw=lw * PT, alpha=alpha, ls=ls,
                            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
        if outline:
            a.set_path_effects([pe.Stroke(linewidth=(lw + 2.5) * PT, foreground=outline), pe.Normal()])
        return self._add(a, clip)

    def text(self, x, y, s, size, color=INK, weight="normal", family=None, ha="left",
             va="baseline", style="normal", alpha=1.0, clip=None, rotation=0, **kw):
        t = self.ax.text(x, y, s, fontsize=size * PT, color=color, fontweight=weight,
                         family=family or SANS, ha=ha, va=va, fontstyle=style, alpha=alpha,
                         rotation=rotation, zorder=self._z(), **kw)
        if clip is not None:
            t.set_clip_path(clip)
        return t

    def gradient(self, x, y, w, h, colors, direction="v", clip=None, alpha=1.0):
        n = 256
        ramp = np.linspace(0, 1, n)
        if direction == "v":
            arr = np.repeat(ramp[:, None], 2, axis=1)
        elif direction == "h":
            arr = np.repeat(ramp[None, :], 2, axis=0)
        else:
            yy, xx = np.mgrid[0:1:n * 1j, 0:1:n * 1j]
            arr = 0.65 * xx + 0.35 * yy
        cmap = LinearSegmentedColormap.from_list("g", colors)
        im = self.ax.imshow(arr, cmap=cmap, extent=(x, x + w, y + h, y), origin="upper",
                            aspect="auto", interpolation="bicubic", zorder=self._z(), alpha=alpha)
        if clip is not None:
            im.set_clip_path(clip)
        return im

    def image(self, arr, x, y, w, h, cmap, clip=None, interpolation="nearest"):
        im = self.ax.imshow(arr, cmap=cmap, extent=(x, x + w, y + h, y), origin="upper",
                            aspect="auto", interpolation=interpolation, zorder=self._z(),
                            vmin=0, vmax=1)
        if clip is not None:
            im.set_clip_path(clip)
        return im

    def card(self, x, y, w, h, r=22):
        self.rect(x, y + 7, w, h, "#0F172A", r=r, alpha=0.035)
        self.rect(x, y + 3, w, h, "#0F172A", r=r, alpha=0.04)
        return self.rect(x, y, w, h, "white", ec=LINE, lw=1.2, r=r)

    def tag(self, x, y, label, fg, bg, width):
        self.rect(x, y, width, 24, bg, r=12)
        self.text(x + width / 2, y + 16.5, label, 12, color=fg, weight="bold", ha="center")

    def save(self, *paths):
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        plt.rcParams["svg.hashsalt"] = "shear-agent-hero"   # deterministic SVG ids
        for p in paths:
            if str(p).endswith(".png"):
                self.fig.savefig(p, dpi=150)
            else:
                self.fig.savefig(p, dpi=DPI, metadata={"Date": None})


# ------------------------------------------------------------------ scene (B)
BASE = 700                         # front edge of the shelf board
DX, DY = 0.42, -0.24               # oblique projection per unit depth
SHELF_D, BOOK_D, FINGER_D = 130, 110, 34


def P(x, y, d):
    return (x + DX * d, y + DY * d)


BOOKS = [  # x0, width, height, colour, style
    (72, 44, 300, "#4C6A92", 0), (116, 36, 270, "#D9A441", 1), (152, 58, 318, "#B5543C", 2),
    (210, 40, 258, "#7A9E7E", 0), (250, 50, 296, "#2F3E5C", 1),
    (380, 38, 282, "#7B4F7A", 0), (418, 62, 326, "#3F4448", 2), (480, 34, 262, "#C08497", 1),
    (514, 48, 304, "#8A8F4A", 0), (562, 54, 290, "#4F7C82", 2), (616, 40, 272, "#C9B48A", 1),
    (656, 46, 314, "#5B6C8F", 0), (702, 36, 284, "#7C8B6A", 2),
]
TARGET = (312, 56, 306, ORANGE)
LIFT = 84


def draw_book(c, clip, x0, w, h, color, style=0, lift=0.0, band=None):
    yb = BASE - lift
    yt = yb - h
    light = band or tint(color, 0.5)
    c.poly([(x0 + w, yt), P(x0 + w, yt, BOOK_D), P(x0 + w, yb, BOOK_D), (x0 + w, yb)],
           shade(color, 0.66), clip=clip)
    c.poly([(x0, yt), (x0 + w, yt), P(x0 + w, yt, BOOK_D), P(x0, yt, BOOK_D)], shade(color, 0.86), clip=clip)
    c.poly([P(x0 + 3, yt, 3), P(x0 + w - 3, yt, 3), P(x0 + w - 3, yt, BOOK_D - 2), P(x0 + 3, yt, BOOK_D - 2)],
           PAGES, clip=clip)
    c.rect(x0, yt, w, h, color, ec=shade(color, 0.6), lw=0.8, clip=clip)
    if style in (0, 2):
        c.rect(x0, yt + 0.08 * h, w, 3.5, light, clip=clip)
        c.rect(x0 + w / 2 - 2.2, yt + 0.22 * h, 4.4, 0.30 * h, light, r=2.2, alpha=0.85, clip=clip)
    if style == 2:
        c.rect(x0, yt + 0.08 * h + 7, w, 2, light, clip=clip)
    if style == 1:
        c.rect(x0 + 4, yt + 0.16 * h, w - 8, 0.15 * h, light, r=2, alpha=0.9, clip=clip)
    c.rect(x0, yb - 0.10 * h, w, 3.5, light, clip=clip)
    c.rect(x0 + 1, yt, 2.5, h, "white", alpha=0.10, clip=clip)
    c.rect(x0 + w - 4, yt, 4, h, "black", alpha=0.10, clip=clip)


def draw_finger(c, clip, x_plate, sensor_side):
    """Thin finger plate (8 px metal + 4 px tactile skin) hanging into a gap."""
    y0, y1 = 290, 474
    if sensor_side == "right":
        xb, xs = x_plate, x_plate + 8
    else:
        xs, xb = x_plate, x_plate + 4
    x_right = x_plate + 12
    c.poly([(x_right, y0), P(x_right, y0, FINGER_D), P(x_right, y1 - 4, FINGER_D), (x_right, y1 - 4)],
           "#64748B", clip=clip)
    c.rect(xb, y0, 8, y1 - y0, "#A3B1C2", ec="#475569", lw=0.8, r=2.5, clip=clip)
    c.rect(xb + 1.6, y0 + 2, 2, y1 - y0 - 6, "white", alpha=0.4, clip=clip)
    c.rect(xs, 332, 4, 134, TEAL_L, ec=TEAL, lw=0.6, r=1.5, clip=clip)


def draw_scene(c):
    x, y, w, h = 30, 30, 800, 740
    clip = c.card(x, y, w, h, r=26)
    c.gradient(x, y, w, h, ["#FCFAF6", "#F1EADF"], "v", clip=clip)

    # bookcase interior (back panel, lower compartment, inner side, board top)
    c.poly([P(72, 330, SHELF_D), P(738, 330, SHELF_D), P(738, 800, SHELF_D), P(72, 800, SHELF_D)],
           "#E3CFB2", clip=clip)
    c.poly([(72, 330), P(72, 330, SHELF_D), P(72, 800, SHELF_D), (72, 800)], "#C99A6B", clip=clip)
    c.poly([(72, 700), (738, 700), P(738, 700, SHELF_D), P(72, 700, SHELF_D)], "#D8AE80", clip=clip)

    # lower shelf: a few book tops peeking out under the board
    xl = 72
    for i, (bw, bh, col) in enumerate([(40, 0, "#5B6C8F"), (52, 6, "#B5543C"), (34, 2, "#7A9E7E"),
                                       (60, 8, "#2F3E5C"), (44, 0, "#D9A441"), (38, 5, "#7B4F7A"),
                                       (56, 1, "#4F7C82"), (46, 7, "#8A8F4A"), (40, 3, "#C08497"),
                                       (50, 0, "#3F4448"), (36, 6, "#C9B48A"), (48, 2, "#4C6A92"),
                                       (40, 4, "#A0522D"), (32, 0, "#7C8B6A")]):
        if xl + bw > 738:
            break
        top = 750 + bh
        c.poly([(xl, top), (xl + bw, top), P(xl + bw, top, BOOK_D), P(xl, top, BOOK_D)], PAGES, clip=clip)
        c.rect(xl, top, bw, 60, col, ec=shade(col, 0.6), lw=0.8, clip=clip)
        xl += bw
    c.rect(72, 726, 666, 16, "black", alpha=0.18, clip=clip)

    # top shelf books, left to right, with the two fingers in the gaps
    for x0, bw, bh, col, st in BOOKS[:5]:
        draw_book(c, clip, x0, bw, bh, col, st)
    tx, tw, th, tcol = TARGET
    c.rect(tx - 9, BASE - LIFT - th - 9, tw + 18, th + 18, "#FDBA74", r=12, alpha=0.35, clip=clip)
    draw_finger(c, clip, 300, "right")
    draw_book(c, clip, tx, tw, th, tcol, 0, lift=LIFT, band="#FCD34D")
    draw_finger(c, clip, 368, "left")
    for x0, bw, bh, col, st in BOOKS[5:]:
        draw_book(c, clip, x0, bw, bh, col, st)

    # board front and side panels
    c.rect(72, 700, 666, 26, "#B37B4E", clip=clip)
    c.rect(72, 700, 666, 3, "#D9A877", clip=clip)
    for px in (50, 738):
        c.rect(px, 330, 22, 470, "#B9855A", clip=clip)
        c.poly([(px, 330), (px + 22, 330), P(px + 22, 330, SHELF_D), P(px, 330, SHELF_D)], "#E2BC92", clip=clip)
    c.poly([(760, 330), P(760, 330, SHELF_D), P(760, 800, SHELF_D), (760, 800)], "#9C6B44", clip=clip)

    # gripper: carriages, housing, flange
    for cx in (282, 368):
        c.poly([(cx + 30, 254), P(cx + 30, 254, 30), P(cx + 30, 292, 30), (cx + 30, 292)], "#334155", clip=clip)
        c.rect(cx, 250, 30, 42, "#475569", r=5, clip=clip)
        c.circle(cx + 15, 271, 3.2, "#94A3B8", clip=clip)
    c.poly([(274, 170), (406, 170), P(406, 170, 60), P(274, 170, 60)], "#475569", clip=clip)
    c.poly([(412, 178), P(412, 178, 60), P(412, 248, 60), (412, 248)], "#1E293B", clip=clip)
    c.rect(268, 170, 144, 84, "#334155", r=10, clip=clip)
    c.rect(278, 234, 124, 8, "#1E293B", r=4, clip=clip)
    c.circle(396, 188, 4.5, TEAL_L, clip=clip)
    flange = c.rect(306, 96, 68, 76, "#CBD5E1", clip=clip)
    c.gradient(306, 96, 68, 76, ["#F1F5F9", "#CBD5E1", "#94A3B8"], "h", clip=flange)
    c.rect(300, 150, 80, 22, "#64748B", r=4, clip=clip)

    # arm (generic 6-DoF arm, cropped by the frame)
    for (x0, y0, x1, y1, wd) in [(880, -95, 540, 22, 74), (540, 22, 340, 70, 58)]:
        c.line([x0, x1], [y0, y1], "#94A3B8", wd + 6, clip=clip)
        c.line([x0, x1], [y0, y1], "#DDE4EB", wd, clip=clip)
        d = np.array([x1 - x0, y1 - y0], float)
        nrm = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        if nrm[1] > 0:
            nrm = -nrm
        off = nrm * wd * 0.22
        c.line([x0 + off[0], x1 + off[0]], [y0 + off[1], y1 + off[1]], "#F8FAFC", wd * 0.25, clip=clip)
    for jx, jy, r in [(540, 22, 46), (340, 70, 36)]:
        c.circle(jx, jy, r, "#E8EDF2", ec="#94A3B8", lw=2, clip=clip)
        c.circle(jx, jy, r * 0.78, "none", ec=TEAL_L, lw=3, clip=clip)
        c.circle(jx, jy, r * 0.62, "#CBD5E1", clip=clip)

    # pull arrow
    c.arrow(458, 248, 458, 150, ORANGE, lw=4.5, head=18, clip=clip)
    c.text(470, 206, "pull", 16, ORANGE, weight="bold", clip=clip)

    # magnifier on the right finger contact
    fx, fy, fr = 374, 442, 18
    mx, my, mr = 646, 222, 98
    s = 6.0

    def zx(v):
        return mx + s * (v - fx)

    def zy(v):
        return my + s * (v - fy)

    D = np.hypot(mx - fx, my - fy)
    th = np.arctan2(my - fy, mx - fx)
    alpha = np.arccos((fr - mr) / D)
    for sgn in (1, -1):
        n = np.array([np.cos(th + sgn * alpha), np.sin(th + sgn * alpha)])
        p1 = np.array([fx, fy]) + fr * n
        p2 = np.array([mx, my]) + mr * n
        c.line([p1[0], p2[0]], [p1[1], p2[1]], INK, 1.4, alpha=0.5, clip=clip)
    c.circle(fx, fy, fr, "none", ec="white", lw=5, clip=clip)
    c.circle(fx, fy, fr, "none", ec=INK, lw=2, clip=clip)
    c.circle(mx + 3, my + 6, mr + 4, "#0F172A", alpha=0.10, clip=clip)
    lens = c.circle(mx, my, mr, "white", clip=clip)
    c.rect(mx - mr, my - mr, zx(368) - (mx - mr), 2 * mr, ORANGE, clip=lens)
    c.gradient(mx - mr, my - mr, zx(368) - (mx - mr), 2 * mr, [shade(ORANGE, 1.05), shade(ORANGE, 0.82)],
               "h", clip=lens)
    c.rect(zx(368), my - mr, 24, 2 * mr, TEAL_L, clip=lens)
    rng = np.random.default_rng(7)
    for _ in range(260):
        px_, py_ = rng.uniform(zx(368) + 1, zx(372) - 1), rng.uniform(my - mr, my + mr)
        c.circle(px_, py_, rng.uniform(0.6, 1.9), "white" if rng.random() < 0.7 else "#0F766E",
                 alpha=rng.uniform(0.25, 0.75), clip=lens)
    c.rect(zx(372), my - mr, 48, 2 * mr, "#A3B1C2", clip=lens)
    c.rect(zx(372) + 7, my - mr, 9, 2 * mr, "white", alpha=0.35, clip=lens)
    c.circle(zx(376), my + 62, 7, "#64748B", clip=lens)
    c.rect(zx(380), my - mr, 120, 2 * mr, "#7B4F7A", clip=lens)
    yb9 = 418 + 0.08 * 282
    c.rect(zx(380), zy(yb9), 120, 3.5 * s, tint("#7B4F7A", 0.5), clip=lens)
    for yy in (my - 70, my - 22, my + 26):
        c.arrow(zx(370), yy, zx(370), yy + 36, "white", lw=2.6, head=10, clip=lens, outline=INK)
    c.text(mx - 60, my - 52, "book", 13, "white", weight="bold", ha="center", clip=lens)
    c.text(zx(376), my - 56, "finger", 11, INK, weight="bold", ha="center", rotation=90, va="center", clip=lens)
    c.circle(mx, my, mr, "none", ec="white", lw=7, clip=clip)
    c.circle(mx, my, mr, "none", ec=INK, lw=2.5, clip=clip)
    c.text(mx - 12, my + mr + 26, "the skin shears as the book drags", 14, SLATE, ha="center", clip=clip,
           path_effects=[pe.withStroke(linewidth=4 * PT, foreground="#F7F2EA")])
    c.text(zx(370), my + mr - 14, "skin", 11, "white", weight="bold", ha="center", clip=lens)

    # caption + leader to the left finger
    c.tag(60, 58, "B · APPLICATION", ORANGE, ORANGE_XL, 150)
    c.text(60, 112, "Pull one book", 24, INK, weight="bold", clip=clip)
    c.text(60, 140, "from a packed shelf", 24, INK, weight="bold", clip=clip)
    c.text(60, 178, "thin, shear-sensing fingers", 15, SLATE, clip=clip)
    c.text(60, 198, "slide into the gap", 15, SLATE, clip=clip)
    c.arrow(120, 212, 294, 412, INK, lw=1.4, head=8, alpha=0.6, clip=clip, ls=(0, (4, 3)), rad=0.3)
    c.card_outline = c.rect(x, y, w, h, "none", ec=LINE, lw=1.2, r=26)


# ------------------------------------------------------------------ right panel
def shear_field(nx=20, ny=15, c_ratio=0.6, seed=4):
    xs = (np.arange(nx) + 0.5) / nx * 2 - 1
    ys = (np.arange(ny) + 0.5) / ny * 2 - 1
    X, Y = np.meshgrid(xs, ys)
    r = np.hypot(X / 0.86, Y / 0.86)
    q = np.where(r < 1, np.sqrt(np.clip(1 - r ** 2, 0, None)), 0.0)
    stick = r < c_ratio
    q[stick] -= c_ratio * np.sqrt(1 - (r[stick] / c_ratio) ** 2)
    q += 0.05 * np.random.default_rng(seed).standard_normal(q.shape) * (r < 1)
    q = np.clip(q, 0, None)
    return q / q.max()


def draw_sense_card(c):
    x, y, w, h = 860, 176, 345, 300
    c.card(x, y, w, h, r=18)
    c.tag(x + 20, y + 18, "SENSE", TEAL, TEAL_XL, 74)
    c.text(x + 20, y + 68, "Feel the shear field", 19, INK, weight="bold")
    hx, hy, hw, hh = x + 24, y + 82, 200, 140
    frame = c.rect(hx, hy, hw, hh, "white", ec=LINE, lw=1, r=6)
    cmap = LinearSegmentedColormap.from_list("shear", ["#F8FAFC", "#CCFBF1", "#5EEAD4", "#14B8A6",
                                                       "#0F766E", "#134E4A"])
    q = shear_field()
    c.image(q, hx, hy, hw, hh, cmap, clip=frame)
    ny, nx = q.shape
    for i in range(1, ny, 2):
        for j in range(0, nx, 2):
            v = q[i, j]
            if v < 0.12:
                continue
            cx_, cy_ = hx + (j + 0.5) * hw / nx, hy + (i + 0.5) * hh / ny
            L = 5 + 13 * v
            c.arrow(cx_, cy_ - L / 2, cx_, cy_ + L / 2, "white" if v > 0.55 else INK, lw=1.3, head=5)
    c.ellipse(hx + hw / 2, hy + hh / 2, 2 * 0.6 * 0.86 * hw / 2, 2 * 0.6 * 0.86 * hh / 2, "none",
              ec=AMBER, lw=2, ls=(0, (4, 3)))
    bx = hx + hw + 14
    c.gradient(bx, hy, 10, hh, ["#134E4A", "#0F766E", "#14B8A6", "#5EEAD4", "#CCFBF1", "#F8FAFC"], "v")
    c.rect(bx, hy, 10, hh, "none", ec=LINE, lw=1)
    c.text(bx + 16, hy + 10, "high", 11, MUTED)
    c.text(bx + 16, hy + hh, "low", 11, MUTED)
    c.text(bx + 16, hy + hh / 2 - 4, "|shear|", 11, MUTED)
    c.line([hx, hx + 22], [hy + hh + 20, hy + hh + 20], AMBER, 2, ls=(0, (4, 3)))
    c.text(hx + 30, hy + hh + 24, "stick / slip front (slip starts at the edge)", 12, SLATE)
    c.text(hx, hy + hh + 46, r"strain-field estimation $\rightarrow$ $\epsilon_{xx},\ \epsilon_{yy},\ \epsilon_{xy}$",
           13, INK)
    c.text(hx, hy + hh + 65, "~50 ms per frame on the robot today", 12.5, MUTED)


def draw_control_card(c):
    x, y, w, h = 1220, 176, 350, 300
    c.card(x, y, w, h, r=18)
    c.tag(x + 20, y + 18, "A · CONTROL", BLUE, BLUE_XL, 112)
    c.text(x + 20, y + 68, "Squeeze just enough", 19, INK, weight="bold")
    ax_, ay_ = 1312, 420
    mu, hgt = 0.47, 140
    c.poly([(ax_, ay_), (ax_ - mu * hgt, ay_ - hgt), (ax_ + mu * hgt, ay_ - hgt)], TEAL_L, alpha=0.14)
    c.line([ax_, ax_ - mu * hgt], [ay_, ay_ - hgt], TEAL, 2)
    c.line([ax_, ax_ + mu * hgt], [ay_, ay_ - hgt], TEAL, 2)
    c.text(ax_ - mu * hgt - 2, ay_ - hgt - 8, "friction cone", 12, TEAL, weight="bold")
    c.line([ax_, ax_], [ay_, ay_ - hgt - 4], MUTED, 1.2, ls=(0, (3, 3)))
    fn, ft = 112, 0.317 * 112
    c.line([ax_, ax_ + ft], [ay_ - fn, ay_ - fn], MUTED, 1.2, ls=(0, (3, 3)))
    c.arrow(ax_, ay_, ax_ + ft, ay_ - fn, INK, lw=3, head=12)
    c.text(ax_ - 22, ay_ - 52, r"$f_n$", 15, SLATE)
    c.text(ax_ + ft / 2 - 6, ay_ - fn - 6, r"$f_t$", 15, SLATE)
    c.text(ax_ + ft + 6, ay_ - fn + 18, r"$\mathbf{f}$", 16, INK)
    c.circle(ax_, ay_, 3.5, INK)
    fx0 = 1398
    c.text(fx0, 300, r"$Q = 1-\frac{\|f_t\|}{\mu\, f_n}$", 21, INK)
    g0, g1, gy = 1398, 1552, 360
    lo, hi, qs, qv = -0.3, 1.0, 0.25, 0.32

    def gx(v):
        return g0 + (v - lo) / (hi - lo) * (g1 - g0)

    bar = c.rect(g0, gy, g1 - g0, 12, "#F1F5F9", r=6)
    c.rect(gx(lo), gy, gx(0) - gx(lo), 12, "#FCA5A5", clip=bar)
    c.rect(gx(0), gy, gx(qs) - gx(0), 12, "#FDE68A", clip=bar)
    c.rect(gx(qs), gy, gx(hi) - gx(qs), 12, "#99F6E4", clip=bar)
    c.poly([(gx(qv) - 6, gy - 10), (gx(qv) + 6, gy - 10), (gx(qv), gy - 1)], INK)
    c.text(gx(qv), gy - 15, "Q = 0.32", 12, INK, weight="bold", ha="center")
    c.text(gx(-0.15), gy + 28, "slip", 11.5, CORAL, ha="center", weight="bold")
    c.text(gx(qs), gy + 28, "Q*", 11.5, SLATE, ha="center", weight="bold")
    c.text(gx(0.7), gy + 28, "stable", 11.5, TEAL, ha="center", weight="bold")
    c.text(fx0, 424, r"$\min\ f_{\mathrm{grip}}\ \ \mathrm{s.t.}\ \ Q \geq Q^{*}$", 15, INK)
    c.text(x + 20, y + h - 30, "shear: tactile skin  ·  normal force: gripper", 12.5, MUTED)
    c.text(x + 20, y + h - 13, "μ: estimated online from slip onsets", 12.5, MUTED)


def draw_agent_card(c):
    x, y, w, h = 860, 494, 710, 276
    c.card(x, y, w, h, r=18)
    c.tag(x + 20, y + 18, "C · AGENT", VIOLET, VIOLET_XL, 96)
    c.text(x + 20, y + 66, "An LLM agent that uses touch as a tool", 19, INK, weight="bold")
    nx_, ny_ = 884, 584
    c.rect(nx_, ny_, 112, 112, VIOLET_XL, ec="#7C3AED", lw=2, r=18)
    pts = [(nx_ + 32, ny_ + 34), (nx_ + 56, ny_ + 24), (nx_ + 80, ny_ + 34)]
    c.line([p[0] for p in pts], [p[1] for p in pts], "#7C3AED", 2)
    for px_, py_ in pts:
        c.circle(px_, py_, 6, "white", ec="#7C3AED", lw=2)
    c.text(nx_ + 56, ny_ + 74, "LLM", 24, "#5B21B6", weight="bold", ha="center")
    c.text(nx_ + 56, ny_ + 96, "agent", 14, "#5B21B6", ha="center")
    c.text(nx_ + 56, ny_ + 136, "plans, monitors,", 12.5, SLATE, ha="center")
    c.text(nx_ + 56, ny_ + 154, "recovers", 12.5, SLATE, ha="center")
    c.arrow(1000, 620, 1026, 620, "#7C3AED", lw=2, head=9)
    c.arrow(1026, 660, 1000, 660, MUTED, lw=2, head=9)

    lx, ly, lw_, lh = 1030, 580, 384, 172
    c.rect(lx, ly, lw_, lh, INK, r=12)
    cw = 0.602 * 12.5                       # DejaVu Sans Mono advance at 12.5 px
    rows = [
        ("contact_state()", "L ✓  R ✓", "#FDE68A"),
        ("shear_margin()", "Q = 0.32, μ = 0.47", "#FDE68A"),
        ("execute_skill(\"pull\")", "pulling 2 cm/s", "#FDE68A"),
        ("incipient_slip()", "TRUE (edge slip)", "#FCA5A5"),
        ("set_margin(q_star=0.3)", "ok", "#FDE68A"),
        ("execute_skill(\"regrasp\")", "Q = 0.41 ✓", "#99F6E4"),
    ]
    for i, (call, result, col) in enumerate(rows):
        yy = ly + 30 + 25 * i
        c.text(lx + 14, yy, ">", 12.5, "#C4B5FD", family=MONO, weight="bold")
        c.text(lx + 14 + 2 * cw, yy, call, 12.5, "#5EEAD4", family=MONO)
        c.text(lx + 14 + 27 * cw, yy, "→", 12.5, MUTED, family=MONO)
        c.text(lx + 14 + 29 * cw, yy, result, 12.5, col, family=MONO)

    sx = 1434
    c.text(sx, 590, "skills", 12, MUTED, weight="bold")
    for i, name in enumerate(["tilt", "pinch", "pull"]):
        yy = 600 + 46 * i
        c.rect(sx, yy, 96, 30, ORANGE_XL, ec="#FDBA74", lw=1.2, r=15)
        c.text(sx + 48, yy + 20, name, 14, ORANGE, weight="bold", ha="center")
        if i < 2:
            c.arrow(sx + 48, yy + 31, sx + 48, yy + 45, MUTED, lw=1.6, head=7)
    c.arrow(sx + 98, 707, sx + 98, 661, CORAL, lw=1.8, head=8, rad=0.9, ls=(0, (3, 2)))
    c.text(sx + 48, 752, "recover by touch", 11.5, CORAL, ha="center", weight="bold")


def draw_title(c):
    c.text(860, 88, "Shear-Aware Agentic Grasping", 44, INK, weight="bold")
    c.text(860, 122, "A thin tactile skin lets a parallel-jaw gripper feel shear;", 17, SLATE)
    c.text(860, 146, "an LLM agent uses touch as a tool to grasp gently and pull books off a shelf.", 17, SLATE)
    c.text(1570, 40, "Dohyun & Jihoon  ·  AM/ME/CE 295, Caltech  ·  Fall 2026", 12, MUTED, ha="right")


def main() -> None:
    c = Canvas()
    c.gradient(0, 0, W, H, ["#F8FAFC", "#E8F0F7"], "d")
    draw_scene(c)
    draw_title(c)
    draw_sense_card(c)
    draw_control_card(c)
    draw_agent_card(c)
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    c.save(out / "hero.png", out / "hero.svg")
    print("wrote", (out / "hero.png").relative_to(ROOT), "and hero.svg")


if __name__ == "__main__":
    main()
