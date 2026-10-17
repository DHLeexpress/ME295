#!/usr/bin/env python3
"""Draw the project hero figure -> docs/figures/hero.png (+ vector hero.svg).

Original illustration built from matplotlib primitives, so it is reproducible
from the repo:  python scripts/make_hero.py
Left (B): a parallel-jaw gripper approaches a packed shelf horizontally, slides
its thin tactile-skin fingers into the gaps and pulls one book out sideways
(there is no headroom to lift it). Right: (sense) the shear field, (A) the
friction-cone margin controller, (C) an LLM agent that calls touch as a tool.
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

    def tag(self, x, y, label, fg, bg, width, clip=None):
        self.rect(x, y, width, 24, bg, r=12, clip=clip)
        self.text(x + width / 2, y + 16.5, label, 12, color=fg, weight="bold", ha="center", clip=clip)

    def pill(self, cx, cy, label, fg, width, size=12.5, clip=None, bg="white"):
        self.rect(cx - width / 2, cy - 12, width, 24, bg, ec=LINE, lw=1, r=12, clip=clip)
        self.text(cx, cy + 4.5, label, size, color=fg, weight="bold", ha="center", clip=clip)

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
BASE = 560                  # front edge of the shelf board the books stand on
TOP_Y = 238                 # underside of the shelf above: almost no headroom
DX, DY = 0.48, -0.27        # oblique projection per unit depth (into the shelf = up-right)
SHELF_D, BOOK_D = 130, 118
PULL = 100                  # how far the target book is already pulled out
FY0, FY1 = 356, 418         # finger height band
FD0, FD1 = -112, 28         # finger depth extent: jaw -> tip inside the gap

LEFT_BOOKS = [  # x0, width, height, colour, style
    (132, 44, 290, "#4C6A92", 0), (176, 36, 266, "#D9A441", 1), (212, 58, 306, "#B5543C", 2),
    (270, 40, 262, "#7A9E7E", 0), (310, 50, 296, "#2F3E5C", 1), (360, 38, 280, "#4F7C82", 2),
    (398, 60, 300, "#3F4448", 0),
]
TARGET = (470, 56, 296, ORANGE)
RIGHT_BOOKS = [
    (538, 38, 282, "#7B4F7A", 0), (576, 62, 312, "#5B6C8F", 2), (638, 34, 266, "#C08497", 1),
    (672, 48, 300, "#8A8F4A", 0), (720, 54, 288, "#4F7C82", 2), (774, 40, 274, "#C9B48A", 1),
    (814, 46, 306, "#7C8B6A", 0),
]


def P(x, y, d):
    return (x + DX * d, y + DY * d)


def draw_book(c, clip, x0, w, h, color, style=0, d0=0.0, band=None):
    """Upright book whose spine (front face) sits at depth d0 (negative = pulled out)."""
    yb, yt = BASE, BASE - h
    light = band or tint(color, 0.5)

    def Q(x, y, d):
        return P(x, y, d0 + d)

    c.poly([Q(x0 + w, yt, 0), Q(x0 + w, yt, BOOK_D), Q(x0 + w, yb, BOOK_D), Q(x0 + w, yb, 0)],
           shade(color, 0.66), clip=clip)
    c.poly([Q(x0, yt, 0), Q(x0 + w, yt, 0), Q(x0 + w, yt, BOOK_D), Q(x0, yt, BOOK_D)], shade(color, 0.86),
           clip=clip)
    c.poly([Q(x0 + 3, yt, 3), Q(x0 + w - 3, yt, 3), Q(x0 + w - 3, yt, BOOK_D - 2), Q(x0 + 3, yt, BOOK_D - 2)],
           PAGES, clip=clip)
    fx, fy = Q(x0, yt, 0)
    c.rect(fx, fy, w, h, color, ec=shade(color, 0.6), lw=0.8, clip=clip)
    if style in (0, 2):
        c.rect(fx, fy + 0.08 * h, w, 3.5, light, clip=clip)
        c.rect(fx + w / 2 - 2.2, fy + 0.22 * h, 4.4, 0.30 * h, light, r=2.2, alpha=0.85, clip=clip)
    if style == 2:
        c.rect(fx, fy + 0.08 * h + 7, w, 2, light, clip=clip)
    if style == 1:
        c.rect(fx + 4, fy + 0.16 * h, w - 8, 0.15 * h, light, r=2, alpha=0.9, clip=clip)
    c.rect(fx, fy + h - 0.10 * h, w, 3.5, light, clip=clip)
    c.rect(fx + 1, fy, 2.5, h, "white", alpha=0.10, clip=clip)
    c.rect(fx + w - 4, fy, 4, h, "black", alpha=0.10, clip=clip)


def draw_box(c, clip, x0, x1, y0, y1, d_near, d_far, front, top, side, r=0.0):
    """Box seen from front-right-above: top face, right face, then the face toward the viewer."""
    c.poly([P(x0, y0, d_near), P(x1, y0, d_near), P(x1, y0, d_far), P(x0, y0, d_far)], top, clip=clip)
    c.poly([P(x1, y0, d_near), P(x1, y0, d_far), P(x1, y1, d_far), P(x1, y1, d_near)], side, clip=clip)
    fx, fy = P(x0, y0, d_near)
    return c.rect(fx, fy, x1 - x0, y1 - y0, front, r=r, clip=clip)


def draw_right_finger(c, clip):
    """Thin finger in the gap right of the target: 6 px skin (inner side) + 6 px metal."""
    xs0, xs1, xm1 = 526, 532, 538
    face = [P(xm1, FY0, FD0), P(xm1, FY0, FD1), P(xm1, FY1, FD1), P(xm1, FY1, FD0)]
    c.poly(face, "none", ec="white", lw=4, clip=clip)
    c.poly(face, "#B8C4D2", ec="#475569", lw=1, clip=clip)
    c.poly([P(xm1, FY0 + 7, FD0), P(xm1, FY0 + 7, FD1), P(xm1, FY0 + 14, FD1), P(xm1, FY0 + 14, FD0)],
           "white", alpha=0.45, clip=clip)
    c.poly([P(xs1, FY0, FD0), P(xm1, FY0, FD0), P(xm1, FY0, FD1), P(xs1, FY0, FD1)], "#E2E8F0", clip=clip)
    c.poly([P(xs0, FY0, FD0), P(xs1, FY0, FD0), P(xs1, FY0, FD1), P(xs0, FY0, FD1)], TEAL_L,
           ec=TEAL, lw=0.6, clip=clip)


def draw_arm_link(c, clip, x0, y0, x1, y1, wd):
    c.line([x0, x1], [y0, y1], "#94A3B8", wd + 6, clip=clip)
    c.line([x0, x1], [y0, y1], "#DDE4EB", wd, clip=clip)
    d = np.array([x1 - x0, y1 - y0], float)
    nrm = np.array([d[1], -d[0]]) / np.linalg.norm(d)
    if nrm[1] > 0:
        nrm = -nrm
    off = nrm * wd * 0.22
    c.line([x0 + off[0], x1 + off[0]], [y0 + off[1], y1 + off[1]], "#F8FAFC", wd * 0.25, clip=clip)


def draw_joint(c, clip, jx, jy, r):
    c.circle(jx, jy, r, "#E8EDF2", ec="#94A3B8", lw=2, clip=clip)
    c.circle(jx, jy, r * 0.78, "none", ec=TEAL_L, lw=3, clip=clip)
    c.circle(jx, jy, r * 0.62, "#CBD5E1", clip=clip)


def draw_magnifier(c, clip, fx, fy, fr, mx, my, mr):
    """Top view inside the right gap: book | skin | finger | next book."""
    D = np.hypot(mx - fx, my - fy)
    th = np.arctan2(my - fy, mx - fx)
    alpha = np.arccos((fr - mr) / D)
    for sgn in (1, -1):
        n = np.array([np.cos(th + sgn * alpha), np.sin(th + sgn * alpha)])
        p1 = np.array([fx, fy]) + fr * n
        p2 = np.array([mx, my]) + mr * n
        c.line([p1[0], p2[0]], [p1[1], p2[1]], INK, 1.4, alpha=0.55, clip=clip)
    c.circle(fx, fy, fr, "none", ec="white", lw=5, clip=clip)
    c.circle(fx, fy, fr, "none", ec=INK, lw=2, clip=clip)
    c.circle(mx + 3, my + 6, mr + 4, "#0F172A", alpha=0.12, clip=clip)
    lens = c.circle(mx, my, mr, "white", clip=clip)

    top, bot = my - mr, my + mr
    y_spine = my + 18                      # front edge of the neighbouring book
    y_tip = y_spine - 46                   # finger tip, already inside the gap
    x_tc, x_sk, x_mt, x_nc, x_np = mx - 30, mx - 24, mx - 6, mx + 30, mx + 36

    c.rect(mx - mr, top, x_tc - (mx - mr), 2 * mr, PAGES, clip=lens)          # target pages
    for xx in np.arange(mx - mr + 3, x_tc, 4):
        c.line([xx, xx], [top, bot], "#E5D9C3", 0.8, clip=lens)
    c.rect(x_tc, top, x_sk - x_tc, 2 * mr, ORANGE, clip=lens)                  # target cover
    c.rect(x_sk, top, x_nc - x_sk, y_tip - top, "#334155", clip=lens)          # empty slot ahead of the tip
    c.rect(x_nc, top, mx + mr - x_nc, y_spine - top, PAGES, clip=lens)         # next book
    for xx in np.arange(x_np + 3, mx + mr, 4):
        c.line([xx, xx], [top, y_spine], "#E5D9C3", 0.8, clip=lens)
    c.rect(x_nc, top, x_np - x_nc, y_spine - top, "#7B4F7A", clip=lens)        # its cover
    c.rect(x_nc, y_spine, mx + mr - x_nc, bot - y_spine, "#EEF2F6", clip=lens)  # outside the shelf
    c.line([x_nc, mx + mr], [y_spine, y_spine], "#64748B", 1.4, ls=(0, (4, 3)), clip=lens)

    c.rect(x_sk, y_tip, x_mt - x_sk, bot - y_tip, TEAL_L, clip=lens)           # skin
    rng = np.random.default_rng(7)
    for _ in range(220):
        px_, py_ = rng.uniform(x_sk + 1, x_mt - 1), rng.uniform(y_tip + 2, bot)
        c.circle(px_, py_, rng.uniform(0.6, 1.8), "white" if rng.random() < 0.7 else "#0F766E",
                 alpha=rng.uniform(0.25, 0.75), clip=lens)
    c.rect(x_mt, y_tip, x_nc - x_mt, bot - y_tip, "#A3B1C2", clip=lens)        # finger metal
    c.rect(x_mt + 8, y_tip, 8, bot - y_tip, "white", alpha=0.35, clip=lens)
    c.circle(x_sk + (x_nc - x_sk) / 2, y_tip, (x_nc - x_sk) / 2, "#A3B1C2", clip=lens)

    for y0 in (my + 6, my + 42):                                               # shear on the skin
        c.arrow((x_sk + x_mt) / 2, y0, (x_sk + x_mt) / 2, y0 - 28, "white", lw=2.4, head=9,
                clip=lens, outline=INK)
    c.arrow(x_mt + 24, my - 14, x_mt + 24, my + 62, ORANGE, lw=3, head=11, clip=lens, outline="white")
    c.text(mx - 62, my + 4, "book", 12, INK, weight="bold", ha="center", va="center", rotation=90, clip=lens)
    c.text((x_sk + x_mt) / 2, my + 68, "skin", 10.5, "white", weight="bold", ha="center",
           va="center", rotation=90, clip=lens,
           path_effects=[pe.withStroke(linewidth=3 * PT, foreground="#0F766E")])
    c.text(x_mt + 9, my + 40, "finger", 10.5, INK, weight="bold", ha="center", va="center", rotation=90,
           clip=lens)
    c.text(x_np + 26, my - 34, "next book", 11, INK, weight="bold", ha="center", va="center", rotation=90,
           clip=lens)
    c.circle(mx, my, mr, "none", ec="white", lw=7, clip=clip)
    c.circle(mx, my, mr, "none", ec=INK, lw=2.5, clip=clip)
    c.pill(mx, top - 18, "top view inside the gap", INK, 168, size=12, clip=clip)


def draw_scene(c):
    x, y, w, h = 30, 30, 800, 740
    clip = c.card(x, y, w, h, r=26)
    c.gradient(x, y, w, h, ["#FCFAF6", "#F1EADF"], "v", clip=clip)

    # bookcase interior: back panel, inner left side, board top
    c.poly([P(132, TOP_Y, SHELF_D), P(940, TOP_Y, SHELF_D), P(940, 820, SHELF_D), P(132, 820, SHELF_D)],
           "#E3CFB2", clip=clip)
    c.poly([(132, TOP_Y), P(132, TOP_Y, SHELF_D), P(132, 820, SHELF_D), (132, 820)], "#C99A6B", clip=clip)
    c.poly([(132, BASE), (940, BASE), P(940, BASE, SHELF_D), P(132, BASE, SHELF_D)], "#D8AE80", clip=clip)

    # lower shelf: book tops under the board
    xl = 132
    for bw, dh, col in [(40, 0, "#5B6C8F"), (52, 6, "#B5543C"), (34, 2, "#7A9E7E"), (60, 8, "#2F3E5C"),
                        (44, 0, "#D9A441"), (38, 5, "#7B4F7A"), (56, 1, "#4F7C82"), (46, 7, "#8A8F4A"),
                        (40, 3, "#C08497"), (50, 0, "#3F4448"), (36, 6, "#C9B48A"), (48, 2, "#4C6A92"),
                        (40, 4, "#A0522D"), (32, 0, "#7C8B6A"), (52, 3, "#B5543C"), (44, 6, "#5B6C8F")]:
        top = 612 + dh
        c.poly([(xl, top), (xl + bw, top), P(xl + bw, top, BOOK_D), P(xl, top, BOOK_D)], PAGES, clip=clip)
        c.rect(xl, top, bw, 200, col, ec=shade(col, 0.6), lw=0.8, clip=clip)
        xl += bw
    c.rect(132, BASE + 26, 808, 16, "black", alpha=0.18, clip=clip)

    # board front (drawn first so the pulled-out book overhangs it)
    c.rect(132, BASE, 808, 26, "#B37B4E", clip=clip)
    c.rect(132, BASE, 808, 3, "#D9A877", clip=clip)

    # top shelf: books left of the target, the target (pulled out), finger, books right
    for x0, bw, bh, col, st in LEFT_BOOKS:
        draw_book(c, clip, x0, bw, bh, col, st)
    tx, tw, th, tcol = TARGET
    sx0, sy0 = P(tx, BASE, -PULL)
    c.poly([(sx0 + 4, sy0), (sx0 + tw + 10, sy0), (sx0 + tw + 22, sy0 + 8), (sx0 + 12, sy0 + 8)],
           "black", alpha=0.18, clip=clip)
    draw_book(c, clip, tx, tw, th, tcol, 0, d0=-PULL, band="#FCD34D")
    draw_right_finger(c, clip)
    for x0, bw, bh, col, st in RIGHT_BOOKS:
        draw_book(c, clip, x0, bw, bh, col, st)

    # left side panel, shelf above (little headroom)
    c.rect(110, 216, 22, 600, "#B9855A", clip=clip)
    c.poly([(110, 216), (940, 216), P(940, 216, SHELF_D), P(110, 216, SHELF_D)], "#E2BC92", clip=clip)
    c.rect(110, 216, 830, 22, "#B9855A", clip=clip)
    c.rect(110, 235, 830, 3, "black", alpha=0.12, clip=clip)

    # gripper: jaws, body, flange, wrist, arm (all in front of the shelf)
    for jx0, jx1 in ((446, 470), (526, 550)):
        draw_box(c, clip, jx0, jx1, 348, 426, -134, -112, "#475569", "#64748B", "#334155", r=3)
    draw_box(c, clip, 436, 560, 346, 430, -196, -134, "#334155", "#475569", "#1E293B", r=8)
    rx, ry = P(436, 346, -196)
    c.rect(rx + 12, ry + 66, 100, 7, "#1E293B", r=3.5, clip=clip)
    c.circle(rx + 108, ry + 14, 4.5, TEAL_L, clip=clip)

    far, near = np.array(P(498, 388, -196)), np.array(P(498, 388, -238))
    u = (near - far) / np.linalg.norm(near - far)
    n = np.array([u[1], -u[0]])
    r_fl = 27
    c.poly([far + r_fl * n, near + r_fl * n, near - r_fl * n, far - r_fl * n], "#B8C4D2", clip=clip)
    c.circle(*near, r_fl, "#E2E8F0", ec="#94A3B8", lw=1.5, clip=clip)
    wx, wy = P(498, 388, -252)
    draw_arm_link(c, clip, wx, wy, 210, 840, 58)
    draw_joint(c, clip, 210, 840, 52)
    draw_joint(c, clip, wx, wy, 36)

    # pull direction (toward the viewer)
    ax0, ay0 = P(606, 452, -30)
    ax1, ay1 = P(606, 452, -190)
    c.arrow(ax0, ay0, ax1, ay1, ORANGE, lw=5, head=20, clip=clip, outline="white")
    c.pill(ax0 + 44, ay0 - 8, "pull out", ORANGE, 82, size=14, clip=clip)

    # magnifier: top view of the gap where the right finger enters
    draw_magnifier(c, clip, 540, 387, 16, 706, 648, 92)

    # caption
    c.tag(60, 52, "B · APPLICATION", ORANGE, ORANGE_XL, 150, clip=clip)
    c.text(60, 106, "Pull one book out of a packed shelf", 24, INK, weight="bold", clip=clip)
    c.text(60, 134, "No headroom to lift it: thin shear-sensing fingers slide", 15, SLATE, clip=clip)
    c.text(60, 155, "into the gaps and pull the book out sideways.", 15, SLATE, clip=clip)
    c.rect(x, y, w, h, "none", ec=LINE, lw=1.2, r=26)


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
    for i, name in enumerate(["insert", "pinch", "pull"]):
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
