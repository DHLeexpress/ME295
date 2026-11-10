"""Shelf-row perception from one RGB-D frame in the robot base frame: the spine plane, every book's edges,
thickness and colour, the air gaps beside each book, and the book to pick.

Method (numpy only; geometry first, colour only to split packed books and to name them):

1. Spine plane. Points inside a prior box around the shelf tier (ShelfPrior) are binned by depth along the prior's
   into-shelf axis; the strongest peak is the spines (they are the largest surface facing the robot). A line fit of
   that peak against the along-row coordinate refines the plane's yaw. The plane is assumed vertical.
2. Ray casting. Every pixel's ray is intersected with the spine plane, which gives the place (u along the row,
   h up) the pixel "looks at" on the plane. The measured point on the same ray says what is there:
       spine   within tol of the plane              -> a book's spine
       behind  deeper than tol                      -> air: a gap (or past the end of the row)
       front   nearer than tol                      -> blocked (the gripper, a hand): unknown, never a gap
       none    no depth                             -> unknown (stereo shadow; a gap if it is short and between books)
   so a gripper in front of the shelf is not mistaken for a gap, whatever the camera pose.
3. Profile. In a horizontal band just above the board, columns of du are labelled book (spine fraction >= 0.5) or
   gap, short no-depth runs between books become gaps, and book runs are split where the spine colour changes.
4. Books. Edges -> thickness and centre; spine heights -> top; gaps on each side; median spine colour -> name.
   select() picks the book to grasp: gaps on both sides wide enough for a blade, then the named colour.

Lengths in metres, all outputs in the base frame. Validated in the simulator against ground truth
(vision/scripts/eval_sim_perception.py); real ZED depth smooths edges, so expect wider error bars on hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import RGBDFrame

# Named colours for "the {colour} book" (sRGB 0..1). The first eight are the simulator's palette
# (baseline/sim/scene.py), "dark blue" is the original target's cloth.
COLOURS: Dict[str, Tuple[float, float, float]] = {
    "red": (0.62, 0.07, 0.07), "blue": (0.06, 0.16, 0.52), "green": (0.07, 0.40, 0.15),
    "yellow": (0.86, 0.72, 0.10), "white": (0.90, 0.89, 0.85), "black": (0.05, 0.05, 0.06),
    "orange": (0.88, 0.40, 0.06), "purple": (0.36, 0.12, 0.46), "dark blue": (0.055, 0.115, 0.36),
}


# ----------------------------------------------------------------------------------------------- colour
def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) sRGB in 0..1 (or uint8) -> CIE Lab (D65)."""
    c = np.asarray(rgb, float)
    if np.issubdtype(np.asarray(rgb).dtype, np.integer):
        c = c / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def colour_distance(lab_a: np.ndarray, lab_b: np.ndarray, w_light: float = 0.5) -> np.ndarray:
    """Lab distance with lightness down-weighted (shading changes L much more than hue)."""
    d = np.asarray(lab_a, float) - np.asarray(lab_b, float)
    return np.sqrt((w_light * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def name_colour(rgb, colours: Dict[str, Sequence[float]] = COLOURS) -> Tuple[str, float]:
    lab = srgb_to_lab(np.asarray(rgb, float) / (255.0 if np.max(rgb) > 1.0 else 1.0))
    names = list(colours)
    d = colour_distance(lab[None], srgb_to_lab(np.array([colours[n] for n in names])))
    k = int(np.argmin(d))
    return names[k], float(d[k])


# ----------------------------------------------------------------------------------------------- types
@dataclass
class ShelfPrior:
    """Where the shelf tier roughly is, in the base frame. half = half extents along (row, into, up)."""
    center: Tuple[float, float, float] = (0.0, 0.87, 0.54)      # simulator shelf A, upper tier
    half: Tuple[float, float, float] = (0.36, 0.09, 0.14)
    into: Tuple[float, float, float] = (0.0, 1.0, 0.0)          # into the shelf, horizontal
    max_yaw_deg: float = 15.0                                   # refuse a plane turned further than this


@dataclass
class Params:
    du: Optional[float] = None       # profile column width (m); None = from the pixel footprint on the plane
    tol: float = 0.006               # |distance to the spine plane| counted as spine (m)
    band: Tuple[float, float] = (0.02, 0.10)   # profile band: from 2 cm above the lowest spines, 10 cm tall
    min_hits: int = 3                # rays per column needed to label it
    max_nodata_gap: float = 0.03     # a no-depth run between two books narrower than this is a gap
    min_gap: float = 0.002           # narrower air than this counts as packed
    split_lab: float = 22.0          # colour step that splits a packed run into two books
    min_book: float = 0.012          # thinnest book a colour split may create (m)
    stride: int = 1                  # use every stride-th pixel


@dataclass
class Book:
    u_left: float                    # edges along the row axis (m, from the plane origin)
    u_right: float
    thickness: float
    center: np.ndarray               # spine-face centre at mid-height, base frame (m)
    z_bottom: float
    z_top: float
    top_seen: bool                   # False: the top edge was blocked or out of view (z_top is a lower bound)
    rgb: np.ndarray                  # median spine colour (uint8)
    colour: str
    colour_dist: float
    gap_left: Optional[float]        # air beside the book (m); 0 = packed; None = unknown (blocked / end of view)
    gap_right: Optional[float]

    @property
    def min_gap(self) -> float:
        g = [x for x in (self.gap_left, self.gap_right)]
        return float(min(g)) if all(x is not None for x in g) else float("nan")

    def summary(self) -> str:
        f = lambda g: "?" if g is None else f"{1e3 * g:.1f}"
        return (f"{self.colour:9s} {1e3 * self.thickness:5.1f} mm at x={self.center[0]:+.3f} "
                f"(gaps {f(self.gap_left)} | {f(self.gap_right)} mm, top z={self.z_top:.3f}{'' if self.top_seen else '+'})")


@dataclass
class ShelfEstimate:
    origin: np.ndarray               # a point on the spine plane (base frame)
    row: np.ndarray                  # along the row, left -> right seen from the robot
    into: np.ndarray                 # spine-plane normal, into the shelf
    up: np.ndarray
    yaw_deg: float                   # plane yaw relative to the prior's into axis
    spine_offset: float              # plane offset along the prior's into axis from the prior centre (m)
    books: List[Book]
    z_board: float                   # lowest spine points (~ the board top)
    profile: dict = field(default_factory=dict)   # columns for plots: u, label, occupancy, colour
    n_points: int = 0

    def point(self, u: float, h: float) -> np.ndarray:
        """Base-frame point on the spine plane at row coordinate u and height h (z)."""
        p = self.origin + u * self.row
        p[2] = h
        return p

    @property
    def spine_y(self) -> float:
        """Spine-plane y at the plane origin (for a shelf facing the robot along +y)."""
        return float(self.origin[1])

    def select(self, colour: Optional[str] = None, min_gap: float = 0.004) -> Optional[Book]:
        """The book to grasp: known air on both sides of at least min_gap (a blade is 3.5 mm thick), then the
        closest named colour (colour given) or the widest smaller gap (no colour)."""
        cands = [b for b in self.books if b.gap_left is not None and b.gap_right is not None
                 and min(b.gap_left, b.gap_right) >= min_gap]
        if not cands:
            return None
        if colour:
            ref = srgb_to_lab(np.array(COLOURS[colour]) if colour in COLOURS else np.asarray(colour, float))
            return min(cands, key=lambda b: float(colour_distance(srgb_to_lab(b.rgb / 255.0), ref)))
        return max(cands, key=lambda b: b.min_gap)

    def describe(self) -> str:
        lines = [f"spine plane y={self.spine_y:.4f} (offset {1e3 * self.spine_offset:+.1f} mm), yaw "
                 f"{self.yaw_deg:+.2f} deg, board z={self.z_board:.3f}, {len(self.books)} books:"]
        lines += [f"  {i:2d}  {b.summary()}" for i, b in enumerate(self.books)]
        return "\n".join(lines)


class PerceptionError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- estimator
def _frame_axes(into, up=(0.0, 0.0, 1.0)):
    into = np.asarray(into, float)
    up = np.asarray(up, float)
    into = into - into.dot(up) * up
    into /= np.linalg.norm(into)
    row = np.cross(into, up)
    return row, into, up


def _runs(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """[(label, start, end_exclusive)] of consecutive equal labels."""
    out, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append((int(labels[s]), s, i))
            s = i
    return out


BOOK, GAP, BLOCKED, NODATA = 1, 2, 3, 0


def estimate_shelf(frame: RGBDFrame, prior: ShelfPrior = ShelfPrior(), params: Params = Params()) -> ShelfEstimate:
    if frame.T_base_cam is None:
        raise PerceptionError("the frame has no camera pose")
    P = params
    row, into, up = _frame_axes(prior.into)
    c0 = np.asarray(prior.center, float)
    hx, hy, hz = prior.half

    # ---- 1. spine plane from the points inside the prior box
    pts, _ = frame.points_base(stride=P.stride)
    q = (pts - c0) @ np.stack([row, into, up], 1)
    box = (np.abs(q[:, 0]) < hx) & (np.abs(q[:, 1]) < hy) & (np.abs(q[:, 2]) < hz)
    q = q[box]
    if len(q) < 500:
        raise PerceptionError(f"only {len(q)} depth points inside the shelf prior box: is the shelf in view?")
    edges = np.arange(-hy, hy + 1e-9, 0.002)
    hist, _ = np.histogram(q[:, 1], edges)
    hist = np.convolve(hist, [1, 2, 1], "same")
    a0 = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    near = np.abs(q[:, 1] - a0) < 0.012
    u_in, a_in = q[near, 0], q[near, 1]
    beta, alpha = 0.0, a0
    for thr in (0.008, 0.004):        # robust line a = alpha + beta u
        A = np.c_[u_in, np.ones_like(u_in)]
        beta, alpha = np.linalg.lstsq(A, a_in, rcond=None)[0]
        keep = np.abs(a_in - (alpha + beta * u_in)) < thr
        u_in, a_in = u_in[keep], a_in[keep]
    yaw = float(np.degrees(np.arctan(beta)))
    if abs(yaw) > prior.max_yaw_deg:
        raise PerceptionError(f"spine plane yawed {yaw:.1f} deg from the prior: not a shelf face?")
    n = into - beta * row
    n /= np.linalg.norm(n)
    row_p = np.cross(n, up)
    origin = c0 + alpha * into

    # ---- 2. cast every pixel's ray onto the plane; label by the measured point on that ray
    intr = frame.intr
    T = frame.T_base_cam
    v_idx, u_idx = np.mgrid[0:intr.height:P.stride, 0:intr.width:P.stride]
    rays_c = np.stack([(u_idx - intr.cx) / intr.fx, (v_idx - intr.cy) / intr.fy, np.ones(u_idx.shape)], -1)
    rays = rays_c.reshape(-1, 3) @ T[:3, :3].T
    cam = T[:3, 3]
    dn = rays @ n
    t = np.where(dn > 1e-6, ((origin - cam) @ n) / np.where(dn > 1e-6, dn, 1.0), np.nan)
    hit = cam + t[:, None] * rays
    hu = (hit - origin) @ row_p
    hh = hit[:, 2]
    xyz = frame.xyz[::P.stride, ::P.stride].reshape(-1, 3)
    rgb = frame.rgb[::P.stride, ::P.stride].reshape(-1, 3)
    valid = np.isfinite(xyz).all(1)
    p_meas = xyz @ T[:3, :3].T + T[:3, 3]
    s = np.where(valid, (p_meas - origin) @ n, np.nan)
    on_plane = np.isfinite(t) & (np.abs(hu) < hx) & (np.abs(hh - c0[2]) < hz)
    lab = np.full(len(s), NODATA, np.int8)
    lab[valid & (np.abs(s) <= P.tol)] = BOOK
    lab[valid & (s > P.tol)] = GAP
    lab[valid & (s < -P.tol)] = BLOCKED
    spine = on_plane & (lab == BOOK)
    if spine.sum() < 200:
        raise PerceptionError("almost no spine pixels on the fitted plane")
    z_board = float(np.percentile(hh[spine], 1.0))

    # ---- 3. profile along the row in a band above the board
    dist = float(np.median(t[spine]))
    du = P.du or max(0.0008, 1.25 * dist / intr.fx * P.stride)
    band = on_plane & (hh > z_board + P.band[0]) & (hh < z_board + P.band[0] + P.band[1])
    u_edges = np.arange(-hx, hx + 1e-9, du)
    ncol = len(u_edges) - 1
    col = np.clip(np.digitize(hu[band], u_edges) - 1, 0, ncol - 1)
    cnt = np.zeros((4, ncol), int)
    np.add.at(cnt, (lab[band].astype(int), col), 1)
    n_book, n_gap, n_block, n_none = cnt[BOOK], cnt[GAP], cnt[BLOCKED], cnt[NODATA]
    seen = n_book + n_gap
    occ = n_book / np.maximum(seen, 1)
    labels = np.where(seen >= P.min_hits, np.where(occ >= 0.5, BOOK, GAP),
                      np.where(n_block > n_none, BLOCKED, NODATA)).astype(int)
    # majority filter over 3 columns (single-column flips are pixel noise)
    sm = labels.copy()
    for i in range(1, ncol - 1):
        w = labels[i - 1:i + 2]
        vals, c = np.unique(w, return_counts=True)
        if c.max() >= 2:
            sm[i] = vals[np.argmax(c)]
    labels = sm
    # short no-depth runs between two books are gaps (the second camera could not see into them)
    runs = _runs(labels)
    for k, (lb, a, b) in enumerate(runs):
        if lb == NODATA and 0 < k < len(runs) - 1 and runs[k - 1][0] == BOOK and runs[k + 1][0] == BOOK \
                and (b - a) * du <= P.max_nodata_gap:
            labels[a:b] = GAP
    # gaps narrower than min_gap are packed books
    for lb, a, b in _runs(labels):
        if lb == GAP and (b - a) * du < P.min_gap:
            labels[a:b] = BOOK

    # per-column spine colour (Lab) for splitting packed runs
    bcol = band & (lab == BOOK)
    colb = np.clip(np.digitize(hu[bcol], u_edges) - 1, 0, ncol - 1)
    col_rgb = np.full((ncol, 3), np.nan)
    order = np.argsort(colb, kind="stable")
    cs, rs = colb[order], rgb[bcol][order]
    starts = np.searchsorted(cs, np.arange(ncol))
    ends = np.searchsorted(cs, np.arange(ncol), side="right")
    for i in range(ncol):
        if ends[i] > starts[i]:
            col_rgb[i] = np.median(rs[starts[i]:ends[i]], axis=0)
    col_lab = srgb_to_lab(np.nan_to_num(col_rgb) / 255.0)
    col_lab[np.isnan(col_rgb[:, 0])] = np.nan

    def splits(a: int, b: int) -> List[int]:
        w = max(2, int(round(0.004 / du)))
        m = max(1, int(round(P.min_book / du)))
        e = np.zeros(b - a)
        for i in range(a + w, b - w):
            L = np.nanmedian(col_lab[i - w:i], axis=0)
            R = np.nanmedian(col_lab[i + 1:i + 1 + w], axis=0)
            if np.isfinite(L).all() and np.isfinite(R).all():
                e[i - a] = float(colour_distance(L, R))
        cuts, last = [], a
        for i in np.argsort(-e):
            if e[i] < P.split_lab:
                break
            c = a + int(i)
            if c - a >= m and b - c >= m and all(abs(c - x) >= m for x in cuts):
                cuts.append(c)
        return sorted(cuts)

    # ---- 4. books with their gaps
    segs = []   # (label, a, b)
    for lb, a, b in _runs(labels):
        if lb == BOOK:
            cuts = [a, *splits(a, b), b]
            segs += [(BOOK, cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        else:
            segs.append((lb, a, b))
    books: List[Book] = []
    full = on_plane & (lab == BOOK)
    for k, (lb, a, b) in enumerate(segs):
        if lb != BOOK:
            continue
        u0, u1 = u_edges[a], u_edges[b]
        if u1 - u0 < P.min_book * 0.75:
            continue

        def gap(j):
            if j < 0 or j >= len(segs):
                return None
            lbj, aj, bj = segs[j]
            if lbj == BOOK:
                return 0.0
            if lbj == GAP:
                return float((bj - aj) * du)
            return None
        inner = full & (hu > u0 + 0.25 * (u1 - u0)) & (hu < u1 - 0.25 * (u1 - u0))
        hs = hh[inner]
        z_top = float(np.percentile(hs, 99.0)) if len(hs) else z_board
        # top seen if the rays just above it see air (behind) rather than a blocker / nothing
        above = on_plane & (hu > u0) & (hu < u1) & (hh > z_top + 0.004) & (hh < z_top + 0.02)
        top_seen = bool(above.sum() > 0 and np.mean(lab[above] == GAP) > 0.6)
        cols = col_rgb[a:b]
        med = np.nanmedian(cols, axis=0) if np.isfinite(cols).any() else np.zeros(3)
        name, dcol = name_colour(med)
        uc = 0.5 * (u0 + u1)
        zc = 0.5 * (z_board + z_top)
        books.append(Book(u_left=float(u0), u_right=float(u1), thickness=float(u1 - u0),
                          center=origin + uc * row_p + (zc - origin[2]) * up, z_bottom=z_board, z_top=z_top,
                          top_seen=top_seen, rgb=np.round(med).astype(np.uint8), colour=name, colour_dist=dcol,
                          gap_left=gap(k - 1), gap_right=gap(k + 1)))
    prof = {"u": 0.5 * (u_edges[:-1] + u_edges[1:]), "label": labels, "occupancy": occ, "rgb": col_rgb,
            "du": du, "n": cnt, "distance": dist}
    return ShelfEstimate(origin=origin, row=row_p, into=n, up=up, yaw_deg=yaw, spine_offset=float(alpha),
                         books=books, z_board=z_board, profile=prof, n_points=int(box.sum()))


# ----------------------------------------------------------------------------------------------- drawing
def draw(frame: RGBDFrame, est: ShelfEstimate, target: Optional[Book] = None) -> np.ndarray:
    """RGB overlay: book outlines on the spine plane (green), gaps (yellow), the target (magenta). Needs OpenCV."""
    import cv2
    img = np.ascontiguousarray(frame.rgb.copy())
    Tinv = np.linalg.inv(frame.T_base_cam)

    def px(p):
        pc = Tinv[:3, :3] @ p + Tinv[:3, 3]
        if pc[2] <= 0:
            return None
        uv = frame.intr.project(pc[None])[0]
        return int(round(uv[0])), int(round(uv[1]))

    def quad(u0, u1, z0, z1, colour, thick):
        c = [px(est.point(u0, z0)), px(est.point(u1, z0)), px(est.point(u1, z1)), px(est.point(u0, z1))]
        if all(x is not None for x in c):
            cv2.polylines(img, [np.array(c, np.int32)], True, colour, thick, cv2.LINE_AA)
        return c
    for b in est.books:
        is_t = target is not None and b is target
        quad(b.u_left, b.u_right, b.z_bottom, b.z_top, (255, 0, 255) if is_t else (0, 220, 0), 3 if is_t else 1)
        for side, g in ((-1, b.gap_left), (1, b.gap_right)):
            if g:
                u = b.u_left if side < 0 else b.u_right
                quad(u, u + side * g, b.z_bottom + 0.02, b.z_bottom + 0.12, (255, 220, 0), 2)
    if target is not None:
        c = px(target.center)
        if c:
            cv2.drawMarker(img, c, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
        txt = f"target: {target.colour} {1e3 * target.thickness:.1f} mm, gaps {1e3 * (target.gap_left or 0):.1f}/" \
              f"{1e3 * (target.gap_right or 0):.1f} mm"
        cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv2.LINE_AA)
    return img
