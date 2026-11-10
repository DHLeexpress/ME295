"""Tabletop perception and top-down grasp planning from one calibrated RGB-D frame (numpy only).

    objs, z_table = segment_objects(frame, T_base_flange)       # what stands on the table
    grasp = plan_topdown_grasp(objs[0], z_table)                 # where to put the TCP, how far to open

1. Table: a plane z = a x + b y + c fitted to the points around the strongest height peak near z_table_guess
   (robust least squares, tilt < max_tilt). A plane, not a height: a hand-eye rotation error (or a table that is
   not level in the base frame) tilts the table by ~1 cm per 0.5 deg across the view, which a flat threshold would
   turn into "objects".
2. Objects: points in the workspace box higher than min_height above the table plane, minus the robot's own gripper and
   camera (points inside the flange-frame envelope of vision.safety, padded), grouped by 8-connected cells of a
   top-view occupancy grid; each group's footprint is the minimum-area rectangle of its top-view convex hull
   (rotating calipers), height from its highest points.
3. Grasp (parallel jaw, approach straight down): jaws close across the footprint's short side; the TCP (between the
   pads) goes to grasp_depth below the top, but never so low that the blade tips come within tip_clearance of the
   table; open to the width + 2 x clearance. Pre-grasp: approach_height above, same orientation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .camera import RGBDFrame
from .safety import ENVELOPE_BOXES, TIP_Z


@dataclass
class TabletopParams:
    x_range: Tuple[float, float] = (-0.35, 0.55)      # workspace (base frame, m)
    y_range: Tuple[float, float] = (-0.85, -0.15)
    z_max: float = 0.40
    z_table_guess: float = 0.0
    table_search: float = 0.06
    min_height: float = 0.010          # above the table plane to count as object
    max_tilt_deg: float = 6.0
    cell: float = 0.005                # top-view grid
    min_points: int = 150
    min_cell_points: int = 3           # a top-view cell belongs to the footprint with at least this many points
    top_percentile: float = 95.0
    stride: int = 1
    robot_pad: float = 0.01            # padding of the gripper envelope when removing the robot's own points
    top_slice: float = 0.05            # the grasp footprint / centre come from the points this far below the top
    envelope: tuple = ENVELOPE_BOXES   # gripper boxes (flange frame) for removing the robot's own points


@dataclass
class TableObject:
    points: np.ndarray
    rgb: np.ndarray
    center: np.ndarray                 # footprint centre at mid-height (base frame)
    yaw: float                         # direction of the footprint's long side (rad)
    length: float                      # footprint long side (m)
    width: float                       # footprint short side (m)
    height: float
    top_z: float
    n: int
    z_base: float = 0.0                # table plane height under the object

    def summary(self) -> str:
        return (f"object at ({self.center[0]:+.3f}, {self.center[1]:+.3f}) {1e3 * self.length:.0f} x "
                f"{1e3 * self.width:.0f} mm footprint, {1e3 * self.height:.0f} mm tall, yaw {np.degrees(self.yaw):+.0f} deg "
                f"({self.n} pts)")


@dataclass
class GripperSpec:
    """2F-140 + blade v1 + PDMS (baseline/embodiment card): pad gap 32.2 mm closed .. 161 mm open."""
    max_open: float = 0.155
    min_width: float = 0.034
    clearance: float = 0.018           # air between each pad and the object while descending
    table_margin: float = 0.013        # lowest gripper point above the table at the grasp (safety margin + 3 mm)
    grasp_depth: float = 0.030         # TCP this far below the object's top (pads 71 mm long)
    approach_height: float = 0.12
    # flange -> TCP (pad centres) distance vs pad gap: the 2F-140 fingertips swing on an arc
    # (baseline/embodiment EndEffector.tcp_pose6, checked against MuJoCo)
    tcp_table: Tuple[Tuple[float, float], ...] = ((0.0322, 0.2924), (0.04, 0.2923), (0.06, 0.2914), (0.08, 0.2894),
                                                  (0.10, 0.2864), (0.12, 0.2821), (0.161, 0.2687))

    def tcp_z(self, width: float) -> float:
        w, z = np.array(self.tcp_table).T
        o = np.argsort(w)                       # np.interp needs increasing widths (profiles list open -> closed)
        return float(np.interp(width, w[o], z[o]))


@dataclass
class TopDownGrasp:
    position: np.ndarray               # TCP at the grasp (base frame)
    yaw: float                         # closing axis (tool x) direction in the table plane (rad)
    width: float                       # object width across the jaws
    opening: float                     # pad gap while approaching
    T_tcp: np.ndarray                  # 4x4 TCP pose: tool z down, tool x = closing axis
    T_pre: np.ndarray                  # pre-grasp TCP pose (approach_height above)
    ok: bool = True
    reason: str = ""
    obj: Optional[TableObject] = None
    stamp: float = 0.0

    def summary(self) -> str:
        return (f"grasp TCP ({self.position[0]:+.3f}, {self.position[1]:+.3f}, {self.position[2]:.3f}) yaw "
                f"{np.degrees(self.yaw):+.0f} deg, object width {1e3 * self.width:.0f} mm, open {1e3 * self.opening:.0f} mm"
                f"{'' if self.ok else ' NOT GRASPABLE: ' + self.reason}")


# ----------------------------------------------------------------------------------------------- geometry helpers
def convex_hull(p: np.ndarray) -> np.ndarray:
    """2-D convex hull (Andrew's monotone chain), counter-clockwise."""
    pts = np.unique(np.round(p, 5), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def half(ps):
        h = []
        for q in ps:
            while len(h) >= 2 and cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, up = half(pts), half(pts[::-1])
    return np.array(lo[:-1] + up[:-1])


def min_area_rect(p: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """(centre (2,), yaw of the long side, length, width) of the minimum-area rectangle around 2-D points."""
    h = convex_hull(p)
    if len(h) < 3:
        c = p.mean(0)
        return c, 0.0, 0.0, 0.0
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        n = np.linalg.norm(e)
        if n < 1e-9:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        a, b = h @ u, h @ v
        area = (a.max() - a.min()) * (b.max() - b.min())
        if best is None or area < best[0]:
            best = (area, u, v, a, b)
    _, u, v, a, b = best
    c = u * 0.5 * (a.max() + a.min()) + v * 0.5 * (b.max() + b.min())
    la, lb = a.max() - a.min(), b.max() - b.min()
    if la >= lb:
        return c, float(np.arctan2(u[1], u[0])), float(la), float(lb)
    return c, float(np.arctan2(v[1], v[0])), float(lb), float(la)


def _label8(occ: np.ndarray) -> Tuple[np.ndarray, int]:
    try:
        from scipy import ndimage
        return ndimage.label(occ, structure=np.ones((3, 3), int))
    except ImportError:   # BFS fallback
        lab = np.zeros(occ.shape, int)
        n = 0
        for s in zip(*np.nonzero(occ)):
            if lab[s]:
                continue
            n += 1
            stack = [s]
            lab[s] = n
            while stack:
                i, j = stack.pop()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a, b = i + di, j + dj
                        if 0 <= a < occ.shape[0] and 0 <= b < occ.shape[1] and occ[a, b] and not lab[a, b]:
                            lab[a, b] = n
                            stack.append((a, b))
        return lab, n


def wrap_half_pi(a: float) -> float:
    """Angle of an undirected axis, in (-pi/2, pi/2]."""
    a = (a + np.pi / 2) % np.pi - np.pi / 2
    return float(a)


# ----------------------------------------------------------------------------------------------- perception
def estimate_table_z(z: np.ndarray, guess: float, search: float) -> Optional[float]:
    sel = z[np.abs(z - guess) < search]
    if len(sel) < 500:
        return None
    edges = np.arange(guess - search, guess + search + 1e-9, 0.002)
    h, _ = np.histogram(sel, edges)
    k = int(np.argmax(np.convolve(h, [1, 2, 1], "same")))
    z0 = 0.5 * (edges[k] + edges[k + 1])
    inl = sel[np.abs(sel - z0) < 0.004]
    return float(np.median(inl))


def fit_table_plane(pts: np.ndarray, guess: float, search: float, max_tilt_deg: float = 6.0) -> Optional[np.ndarray]:
    """(a, b, c) of the table plane z = a x + b y + c (base frame), or None."""
    cand = pts[np.abs(pts[:, 2] - guess) < search]
    if len(cand) < 500:
        return None
    z0 = estimate_table_z(cand[:, 2], guess, search)
    sel = cand[np.abs(cand[:, 2] - z0) < 0.012]
    coef = np.array([0.0, 0.0, z0])
    for thr in (0.012, 0.006, 0.003):
        A = np.c_[sel[:, 0], sel[:, 1], np.ones(len(sel))]
        coef = np.linalg.lstsq(A, sel[:, 2], rcond=None)[0]
        r = cand[:, 2] - np.c_[cand[:, :2], np.ones(len(cand))] @ coef
        sel = cand[np.abs(r) < thr]
        if len(sel) < 300:
            break
    if np.degrees(np.arctan(np.hypot(coef[0], coef[1]))) > max_tilt_deg:
        return np.array([0.0, 0.0, z0])
    return coef


def robot_mask(points: np.ndarray, T_base_flange: np.ndarray, pad: float, boxes=ENVELOPE_BOXES) -> np.ndarray:
    """True for points inside the gripper + camera envelope (the robot seeing itself)."""
    Tinv = np.linalg.inv(T_base_flange)
    pf = points @ Tinv[:3, :3].T + Tinv[:3, 3]
    inside = np.zeros(len(points), bool)
    for lo, hi in boxes:
        inside |= np.all((pf >= np.asarray(lo) - pad) & (pf <= np.asarray(hi) + pad), axis=1)
    return inside


def segment_objects(frame: RGBDFrame, T_base_flange: Optional[np.ndarray] = None,
                    params: TabletopParams = TabletopParams()) -> Tuple[List[TableObject], Optional[float]]:
    """Objects standing on the table, largest first, and the table height at the workspace centre (None if the
    table is not in view; each object carries the plane's height under it in z_base)."""
    P = params
    pts, rgb = frame.points_base(stride=P.stride)
    box = ((pts[:, 0] > P.x_range[0]) & (pts[:, 0] < P.x_range[1]) & (pts[:, 1] > P.y_range[0])
           & (pts[:, 1] < P.y_range[1]) & (pts[:, 2] < P.z_max) & (pts[:, 2] > P.z_table_guess - P.table_search))
    pts, rgb = pts[box], rgb[box]
    if T_base_flange is not None and len(pts):
        keep = ~robot_mask(pts, T_base_flange, P.robot_pad, P.envelope)
        pts, rgb = pts[keep], rgb[keep]
    plane = fit_table_plane(pts, P.z_table_guess, P.table_search, P.max_tilt_deg) if len(pts) else None
    if plane is None:
        return [], None
    xc, yc = 0.5 * (P.x_range[0] + P.x_range[1]), 0.5 * (P.y_range[0] + P.y_range[1])
    zt = float(plane @ [xc, yc, 1.0])
    base = np.c_[pts[:, :2], np.ones(len(pts))] @ plane
    up = pts[:, 2] - base > P.min_height
    pts, rgb, base = pts[up], rgb[up], base[up]
    if len(pts) < P.min_points:
        return [], zt
    ij = np.floor((pts[:, :2] - [P.x_range[0], P.y_range[0]]) / P.cell).astype(int)
    shape = (int(np.ceil((P.x_range[1] - P.x_range[0]) / P.cell)) + 1,
             int(np.ceil((P.y_range[1] - P.y_range[0]) / P.cell)) + 1)
    cnt = np.zeros(shape, int)
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    occ = cnt >= P.min_cell_points          # sparse cells are stereo outliers along the viewing rays
    lab, n = _label8(occ)
    pl = lab[ij[:, 0], ij[:, 1]]
    origin = np.array([P.x_range[0], P.y_range[0]])
    objs = []
    for k in range(1, n + 1):
        sel = pl == k
        if sel.sum() < P.min_points:
            continue
        p = pts[sel]
        top = float(np.percentile(p[:, 2], P.top_percentile))
        # footprint of the part the jaws will meet: the cells holding points within top_slice of the top (a tall
        # bottle touching a low box is grasped by its own top part, not by both)
        up = p[:, 2] > top - P.top_slice
        ijk = np.unique(ij[sel][up], axis=0)
        if len(ijk) < 3:
            ijk = np.argwhere(lab == k)
        cells = ijk
        corners = np.concatenate([origin + (cells + np.array(o)) * P.cell for o in ((0, 0), (1, 0), (0, 1), (1, 1))])
        c2, yaw, L, W = min_area_rect(corners)
        zb = float(plane @ [c2[0], c2[1], 1.0])
        h = top - zb
        objs.append(TableObject(points=p, rgb=rgb[sel], center=np.array([c2[0], c2[1], zb + 0.5 * h]), yaw=yaw,
                                length=L, width=W, height=h, top_z=top, n=int(sel.sum()), z_base=zb))
    objs.sort(key=lambda o: -o.n)
    return objs, zt


# ----------------------------------------------------------------------------------------------- grasp
def tcp_pose(position: Sequence[float], yaw: float) -> np.ndarray:
    """TCP pose: tool z straight down, tool x (the jaws' closing axis) at angle yaw in the table plane."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = position
    return T


def plan_topdown_grasp(obj: TableObject, z_table: Optional[float] = None, grip: GripperSpec = GripperSpec(),
                       prefer_yaw: Optional[float] = None, stamp: float = 0.0, profile=None) -> TopDownGrasp:
    """Grasp across the footprint's short side. z_table: the table height the safety filter enforces; the grasp is
    planned against the higher of it and the fitted plane under the object (obj.z_base), so the goal is always one
    the safety filter lets the gripper reach. prefer_yaw: of the two equivalent jaw directions (yaw, yaw + pi), take the one closer to this (e.g.
    the current tool yaw) to avoid needless wrist turns."""
    z_table = obj.z_base if z_table is None else max(z_table, obj.z_base)
    if profile is not None:          # another finger set (vision.grippers): its widths, TCP and tips
        from dataclasses import replace
        grip = replace(grip, max_open=profile.max_open, min_width=profile.min_width,
                       tcp_table=tuple((w, z) for w, z, _ in profile.width_table))
        tip_z = profile.tip_z
    else:
        tip_z = TIP_Z
    yaw = obj.yaw + np.pi / 2
    if prefer_yaw is not None:
        alt = yaw + np.pi
        d = lambda a: abs((a - prefer_yaw + np.pi) % (2 * np.pi) - np.pi)
        yaw = yaw if d(yaw) <= d(alt) else alt
    yaw = float((yaw + np.pi) % (2 * np.pi) - np.pi)
    width = obj.width
    opening = min(grip.max_open, width + 2 * grip.clearance)
    # the TCP is set between the pads at the object's width; the gripper's lowest point is TIP_Z - tcp_z below it
    z = max(z_table + grip.table_margin + (tip_z - grip.tcp_z(width)), obj.top_z - grip.grasp_depth)
    pos = np.array([obj.center[0], obj.center[1], z])
    ok, why = True, ""
    if width + 2 * 0.005 > grip.max_open:
        ok, why = False, f"{1e3 * width:.0f} mm wide, the jaws open to {1e3 * grip.max_open:.0f} mm"
    elif width < grip.min_width:
        ok, why = False, f"{1e3 * width:.0f} mm wide, the pads close to {1e3 * grip.min_width:.0f} mm"
    T = tcp_pose(pos, yaw)
    Tp = T.copy()
    Tp[2, 3] += grip.approach_height
    return TopDownGrasp(position=pos, yaw=yaw, width=width, opening=opening, T_tcp=T, T_pre=Tp, ok=ok, reason=why,
                        obj=obj, stamp=stamp)


def draw_tabletop(frame: RGBDFrame, objs: List[TableObject], grasp: Optional[TopDownGrasp] = None,
                  z_table: Optional[float] = None) -> np.ndarray:
    """RGB overlay: footprints (green), the grasp (magenta jaw line + pre-grasp), table grid. Needs OpenCV."""
    import cv2
    img = np.ascontiguousarray(frame.rgb.copy())
    Tinv = np.linalg.inv(frame.T_base_cam)

    def px(p):
        pc = Tinv[:3, :3] @ np.asarray(p, float) + Tinv[:3, 3]
        if pc[2] <= 0.02:
            return None
        u = frame.intr.project(pc[None])[0]
        return int(round(u[0])), int(round(u[1]))
    if z_table is not None:
        for x in np.arange(-0.3, 0.51, 0.1):
            a, b = px([x, -0.8, z_table]), px([x, -0.2, z_table])
            if a and b:
                cv2.line(img, a, b, (200, 200, 200), 1, cv2.LINE_AA)
        for y in np.arange(-0.8, -0.19, 0.1):
            a, b = px([-0.3, y, z_table]), px([0.5, y, z_table])
            if a and b:
                cv2.line(img, a, b, (200, 200, 200), 1, cv2.LINE_AA)
    for o in objs:
        u = np.array([np.cos(o.yaw), np.sin(o.yaw)])
        v = np.array([-u[1], u[0]])
        for z in (o.top_z, o.top_z - o.height):
            cs = [px([*(o.center[:2] + sa * 0.5 * o.length * u + sb * 0.5 * o.width * v), z])
                  for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            if all(cs):
                cv2.polylines(img, [np.array(cs)], True, (0, 220, 0), 2, cv2.LINE_AA)
    if grasp is not None:
        x = grasp.T_tcp[:3, 0]
        a = px(grasp.position - 0.5 * grasp.width * x)
        b = px(grasp.position + 0.5 * grasp.width * x)
        c = px(grasp.T_pre[:3, 3])
        if a and b:
            cv2.line(img, a, b, (255, 0, 255), 3, cv2.LINE_AA)
        if c:
            cv2.drawMarker(img, c, (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(img, grasp.summary()[:100], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    return img
