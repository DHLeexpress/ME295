"""Table / workspace safety filter for every command sent to the UR10e (simulator or robot).

Same contract as the lab's ../ta_utils.py (safety_filter, joint_path_safety_filter) and fk.SafetyCheck: fail closed,
check sampled joint paths before a move, standard UR10e DH kinematics (numpy only, runs on the robot PC). What it
checks is the real end effector instead of the ultrasound probe point:

  tool points   the 8 corners of the gripper + wrist-ZED envelope (axis-aligned box in the flange frame, measured on
                the MuJoCo model of the 2F-140 + blades + ZED, fingers open and closed, padded) must stay
                z_table + margin_tool above the table;
  link frames   every joint frame must stay z_table + margin_links above it (fk.SafetyCheck with a margin);
  workspace     the TCP stays inside a box (x, y, z limits).

For streamed velocity commands (speedl), filter_twist() is a control-barrier-function filter (Ames et al. 2019): every
tool point may approach the table no faster than gamma x its clearance, d/dt(z_i) >= -gamma (z_i - z_min), likewise the
workspace faces; the twist is projected onto those half-spaces (closest twist, Dykstra's projection) and capped.
The table height defaults to the top of the ChArUco board recorded by the hand-eye calibration (z = 6.5 mm in the
base frame); the vision estimate of the table plane is only used to warn when the two disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

# UR10e standard DH (Universal Robots; identical to ../fk.py and baseline/embodiment/cards/arm_ur10e.toml)
DH_A = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0])
DH_D = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

# Gripper + ZED envelope in the flange (tool0) frame, metres: 2F-140 + blade v1 + PDMS + wrist ZED Mini, every
# opening from 161 mm to closed, measured on the MuJoCo assembly (geom bounding boxes), padded 5 mm. Two boxes: the
# body (coupling, 2F-140 base and linkages, ZED) and, below it, the fingertips + blades, which are much narrower.
ENVELOPE_BOXES = (((-0.176, -0.093, -0.017), (0.176, 0.130, 0.265)),     # body + ZED
                  ((-0.089, -0.044, 0.255), (0.089, 0.044, 0.338)))      # fingertips + blades (tips at 0.333)
ENVELOPE_MIN = (-0.176, -0.093, -0.017)        # their union's bounding box
ENVELOPE_MAX = (0.176, 0.130, 0.338)
TIP_Z = 0.338                                  # lowest point along the tool (blade tips + pad), flange frame


def fk_frames(q: Sequence[float]) -> np.ndarray:
    """(7, 4, 4): base, then joint frames 1..6 (the last is the flange, tool0)."""
    T = np.eye(4)
    out = [T.copy()]
    for a, d, al, th in zip(DH_A, DH_D, DH_ALPHA, np.asarray(q, float)):
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]])
        out.append(T.copy())
    return np.array(out)


def box_corners(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _rotvec(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-6:
        w, V = np.linalg.eigh(0.5 * (R + R.T))
        k = V[:, np.argmax(w)]
        return th * k
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _slerp_R(R0, R1, s):
    r = _rotvec(R0.T @ R1) * s
    th = np.linalg.norm(r)
    if th < 1e-12:
        return R0.copy()
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return R0 @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)


@dataclass
class SafetyConfig:
    z_table: float = 0.0065                 # table top (base frame, m): ChArUco board top in the calibration
    margin_tool: float = 0.010              # tool points above the table
    margin_links: float = 0.050             # joint frames above the table
    x_range: Tuple[float, float] = (-0.70, 0.90)     # TCP workspace (base frame)
    y_range: Tuple[float, float] = (-1.05, 0.95)
    z_max: float = 1.10
    v_max: float = 0.20                     # TCP speed cap (m/s)
    w_max: float = 0.60                     # angular speed cap (rad/s)
    gamma: float = 4.0                      # barrier rate (1/s): at 5 cm clearance, descend at most 0.2 m/s
    point_v_max: Optional[float] = None     # cap on the speed of EVERY tool point + the flange (rotation included)
    tool_points: np.ndarray = field(default_factory=lambda: np.vstack([box_corners(*b) for b in ENVELOPE_BOXES]))


@dataclass
class SafetyReport:
    safe: bool
    clearance: float          # lowest tool point above z_table + margin_tool would be 0 (m); < 0 unsafe
    link_clearance: float
    message: str


class SafetyFilter:
    def __init__(self, cfg: SafetyConfig = SafetyConfig()):
        self.cfg = cfg
        self.interventions = 0      # barrier (table / workspace) changed a command
        self.speed_caps = 0         # only the speed cap did

    # ------------------------------------------------------------------ static checks
    def tool_points(self, T_base_flange: np.ndarray) -> np.ndarray:
        P = self.cfg.tool_points
        return P @ T_base_flange[:3, :3].T + T_base_flange[:3, 3]

    def check_pose(self, T_base_flange: np.ndarray, tcp_xyz: Optional[np.ndarray] = None) -> SafetyReport:
        c = self.cfg
        zmin = float(self.tool_points(T_base_flange)[:, 2].min())
        clear = zmin - (c.z_table + c.margin_tool)
        msgs = []
        if clear < 0:
            msgs.append(f"tool {1e3 * (zmin - c.z_table):.1f} mm above the table (< {1e3 * c.margin_tool:.0f} mm)")
        p = T_base_flange[:3, 3] if tcp_xyz is None else np.asarray(tcp_xyz)
        if not (c.x_range[0] <= p[0] <= c.x_range[1] and c.y_range[0] <= p[1] <= c.y_range[1] and p[2] <= c.z_max):
            msgs.append(f"TCP {np.round(p, 3).tolist()} outside the workspace box")
            clear = min(clear, -1e-3)
        return SafetyReport(not msgs, clear, float("inf"), "; ".join(msgs) or f"safe (tool clearance {1e3 * clear:.0f} mm)")

    def check_q(self, q: Sequence[float], T_flange_tcp: Optional[np.ndarray] = None) -> SafetyReport:
        """One joint vector: links and tool points (fail closed on bad input)."""
        try:
            F = fk_frames(q)
        except Exception as e:  # noqa: BLE001
            return SafetyReport(False, -np.inf, -np.inf, f"unsafe: cannot evaluate {q!r}: {e}")
        tcp = (F[-1] @ T_flange_tcp)[:3, 3] if T_flange_tcp is not None else None
        r = self.check_pose(F[-1], tcp)
        link = float(F[1:, 2, 3].min()) - (self.cfg.z_table + self.cfg.margin_links)
        if link < 0:
            return SafetyReport(False, r.clearance, link, f"a joint frame is {1e3 * (link + self.cfg.margin_links):.0f} "
                                                          f"mm above the table; {r.message}")
        return SafetyReport(r.safe, r.clearance, link, r.message)

    def check_path(self, q0, q1, steps: int = 25, T_flange_tcp=None) -> Tuple[bool, SafetyReport, int]:
        """movej: sampled joint-space path (ta_utils.joint_path_safety_filter semantics)."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        rep = SafetyReport(False, -np.inf, -np.inf, "not evaluated")
        for i, s in enumerate(np.linspace(0.0, 1.0, max(2, steps))):
            rep = self.check_q((1 - s) * q0 + s * q1, T_flange_tcp)
            if not rep.safe:
                return False, rep, i
        return True, rep, max(2, steps) - 1

    def check_linear(self, T0: np.ndarray, T1: np.ndarray, steps: int = 25) -> Tuple[bool, SafetyReport, int]:
        """movel between two flange poses: the controller moves the TCP on a line; the tool points are checked at
        sampled poses (orientation interpolated). Joint frames need IK: check those with check_q at the ends."""
        rep = SafetyReport(False, -np.inf, -np.inf, "not evaluated")
        for i, s in enumerate(np.linspace(0.0, 1.0, max(2, steps))):
            T = np.eye(4)
            T[:3, :3] = _slerp_R(T0[:3, :3], T1[:3, :3], s)
            T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
            rep = self.check_pose(T)
            if not rep.safe:
                return False, rep, i
        return True, rep, max(2, steps) - 1

    # ------------------------------------------------------------------ velocity filter
    def filter_twist(self, T_base_flange: np.ndarray, tcp_xyz: np.ndarray, v: Sequence[float],
                     iters: int = 60) -> Tuple[np.ndarray, bool]:
        """Closest safe speedl twist [v; w] (base frame, linear velocity of the TCP) to the requested one.
        Returns (twist, barrier_active): barrier_active is True when a table / workspace barrier changed the command
        (speed caps alone are counted in self.speed_caps)."""
        c = self.cfg
        v_des = np.asarray(v, float).copy()
        A, b = [], []
        ez = np.array([0.0, 0.0, 1.0])
        z_min = c.z_table + c.margin_tool
        for p in self.tool_points(T_base_flange):          # tool points above the table
            r = p - tcp_xyz
            A.append(np.r_[ez, np.cross(r, ez)])
            b.append(-c.gamma * (p[2] - z_min))
        x, y, z = tcp_xyz
        for axis, lo, hi in ((0, *c.x_range), (1, *c.y_range)):  # TCP workspace box
            e = np.zeros(6)
            e[axis] = 1.0
            A.append(e.copy())
            b.append(-c.gamma * (tcp_xyz[axis] - lo))
            A.append(-e)
            b.append(-c.gamma * (hi - tcp_xyz[axis]))
        e = np.zeros(6)
        e[2] = -1.0
        A.append(e)
        b.append(-c.gamma * (c.z_max - z))
        A, b = np.array(A), np.array(b)
        out = v_des
        barrier = bool(np.any(A @ out < b - 1e-12))
        if barrier:
            # Dykstra: exact projection onto the intersection of the half-spaces {u: a_i u >= b_i}
            n = len(b)
            incr = np.zeros((n, 6))
            u = v_des.copy()
            for _ in range(iters):
                for i in range(n):
                    w = u + incr[i]
                    viol = b[i] - A[i] @ w
                    proj = w + (viol / (A[i] @ A[i])) * A[i] if viol > 0 else w
                    incr[i] = w - proj
                    u = proj
                if np.all(A @ u >= b - 1e-6):
                    break
            out = u
        lin, ang = out[:3], out[3:]
        sl, sa = np.linalg.norm(lin), np.linalg.norm(ang)
        if sl > c.v_max:
            lin = lin * (c.v_max / sl)
        if sa > c.w_max:
            ang = ang * (c.w_max / sa)
        out2 = np.r_[lin, ang]
        if c.point_v_max is not None:            # slowest uniform scaling that keeps every point under the cap
            P = np.vstack([self.tool_points(T_base_flange), T_base_flange[:3, 3], tcp_xyz])
            sp = np.linalg.norm(lin[None, :] + np.cross(ang[None, :], P - tcp_xyz), axis=1).max()
            if sp > c.point_v_max:
                out2 = out2 * (c.point_v_max / sp)
        capped = not np.allclose(out2, out)
        out = out2
        self.interventions += int(barrier)
        self.speed_caps += int(capped and not barrier)
        return out, barrier
