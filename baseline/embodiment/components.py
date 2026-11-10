"""Components of a robot end-effector chain, each described by a TOML card in baseline/embodiment/cards/.

The chain mirrors the hardware stack in model/ (see model/DESCRIPTION.md):

    arm (DH)  ->  coupling (fixed)  ->  gripper linkage (q)  ->  finger tool on each fingertip (fixed)
              ->  sensor pad on each tool (fixed; its free face is the contact surface)

Every component is a frozen dataclass built from its card, so a new blade or a new skin is a new card, not new code.
All lengths in metres, angles in radians, frames right-handed. Numpy only (no MuJoCo): the same code runs on the
robot PC. baseline/embodiment/check.py verifies every card against the MuJoCo assembly the simulator uses.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import se3

CARDS = Path(__file__).resolve().parent / "cards"


def load_card(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CARDS / p
    with open(p, "rb") as f:
        card = tomllib.load(f)
    card["_path"] = str(p)
    return card


def _pose(tbl: dict) -> np.ndarray:
    """{pos, rpy} or {pos, quat} table -> 4x4."""
    pos = tbl.get("pos", (0.0, 0.0, 0.0))
    if "quat" in tbl:
        return se3.from_pos_quat(pos, tbl["quat"])
    return se3.from_pos_rpy(pos, tbl.get("rpy", (0.0, 0.0, 0.0)))


# --------------------------------------------------------------------------------------------- arm
@dataclass(frozen=True)
class DHArm:
    """Serial arm in Denavit-Hartenberg form.

    standard (UR):          T_i = Rz(q_i + offset_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
    modified (Craig/Franka): T_i = Rx(alpha_{i-1}) Tx(a_{i-1}) Rz(q_i + offset_i) Tz(d_i)
    then the fixed `flange` transform. fk(q) is the flange (tool0) pose in the arm's base frame.
    """
    name: str
    convention: str
    a: Tuple[float, ...]
    d: Tuple[float, ...]
    alpha: Tuple[float, ...]
    offset: Tuple[float, ...]
    flange: np.ndarray = field(default_factory=lambda: np.eye(4))
    q_home: Tuple[float, ...] = ()
    q_min: Tuple[float, ...] = ()
    q_max: Tuple[float, ...] = ()
    source: str = ""

    @classmethod
    def from_card(cls, c: dict) -> "DHArm":
        n = len(c["d"])
        return cls(name=c["name"], convention=c["convention"], a=tuple(c["a"]), d=tuple(c["d"]),
                   alpha=tuple(c["alpha"]), offset=tuple(c.get("offset", [0.0] * n)),
                   flange=_pose(c.get("flange", {})), q_home=tuple(c.get("q_home", [0.0] * n)),
                   q_min=tuple(c.get("q_min", [-2 * np.pi] * n)), q_max=tuple(c.get("q_max", [2 * np.pi] * n)),
                   source=c.get("source", ""))

    @property
    def dof(self) -> int:
        return len(self.d)

    def _link(self, i: int, qi: float) -> np.ndarray:
        th = qi + self.offset[i]
        if self.convention == "standard":
            return se3.make(se3.rot_z(th)) @ se3.make(p=(0, 0, self.d[i])) @ se3.make(p=(self.a[i], 0, 0)) \
                @ se3.make(se3.rot_x(self.alpha[i]))
        if self.convention == "modified":
            return se3.make(se3.rot_x(self.alpha[i])) @ se3.make(p=(self.a[i], 0, 0)) \
                @ se3.make(se3.rot_z(th)) @ se3.make(p=(0, 0, self.d[i]))
        raise ValueError(f"unknown DH convention {self.convention!r}")

    def frames(self, q: Sequence[float]) -> List[np.ndarray]:
        """Joint frames: frames[i] is the frame whose z is joint i's axis (i = 0..dof-1), frames[dof] the flange."""
        T = np.eye(4)
        out = []
        for i, qi in enumerate(q):
            if self.convention == "standard":
                out.append(T.copy())          # joint i rotates about z of the previous frame
                T = T @ self._link(i, qi)
            else:
                T_pre = T @ se3.make(se3.rot_x(self.alpha[i])) @ se3.make(p=(self.a[i], 0, 0))
                out.append(T_pre)             # modified DH: joint i rotates about z after Rx Tx
                T = T @ self._link(i, qi)
        out.append(T @ self.flange)
        return out

    def fk(self, q: Sequence[float]) -> np.ndarray:
        return self.frames(q)[-1]

    def jacobian(self, q: Sequence[float], T_tool: np.ndarray = np.eye(4)) -> np.ndarray:
        """Geometric Jacobian (6 x dof, base frame) of the point/frame flange @ T_tool."""
        fr = self.frames(q)
        p = (fr[-1] @ T_tool)[:3, 3]
        J = np.zeros((6, self.dof))
        for i in range(self.dof):
            z, o = fr[i][:3, 2], fr[i][:3, 3]
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return J

    def ik(self, T_goal: np.ndarray, q_seed: Sequence[float], T_tool: np.ndarray = np.eye(4),
           iters: int = 300, tol: float = 1e-9, max_step: float = 0.2, q_rest: Sequence[float] | None = None,
           w_rest: float = 0.0) -> Tuple[np.ndarray, float]:
        """Levenberg-Marquardt (adaptive damping, step <= max_step rad, joint limits) on flange @ T_tool = T_goal.
        For a redundant arm (dof > 6), q_rest/w_rest pull the null space toward a rest posture. Returns
        (q, |[position error (m), rotation error (rad)]|)."""
        q = np.array(q_seed, float)
        lo, hi = np.array(self.q_min), np.array(self.q_max)
        lam = 1e-2
        e = se3.twist_error(self.fk(q) @ T_tool, T_goal)
        err = float(np.linalg.norm(e))
        for _ in range(iters):
            if err < tol:
                break
            J = self.jacobian(q, T_tool)
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            if q_rest is not None and w_rest > 0 and self.dof > 6:
                N = np.eye(self.dof) - np.linalg.pinv(J) @ J
                dq += N @ (w_rest * (np.asarray(q_rest, float) - q))
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q_new = np.clip(q + dq, lo, hi)
            e_new = se3.twist_error(self.fk(q_new) @ T_tool, T_goal)
            err_new = float(np.linalg.norm(e_new))
            if err_new < err:
                q, e, err = q_new, e_new, err_new
                lam = max(lam / 3.0, 1e-7)
            else:
                lam = min(lam * 5.0, 1e3)
        return q, err


# --------------------------------------------------------------------------------------------- fixed mount
@dataclass(frozen=True)
class Mount:
    """A rigid adapter (e.g. the Robotiq coupling): child frame in the parent frame."""
    name: str
    T: np.ndarray
    source: str = ""

    @classmethod
    def from_card(cls, c: dict) -> "Mount":
        return cls(name=c["name"], T=_pose(c["mount"]), source=c.get("source", ""))


# --------------------------------------------------------------------------------------------- gripper
@dataclass(frozen=True)
class LinkStep:
    T: np.ndarray                 # fixed transform from the previous frame (applied before the joint)
    axis: Tuple[float, float, float] | None = None   # joint axis in this frame, None = fixed
    coef: float = 0.0             # joint angle = coef * q (q = driver angle; URDF mimic ratios)


@dataclass(frozen=True)
class LinkageGripper:
    """Parallel gripper whose fingertip frames are serial chains of the driver angle q (mimic joints), e.g. the
    Robotiq 2F-140 four-bar: q = 0 open .. q_closed closed, Robotiq register 0..255 linear in q."""
    name: str
    mount: np.ndarray                         # gripper base in the coupling/flange frame
    chains: Dict[str, Tuple[LinkStep, ...]]   # "left"/"right": base -> fingertip frame
    q_closed: float
    register_max: int = 255
    stroke: float = 0.0
    force_range: Tuple[float, float] = (0.0, 0.0)
    source: str = ""

    @classmethod
    def from_card(cls, c: dict) -> "LinkageGripper":
        chains = {}
        for side, steps in c["chain"].items():
            chains[side] = tuple(LinkStep(T=_pose(s), axis=tuple(s["axis"]) if "axis" in s else None,
                                          coef=float(s.get("coef", 0.0))) for s in steps)
        return cls(name=c["name"], mount=_pose(c["mount"]), chains=chains, q_closed=float(c["q_closed"]),
                   register_max=int(c.get("register_max", 255)), stroke=float(c.get("stroke", 0.0)),
                   force_range=tuple(c.get("force_range", (0.0, 0.0))), source=c.get("source", ""))

    def fingertip(self, side: str, q: float) -> np.ndarray:
        """Fingertip frame of `side` in the gripper base frame at driver angle q."""
        T = np.eye(4)
        for s in self.chains[side]:
            T = T @ s.T
            if s.axis is not None:
                T = T @ se3.make(se3.axis_angle(s.axis, s.coef * q))
        return T

    def q_from_register(self, reg: float) -> float:
        return float(np.clip(reg, 0, self.register_max)) / self.register_max * self.q_closed

    def register_from_q(self, q: float) -> int:
        return int(round(np.clip(q / self.q_closed, 0.0, 1.0) * self.register_max))


# --------------------------------------------------------------------------------------------- finger tool
@dataclass(frozen=True)
class FingerTool:
    """A custom fingertip (e.g. the blade gripper_dohyun_v1) bolted to each fingertip frame."""
    name: str
    version: str
    mount: np.ndarray            # tool (STL) frame in the fingertip frame, same for both sides
    mesh: str = ""
    mass: float = 0.0
    thickness: float = 0.0       # plate + pad thickness that must fit in a gap (informational, from the STL)
    source: str = ""

    @classmethod
    def from_card(cls, c: dict) -> "FingerTool":
        return cls(name=c["name"], version=str(c.get("version", "")), mount=_pose(c["mount"]),
                   mesh=c.get("mesh", ""), mass=float(c.get("mass", 0.0)),
                   thickness=float(c.get("thickness", 0.0)), source=c.get("source", ""))


# --------------------------------------------------------------------------------------------- sensor pad
@dataclass(frozen=True)
class PadSensor:
    """A tactile pad on each finger tool (e.g. the 1.5 mm PDMS skin).

    `mount` is the pad frame (mid-thickness centre; x across the blade, y toward the tip, +z into the layer) in the
    tool frame. The contact surface is the free face at z = -thickness/2. face_frame() returns a frame on the free
    face whose +z is the outward normal (toward the object) and +y toward the fingertip.
    """
    name: str
    mount: np.ndarray
    size: Tuple[float, float]          # (across x, along y)
    thickness: float
    taxels: Tuple[int, int] = (5, 7)   # (nx, ny)
    source: str = ""

    @classmethod
    def from_card(cls, c: dict) -> "PadSensor":
        return cls(name=c["name"], mount=_pose(c["mount"]), size=tuple(c["size"]), thickness=float(c["thickness"]),
                   taxels=tuple(c.get("taxels", (5, 7))), source=c.get("source", ""))

    def face_frame(self) -> np.ndarray:
        """Free-face centre, +z outward (toward the object), in the pad frame."""
        return se3.make(se3.rot_y(np.pi), (0.0, 0.0, -0.5 * self.thickness))  # x' = -x, y' = y, z' = -z
