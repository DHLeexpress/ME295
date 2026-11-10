"""End-effector assemblies: compose component cards into forward kinematics for the arm + gripper + tool + sensor.

    ee = EndEffector.load("assembly_ur10e_2f140_dohyun_v1.toml")
    T = ee.fk(q_arm, q_grip)["tcp"]            # TCP pose in the robot base frame
    rob.set_tcp(ee.tcp_pose6(width=0.040))      # urx TCP for a 40 mm book
    reg = ee.register_for_width(0.046)          # Robotiq position register that opens the pads to 46 mm

TCP: origin midway between the two sensor pads' free faces. Two orientations:
  axes="flange" (default; what set_tcp uses and what the task poses in scripts_sim/run_librarian.py assume): the
      flange's axes. On this gripper the jaw axis is the flange's x axis.
  axes="jaw" (to retarget end-effector motion between robots, e.g. the DROID Franka): +z the approach direction,
      +y the jaw axis from the left pad to the right pad, +x = y x z.
On this gripper the TCP moves ~24 mm along z as the jaw closes (the 2F-140 fingertips swing on an arc), so the TCP is
a function of the opening. The blade's pads sit 0.21 mm off the fingertip axis, so the jaw axis is tilted 0.5 deg from
the flange x axis; the midpoint stays on the flange axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import se3
from .components import CARDS, DHArm, FingerTool, LinkageGripper, Mount, PadSensor, load_card

SIDES = ("left", "right")


@dataclass
class EndEffector:
    name: str
    arm: DHArm
    coupling: Mount
    gripper: LinkageGripper
    tool: FingerTool
    sensor: PadSensor
    mjcf: dict
    card_path: str = ""

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, card: str | Path = "assembly_ur10e_2f140_dohyun_v1.toml") -> "EndEffector":
        c = load_card(card)
        return cls(name=c["name"],
                   arm=DHArm.from_card(load_card(c["arm"])),
                   coupling=Mount.from_card(load_card(c["coupling"])),
                   gripper=LinkageGripper.from_card(load_card(c["gripper"])),
                   tool=FingerTool.from_card(load_card(c["tool"])),
                   sensor=PadSensor.from_card(load_card(c["sensor"])),
                   mjcf=c.get("mjcf", {}), card_path=c["_path"])

    # ------------------------------------------------------------------ frames in the flange frame
    def gripper_base(self) -> np.ndarray:
        return self.coupling.T @ self.gripper.mount

    def fingertip(self, side: str, q: float) -> np.ndarray:
        return self.gripper_base() @ self.gripper.fingertip(side, q)

    def tool_frame(self, side: str, q: float) -> np.ndarray:
        return self.fingertip(side, q) @ self.tool.mount

    def pad(self, side: str, q: float) -> np.ndarray:
        """Sensor pad frame (mid-thickness; +z into the layer) in the flange frame."""
        return self.tool_frame(side, q) @ self.sensor.mount

    def pad_face(self, side: str, q: float) -> np.ndarray:
        """Free-face centre of the pad, +z outward (toward the grasped object), in the flange frame."""
        return self.pad(side, q) @ self.sensor.face_frame()

    def jaw_axis(self, q: float) -> np.ndarray:
        """Unit vector from the left pad's face centre to the right one's, normal to the flange z (flange frame)."""
        pl, pr = self.pad_face("left", q)[:3, 3], self.pad_face("right", q)[:3, 3]
        y = pr - pl
        y[2] = 0.0
        return y / np.linalg.norm(y)

    def tcp(self, q: float, axes: str = "flange") -> np.ndarray:
        """TCP in the flange frame at driver angle q (see the module docstring for the two orientations)."""
        pl, pr = self.pad_face("left", q)[:3, 3], self.pad_face("right", q)[:3, 3]
        if axes == "flange":
            return se3.make(None, 0.5 * (pl + pr))
        if axes == "jaw":
            z = np.array([0.0, 0.0, 1.0])
            y = self.jaw_axis(q)
            return se3.make(np.stack([np.cross(y, z), y, z], axis=1), 0.5 * (pl + pr))
        raise ValueError(f"axes must be 'flange' or 'jaw', not {axes!r}")

    def width(self, q: float) -> float:
        """Gap between the two pads' free faces along the jaw axis (m)."""
        pl, pr = self.pad_face("left", q), self.pad_face("right", q)
        return float((pr[:3, 3] - pl[:3, 3]) @ self.jaw_axis(q))

    def q_for_width(self, w: float) -> float:
        """Driver angle at which the pad gap equals w (bisection; the gap shrinks monotonically with q)."""
        lo, hi = 0.0, self.gripper.q_closed
        if w >= self.width(lo):
            return lo
        if w <= self.width(hi):
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.width(mid) > w else (lo, mid)
        return 0.5 * (lo + hi)

    def register_for_width(self, w: float) -> int:
        return self.gripper.register_from_q(self.q_for_width(w))

    def width_for_register(self, reg: float) -> float:
        return self.width(self.gripper.q_from_register(reg))

    def tcp_pose6(self, width: Optional[float] = None, q: Optional[float] = None, axes: str = "flange") -> tuple:
        """urx set_tcp() tuple [x, y, z, rx, ry, rz] for the TCP at a given pad gap (or driver angle)."""
        if q is None:
            q = self.q_for_width(width if width is not None else 0.0)
        return tuple(float(v) for v in se3.pose6(self.tcp(q, axes)))

    # ------------------------------------------------------------------ base frame
    def fk(self, q_arm: Sequence[float], q_grip: float) -> Dict[str, np.ndarray]:
        """All frames in the robot base frame."""
        F = self.arm.fk(q_arm)
        out = {"flange": F, "gripper_base": F @ self.gripper_base(), "tcp": F @ self.tcp(q_grip),
               "tcp_jaw": F @ self.tcp(q_grip, "jaw")}
        for s in SIDES:
            out[f"{s}_fingertip"] = F @ self.fingertip(s, q_grip)
            out[f"{s}_tool"] = F @ self.tool_frame(s, q_grip)
            out[f"{s}_pad"] = F @ self.pad(s, q_grip)
            out[f"{s}_pad_face"] = F @ self.pad_face(s, q_grip)
        return out

    def ik_tcp(self, T_tcp: np.ndarray, q_seed: Sequence[float], q_grip: float, **kw):
        """Arm joints that put the TCP (at opening q_grip) at T_tcp."""
        return self.arm.ik(T_tcp, q_seed, T_tool=self.tcp(q_grip), **kw)

    def describe(self) -> str:
        q_open, q_closed = 0.0, self.gripper.q_closed
        lines = [f"{self.name}  ({Path(self.card_path).name})",
                 f"  arm      {self.arm.name} ({self.arm.convention} DH, {self.arm.dof} dof)",
                 f"  coupling {self.coupling.name}",
                 f"  gripper  {self.gripper.name}",
                 f"  tool     {self.tool.name} {self.tool.version} ({self.tool.mesh})",
                 f"  sensor   {self.sensor.name}: {self.sensor.size[0] * 1e3:.1f} x {self.sensor.size[1] * 1e3:.1f} x "
                 f"{self.sensor.thickness * 1e3:.2f} mm, {self.sensor.taxels[0]} x {self.sensor.taxels[1]} taxels",
                 f"  pad gap  open {self.width(q_open) * 1e3:.1f} mm, closed {self.width(q_closed) * 1e3:.1f} mm",
                 f"  TCP z    open {self.tcp(q_open)[2, 3] * 1e3:.1f} mm, closed {self.tcp(q_closed)[2, 3] * 1e3:.1f}"
                 f" mm from the flange"]
        return "\n".join(lines)


@dataclass
class FixedTCPArm:
    """An arm with a fixed TCP in the same convention (used for the virtual DROID Franka when retargeting)."""
    name: str
    arm: DHArm
    T_flange_tcp: np.ndarray
    base_in_world: np.ndarray
    source: str = ""

    @classmethod
    def load(cls, card: str | Path) -> "FixedTCPArm":
        from .components import _pose
        c = load_card(card)
        return cls(name=c["name"], arm=DHArm.from_card(load_card(c["arm"])), T_flange_tcp=_pose(c["tcp"]),
                   base_in_world=_pose(c.get("base_in_world", {})), source=c.get("source", ""))

    def tcp(self, q: Sequence[float]) -> np.ndarray:
        """TCP in this arm's base frame."""
        return self.arm.fk(q) @ self.T_flange_tcp

    def ik(self, T_tcp: np.ndarray, q_seed: Sequence[float], **kw):
        return self.arm.ik(T_tcp, q_seed, T_tool=self.T_flange_tcp, **kw)


def available_cards() -> list:
    return sorted(p.name for p in CARDS.glob("*.toml"))
