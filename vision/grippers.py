"""Gripper profiles: what the planner and the safety filter need to know about the fingers on the robot.

    BLADE_V1   2F-140 + blade v1 + PDMS skin (the MuJoCo assembly, ME295's sensor fingers): pads 32-161 mm apart,
               pad centre 0.269-0.292 m from the flange, blade tips at 0.333 m
    BARE_2F140 2F-140 with its standard fingertip pads (what was mounted on 2026-11-09): pads 7-136 mm apart,
               fingertip frames 0.191-0.215 m from the flange; pad centre 15 mm past the frame, tips <= 0.25 m (frame + 35 mm)
Both come from the same four-bar kinematics (baseline/embodiment, gripper_robotiq_2f140.toml, checked against MuJoCo);
the bare profile's pad length is measured from the wrist ZED's view of the real fingers (visible to 0.22 m open).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Tuple

import numpy as np

from .safety import ENVELOPE_BOXES, box_corners


@dataclass(frozen=True)
class GripperProfile:
    name: str
    width_table: Tuple[Tuple[float, float, int], ...]    # (pad gap m, TCP z m in the flange frame, register)
    tip_z: float                                         # lowest gripper point along the tool (flange frame, m)
    envelope: Tuple                                      # boxes (lo, hi) in the flange frame
    min_width: float
    max_open: float

    def _cols(self):
        a = np.array(self.width_table, float)
        o = np.argsort(a[:, 0])
        return a[o, 0], a[o, 1], a[o, 2]

    def register_for_width(self, width: float) -> int:
        w, _, r = self._cols()
        return int(round(np.interp(width, w, r)))

    def tcp_z(self, width: float) -> float:
        w, z, _ = self._cols()
        return float(np.interp(width, w, z))

    def tcp_pose6(self, width: float):
        return (0.0, 0.0, self.tcp_z(width), 0.0, 0.0, 0.0)

    def tool_points(self) -> np.ndarray:
        return np.vstack([box_corners(*b) for b in self.envelope])


@lru_cache(maxsize=None)
def _profiles():
    from baseline.embodiment import EndEffector
    ee = EndEffector.load("assembly_ur10e_2f140_dohyun_v1.toml")
    rows_blade, rows_bare = [], []
    for reg in range(0, 256, 5):
        w = ee.width_for_register(reg)
        q = ee.q_for_width(w)
        out = ee.fk(np.zeros(6), q)
        Fi = np.linalg.inv(out["flange"])
        L, R = Fi @ out["left_fingertip"], Fi @ out["right_fingertip"]
        rows_blade.append((float(w), float(ee.tcp_pose6(width=w)[2]), reg))
        gap = float(abs(L[0, 3] - R[0, 3]))
        rows_bare.append((gap, float(0.5 * (L[2, 3] + R[2, 3]) + 0.015), reg))
    blade = GripperProfile("2F-140 + blade v1 + PDMS", tuple(rows_blade), 0.338, ENVELOPE_BOXES, 0.034, 0.155)
    bare_env = (ENVELOPE_BOXES[0][0], (0.176, 0.130, 0.20)), ((-0.10, -0.035, 0.17), (0.10, 0.035, 0.25))
    bare = GripperProfile("2F-140, standard fingertips", tuple(rows_bare), 0.25,
                          ((tuple(bare_env[0][0]), tuple(bare_env[0][1])), (tuple(bare_env[1][0]), tuple(bare_env[1][1]))),
                          0.012, 0.130)
    return {"blade": blade, "bare": bare}


def profile(name: str) -> GripperProfile:
    return _profiles()[name]
