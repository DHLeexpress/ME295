"""Closed-loop position-based visual servoing (Chaumette & Hutchinson 2006; closed-loop grasping as in GG-CNN,
Morrison et al. 2018): perception at 5 Hz updates a filtered grasp goal; align above it, then descend while still
correcting laterally; the goal freezes 8 cm above the grasp (the camera is too close / the fingers hide the object),
and the last part is open loop."""
from __future__ import annotations

import numpy as np

from .base import TopDownGraspController


class PBVSGrasp(TopDownGraspController):
    name = "PBVS (closed loop)"
    closed_loop = True

    def __init__(self, *a, k=2.0, v_max=0.12, v_descend=0.04, align_tol=0.008, **kw):
        super().__init__(*a, **kw)
        self.k, self.v_max, self.v_descend, self.align_tol = k, v_max, v_descend, align_tol

    def reset_approach(self, obs):
        self.stage = "align"

    def approach(self, obs, goal):
        p = obs.tcp_pose[:3]
        if self.stage == "align":
            e = goal.T_pre[:3, 3] - p
            v = self.k * e
            n = np.linalg.norm(v)
            if n > self.v_max:
                v *= self.v_max / n
            if np.linalg.norm(e[:2]) < self.align_tol and abs(e[2]) < 0.02:
                self.stage = "descend"
            return v, False
        e = goal.position - p
        v = np.r_[self.k * 1.5 * e[:2], np.clip(self.k * e[2], -self.v_descend, self.v_descend)]
        return v, bool(np.linalg.norm(e) < self.reach_tol)
