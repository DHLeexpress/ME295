"""Open-loop baseline (sense -> plan -> act, as Dex-Net / GPD grasps are executed): one look from the view
configuration, then a time-parametrised path pre-grasp -> grasp (trapezoidal speed, tracked with a P term), no
further perception."""
from __future__ import annotations

import numpy as np

from .base import TopDownGraspController


def _trapezoid(L, v, a):
    if L < 1e-9:
        return 0.0, lambda t: L
    ta = v / a
    if a * ta * ta > L:
        ta = np.sqrt(L / a)
        v = a * ta
    tf = (L - a * ta * ta) / v
    T = 2 * ta + tf

    def s(t):
        t = min(max(t, 0.0), T)
        if t < ta:
            return 0.5 * a * t * t
        if t < ta + tf:
            return 0.5 * a * ta * ta + v * (t - ta)
        return L - 0.5 * a * (T - t) ** 2
    return T, s


class ScriptedGrasp(TopDownGraspController):
    name = "scripted (open loop)"
    closed_loop = False

    def __init__(self, *a, v_approach=0.12, v_descend=0.04, acc=0.3, kp=3.0, **kw):
        super().__init__(*a, **kw)
        self.v_approach, self.v_descend, self.acc, self.kp = v_approach, v_descend, acc, kp

    def reset_approach(self, obs):
        self.plan = None

    def _segment(self, t0, start, end, v):
        L = float(np.linalg.norm(end - start))
        T, s = _trapezoid(L, v, self.acc)
        return (t0, T, start.copy(), end.copy(), L, s)

    def approach(self, obs, goal):
        p = obs.tcp_pose[:3]
        if self.plan is None:                                   # 1: straight to the pre-grasp
            self.plan = self._segment(obs.t, p, goal.T_pre[:3, 3], self.v_approach)
            self.stage = "pre"
        t0, T, a, b, L, s = self.plan
        if obs.t <= t0 + T:
            u = (b - a) / L if L > 1e-9 else np.zeros(3)
            tau = obs.t - t0
            vref = u * (s(tau + 1e-3) - s(tau)) / 1e-3
            return vref + self.kp * (a + u * s(tau) - p), False
        err = b - p
        if self.stage == "pre":
            if np.linalg.norm(err) < 0.003 and self.rot_error_deg(obs) <= self.align_deg:
                self.plan = self._segment(obs.t, p, goal.position, self.v_descend)   # 2: straight down
                self.stage = "down"
            return self.kp * err, False
        return self.kp * err, bool(np.linalg.norm(err) < self.reach_tol)
