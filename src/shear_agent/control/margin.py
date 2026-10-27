"""Placeholder contact-phase policy: squeeze just enough, then pull the book out and lift it.

Layer A of the project in its simplest rule-based form. Each blade's skin gives the margin of its
contact patch against sliding and twisting, q = 1 - sqrt((|T|/(mu N))^2 + (|M|/(mu N r))^2)
(TactileFrame.net_margin; the twist M comes from the shear field, 20 Hz frames, see
robot.tactile.SkinSensor); the grasp margin is the weaker blade, Q = min(q_left, q_right). The gripper force register is raised whenever Q < q_star and the arm pauses while
Q < q_pause. (The per-taxel minimum is not used: edge taxels sit near the friction limit at any
squeeze, the partial-slip rim of a frictional contact.) To be replaced by MPC / MPPI / a learned
policy with the same Controller interface.
"""
from __future__ import annotations

import numpy as np

from .base import Command, Controller, Observation


class ShearMarginGrasp(Controller):
    name = "shear-margin (rule-based placeholder)"
    rate_hz = 50.0

    def __init__(self, pull_dir=(0.0, -1.0, 0.0), pull_dist: float = 0.20, pull_speed: float = 0.04,
                 lift: float = 0.10, lift_speed: float = 0.03, q_star: float = 0.5, q_pause: float = 0.2,
                 contact_force: float = 3.0, force_start: int = 0, force_step: int = 6, hold_time: float = 1.0):
        self.pull_dir = np.asarray(pull_dir, float) / np.linalg.norm(pull_dir)
        self.pull_dist, self.pull_speed = pull_dist, pull_speed
        self.lift, self.lift_speed = lift, lift_speed
        self.q_star, self.q_pause = q_star, q_pause
        self.contact_force = contact_force
        self.force_start, self.force_step = force_start, force_step
        self.hold_time = hold_time

    def reset(self, obs: Observation) -> None:
        self.phase = "close"
        self.force = self.force_start
        self.p0 = obs.tcp_pose[:3].copy()
        self.t_phase = obs.t
        self.margin = 1.0
        self.stamp = None

    def _margin(self, obs: Observation) -> float:
        if not obs.tactile:
            return 1.0  # no skin (e.g. real robot before the sensor is wired): rely on the force register
        return min(f.net_margin() for f in obs.tactile.values())

    def _normal(self, obs: Observation) -> float:
        if not obs.tactile:
            return np.inf if obs.gripper_object == 2 else 0.0
        return min(f.total_normal for f in obs.tactile.values())

    def act(self, obs: Observation) -> Command:
        self.margin = q = self._margin(obs)
        stamp = max(f.stamp for f in obs.tactile.values()) if obs.tactile else obs.t
        fresh = stamp != self.stamp  # react once per sensor frame, not once per control tick
        self.stamp = stamp
        if self.phase == "close":  # close with the lowest force until both blades touch the book
            if self._normal(obs) > self.contact_force and obs.gripper_object != 0:
                self.phase, self.p0, self.t_phase = "pull", obs.tcp_pose[:3].copy(), obs.t
            return Command(np.zeros(6), gripper_position=255, gripper_force=self.force, phase=self.phase)

        if q < self.q_star and fresh:  # the patch is approaching slip: squeeze harder
            self.force = int(min(255, self.force + self.force_step))
        v = np.zeros(6)
        moved = obs.tcp_pose[:3] - self.p0
        if self.phase == "pull":
            if moved @ self.pull_dir >= self.pull_dist:
                self.phase, self.p0, self.t_phase = "lift", obs.tcp_pose[:3].copy(), obs.t
            else:
                v[:3] = self.pull_speed * self.pull_dir
        elif self.phase == "lift":
            if moved[2] >= self.lift:
                self.phase, self.t_phase = "hold", obs.t
            else:
                v[2] = self.lift_speed
        elif self.phase == "hold" and obs.t - self.t_phase > self.hold_time:
            return Command(np.zeros(6), gripper_force=self.force, done=True, phase="done")
        if q < self.q_pause:  # close to slipping: stop the arm until the squeeze catches up
            v[:] = 0.0
        return Command(v, gripper_force=self.force, phase=self.phase)
