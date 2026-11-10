"""Top-down grasp controllers on ME295's controller interface (shear_agent.control: Observation -> Command, speedl +
Robotiq registers), and the loop that runs them with perception and the safety filter in the loop.

Every controller shares: the grasp estimate (vision.tabletop), the orientation control (P on the rotation error,
capped), the gripper sequence (open to the planned opening while approaching; close at a fixed force register;
lift 10 cm; hold 1 s) and the safety filter on every twist. They differ only in approach(): how the TCP gets from the
view configuration to the grasp pose. That is the ablation.
"""
from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from shear_agent.control.base import Command, Controller, Observation, apply, observe

from ..tabletop import TopDownGrasp


def rotvec(R: np.ndarray) -> np.ndarray:
    from ..safety import _rotvec
    return _rotvec(R)


def rotvec_to_mat(r) -> np.ndarray:
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def tool_yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


@dataclass
class GraspObservation(Observation):
    grasp: Optional[TopDownGrasp] = None      # latest accepted estimate (None: nothing seen yet)
    T_base_flange: Optional[np.ndarray] = None
    barrier: bool = False                     # the safety barrier changed the previous command


class GoalFilter:
    """Accepts new grasp estimates: exponential smoothing of position and jaw yaw, an occlusion check (approaching an
    object adds points and keeps its size; the fingers hiding it removes points and shrinks it: an estimate with
    fewer than `min_points_frac` of the most points seen, or a footprint / height off by more than `size_tol`, is
    rejected, and two rejections in a row freeze the goal), an outlier gate (a jump larger than `gate` is accepted
    only when the next estimate agrees), and a freeze below `freeze_height` above the grasp."""

    def __init__(self, alpha: float = 0.5, gate: float = 0.04, freeze_height: float = 0.08,
                 min_points_frac: float = 0.6, size_tol: float = 0.25):
        self.alpha, self.gate, self.freeze_height = alpha, gate, freeze_height
        self.min_points_frac, self.size_tol = min_points_frac, size_tol
        self.goal: Optional[TopDownGrasp] = None
        self.pending: Optional[np.ndarray] = None
        self.stamp = -np.inf
        self.frozen = False
        self.updates = 0
        self.rejected = 0
        self.n_max = 0
        self.ref = None          # (width, length, height) of the accepted object

    def _consistent(self, est: TopDownGrasp) -> bool:
        o = est.obj
        if o is None or self.ref is None:
            return True
        if o.n < self.min_points_frac * self.n_max:
            return False
        w, L, h = self.ref
        tol = lambda a, b: abs(a - b) <= max(0.012, self.size_tol * b)
        return tol(o.width, w) and tol(o.length, L) and tol(o.height, h)

    def update(self, est: Optional[TopDownGrasp], tcp_z: float) -> Optional[TopDownGrasp]:
        if est is None or est.stamp <= self.stamp:
            return self.goal
        self.stamp = est.stamp
        if self.goal is None:
            if not est.ok:
                return None
            self.goal = est
            if est.obj is not None:
                self.n_max, self.ref = est.obj.n, (est.obj.width, est.obj.length, est.obj.height)
            return self.goal
        if self.frozen or tcp_z < self.goal.position[2] + self.freeze_height:
            self.frozen = True
            return self.goal
        if not est.ok or not self._consistent(est):
            self.rejected += 1
            self.consecutive = getattr(self, "consecutive", 0) + 1
            if self.consecutive >= 2:
                self.frozen = True
            return self.goal
        self.consecutive = 0
        if est.obj is not None:
            self.n_max = max(self.n_max, est.obj.n)
        jump = np.linalg.norm(est.position[:2] - self.goal.position[:2])
        if jump > self.gate:
            if self.pending is None or np.linalg.norm(est.position[:2] - self.pending) > self.gate / 2:
                self.pending = est.position[:2].copy()
                return self.goal
        self.pending = None
        a = self.alpha
        g = self.goal
        dyaw = (est.yaw - g.yaw + np.pi / 2) % np.pi - np.pi / 2          # jaw axis is undirected
        yaw = g.yaw + a * dyaw
        pos = (1 - a) * g.position + a * est.position
        from ..tabletop import tcp_pose
        T = tcp_pose(pos, yaw)
        Tp = T.copy()
        Tp[2, 3] = g.T_pre[2, 3] - g.position[2] + pos[2]
        self.goal = TopDownGrasp(position=pos, yaw=yaw, width=(1 - a) * g.width + a * est.width,
                                 opening=max(g.opening, est.opening), T_tcp=T, T_pre=Tp, ok=True, obj=est.obj,
                                 stamp=est.stamp)
        self.updates += 1
        return self.goal


class TopDownGraspController(Controller):
    rate_hz = 50.0
    name = "top-down grasp"
    closed_loop = False

    def __init__(self, register_for_width: Callable[[float], int], force: int = 80, lift: float = 0.10,
                 hold: float = 1.0, v_lift: float = 0.05, k_rot: float = 2.0, w_max: float = 0.5,
                 close_timeout: float = 3.0, reach_tol: float = 0.0025, blocked_close: float = 0.3,
                 blocked_abort: float = 2.0):
        self.register_for_width = register_for_width
        self.force, self.lift, self.hold, self.v_lift = force, lift, hold, v_lift
        self.k_rot, self.w_max, self.close_timeout, self.reach_tol = k_rot, w_max, close_timeout, reach_tol
        self.blocked_close, self.blocked_abort = blocked_close, blocked_abort
        self.filter = GoalFilter()

    # ------------------------------------------------------------------ interface
    def reset(self, obs: GraspObservation) -> None:
        self.phase, self.t_phase = "approach", obs.t
        self.filter = GoalFilter()
        self.goal = self.filter.update(obs.grasp, obs.tcp_pose[2])
        self.open_reg = self.register_for_width(self.goal.opening) if self.goal else 0
        self.z0 = None
        self.result = "running"
        self.t_blocked = None
        self.reset_approach(obs)

    def reset_approach(self, obs) -> None:  # noqa: B027
        pass

    @abstractmethod
    def approach(self, obs: GraspObservation, goal: TopDownGrasp) -> tuple:
        """-> (linear velocity (3,), reached: bool)"""

    def wants_perception(self, obs: GraspObservation) -> bool:
        return self.closed_loop and self.phase == "approach" and not self.filter.frozen

    def rot_velocity(self, obs: GraspObservation, R_goal: np.ndarray) -> np.ndarray:
        R = rotvec_to_mat(obs.tcp_pose[3:])
        w = self.k_rot * rotvec(R_goal @ R.T)
        n = np.linalg.norm(w)
        return w * (self.w_max / n) if n > self.w_max else w

    def act(self, obs: GraspObservation) -> Command:
        z = np.zeros(6)
        if self.phase == "approach":
            if self.closed_loop:
                self.goal = self.filter.update(obs.grasp, obs.tcp_pose[2])
            if self.goal is None:
                return Command(tcp_velocity=z, phase="wait for a grasp")
            if self.open_reg == 0:
                self.open_reg = self.register_for_width(self.goal.opening)
            v, reached = self.approach(obs, self.goal)
            w = self.rot_velocity(obs, self.goal.T_tcp[:3, :3])
            # the safety barrier holds the gripper above the table: if it does so with the jaws over the object,
            # this is as low as is safe -> grasp here; blocked elsewhere for long -> give up
            if obs.barrier:
                self.t_blocked = obs.t if self.t_blocked is None else self.t_blocked
                e = self.goal.position - obs.tcp_pose[:3]
                if obs.t - self.t_blocked >= self.blocked_close and np.linalg.norm(e[:2]) < 0.005 and abs(e[2]) < 0.03:
                    reached = True
                elif obs.t - self.t_blocked >= self.blocked_abort:
                    self.result = "blocked by the safety filter"
                    return Command(tcp_velocity=z, done=True, phase="blocked")
            else:
                self.t_blocked = None
            if reached:
                self.phase, self.t_phase = "close", obs.t
                return Command(tcp_velocity=z, gripper_position=255, gripper_force=self.force, phase="close")
            return Command(tcp_velocity=np.r_[v, w], gripper_position=self.open_reg, phase="approach")
        if self.phase == "close":
            dt = obs.t - self.t_phase
            if obs.gripper_object in (1, 2) and dt > 0.2:
                self.phase, self.t_phase, self.z0 = "lift", obs.t, float(obs.tcp_pose[2])
            elif obs.gripper_object == 3 and dt > 0.5:
                self.result = "missed (jaws closed on nothing)"
                return Command(tcp_velocity=z, done=True, phase="missed")
            elif dt > self.close_timeout:
                self.phase, self.t_phase, self.z0 = "lift", obs.t, float(obs.tcp_pose[2])
            return Command(tcp_velocity=z, gripper_position=255, gripper_force=self.force, phase="close")
        if self.phase == "lift":
            if obs.tcp_pose[2] >= self.z0 + self.lift:
                self.phase, self.t_phase = "hold", obs.t
            return Command(tcp_velocity=np.r_[0, 0, self.v_lift, 0, 0, 0], phase="lift")
        if self.phase == "hold":
            if obs.t - self.t_phase >= self.hold:
                self.result = "done"
                return Command(tcp_velocity=z, done=True, phase="done")
            return Command(tcp_velocity=z, phase="hold")
        return Command(tcp_velocity=z, done=True, phase=self.phase)


def run_grasp(controller: TopDownGraspController, rob, gripper, perceive: Callable[[], Optional[TopDownGrasp]],
              safety, tcp_for_width: Callable[[float], tuple], flange_pose: Callable[[], np.ndarray],
              perception_hz: float = 5.0, max_time: float = 30.0, events: Optional[List[tuple]] = None,
              on_step: Optional[Callable] = None, clock: Callable[[], float] = None, say=print) -> dict:
    """Look, set the TCP between the pads at the object's width, then close the loop at controller.rate_hz:
    perception at perception_hz while the controller wants it, every twist through safety.filter_twist().
    events: [(t_after_start, fn)] run once (e.g. push the object in the simulator)."""
    clock = clock or (lambda: float(rob.time) if hasattr(rob, "time") else time.monotonic())
    period = 1.0 / controller.rate_hz
    t0 = clock()
    g = perceive()
    if g is None:
        return {"status": "nothing seen", "log": [], "perceptions": 1}
    if not g.ok:
        return {"status": f"not graspable: {g.reason}", "log": [], "perceptions": 1}
    rob.set_tcp(tcp_for_width(g.width))
    say(f"[{controller.name}] {g.summary()}")

    state = {"barrier": False}

    def obs_now(grasp):
        o = observe(rob, gripper)
        return GraspObservation(**o.__dict__, grasp=grasp, T_base_flange=flange_pose(), barrier=state["barrier"])
    obs = obs_now(g)
    controller.reset(obs)
    events = sorted(events or [], key=lambda e: e[0])
    t_last, n_perc, n_mod = clock(), 1, 0
    log = []
    while True:
        t = clock()
        while events and t - t0 >= events[0][0]:
            events.pop(0)[1]()
        if controller.wants_perception(obs) and t - t_last >= 1.0 / perception_hz:
            t_last = t
            n_perc += 1
            est = perceive()
            if est is not None:
                g = est
        obs = obs_now(g)
        cmd = controller.act(obs)
        if cmd.done or t - t0 > max_time:
            rob.speedl([0, 0, 0, 0, 0, 0], acc=0.5, min_time=period)
            break
        if cmd.tcp_velocity is not None:
            v, mod = safety.filter_twist(obs.T_base_flange, obs.tcp_pose[:3], cmd.tcp_velocity)
            n_mod += int(mod)
            state["barrier"] = mod
            cmd.tcp_velocity = v
        apply(rob, gripper, cmd, period)
        rob.sleep(period)
        log.append((t - t0, cmd.phase, *obs.tcp_pose[:3]))
        if on_step:
            on_step(obs, cmd)
    status = controller.result if t - t0 <= max_time else "timeout"
    return {"status": status, "time": clock() - t0, "log": log, "perceptions": n_perc,
            "filter_updates": controller.filter.updates, "safety_interventions": n_mod,
            "speed_caps": getattr(safety, "speed_caps", 0),
            "goal": None if controller.goal is None else controller.goal.position.tolist()}
