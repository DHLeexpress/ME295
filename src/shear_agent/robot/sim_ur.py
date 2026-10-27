"""MuJoCo stand-in for python-urx: urx.Robot and Robotiq_Two_Finger_Gripper.

Same method names, arguments, units and pose convention as python-urx (base frame, metres,
rotation vectors), so task code written against it runs on the real UR10e by building the
real objects instead (see real_ur.connect):

    rob = SimRobot()                       # real: rob = urx.Robot("192.168.1.10")
    gripper = SimRobotiqGripper(rob)       # real: Robotiq_Two_Finger_Gripper(rob)
    rob.set_tcp((0, 0, 0.292, 0, 0, 0))
    rob.movej(q, acc=0.5, vel=0.3)         # joints (rad), rad/s^2, rad/s
    rob.movel(pose, acc=0.1, vel=0.05)     # pose = [x, y, z, rx, ry, rz], m/s^2, m/s
    gripper.gripper_action(200)            # 0 open .. 255 closed
    print(rob.getj(), rob.getl(), rob.get_tcp_force())

The arm kinematics are the UR10e DH parameters (assembly/ur10e_2f140.xml), so getl()/movel()
poses equal the real controller's for the same TCP. Simulated time only advances inside the
robot's calls: blocking moves, gripper actions and rob.sleep(dt). A non-blocking move
(wait=False) runs while you call rob.sleep(dt); on the real robot that is time.sleep(dt).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless Linux; macOS/Windows keep their default
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE = ROOT / "assembly" / "scene_bookshelf.xml"
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
HOME_Q = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
GRAVITY = 9.81


class RobotException(Exception):
    """Same role as urx.RobotException: a move cannot be executed."""


# ---------------------------------------------------------------- rotations (rotation vectors)
def rotvec_to_mat(r) -> np.ndarray:
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rotvec(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def pose_to_mat(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotvec_to_mat(pose[3:6])
    T[:3, 3] = pose[:3]
    return T


def mat_to_pose(T) -> np.ndarray:
    return np.r_[T[:3, 3], mat_to_rotvec(T[:3, :3])]


def _trapezoid(length: float, vel: float, acc: float):
    """Duration and s(t) of a trapezoidal (or triangular) profile covering `length`."""
    length, vel, acc = abs(length), max(abs(vel), 1e-9), max(abs(acc), 1e-9)
    if length < 1e-12:
        return 0.0, lambda t: 0.0
    t_acc = vel / acc
    if acc * t_acc ** 2 > length:  # triangular: never reaches vel
        t_acc = np.sqrt(length / acc)
        vel = acc * t_acc
    t_flat = (length - acc * t_acc ** 2) / vel
    total = 2 * t_acc + t_flat

    def s(t):
        t = min(max(t, 0.0), total)
        if t < t_acc:
            return 0.5 * acc * t * t
        if t < t_acc + t_flat:
            return 0.5 * acc * t_acc ** 2 + vel * (t - t_acc)
        td = total - t
        return length - 0.5 * acc * td * td
    return total, s


# ---------------------------------------------------------------- arm
class SimRobot:
    """urx.Robot-compatible UR10e in MuJoCo (methods, arguments and units as in python-urx)."""

    def __init__(self, scene: str | Path = DEFAULT_SCENE, control_hz: float = 500.0, verbose: bool = True,
                 init_q: Optional[Sequence[float]] = None, model: Optional[mujoco.MjModel] = None):
        self.m = model if model is not None else mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self._ik_data = mujoco.MjData(self.m)
        self.dt = 1.0 / control_hz
        self.nsub = max(1, int(round(self.dt / self.m.opt.timestep)))
        self.dt = self.nsub * self.m.opt.timestep
        self.verbose = verbose
        self._qadr = np.array([self.m.joint(n).qposadr[0] for n in ARM_JOINTS])
        self._dadr = np.array([self.m.joint(n).dofadr[0] for n in ARM_JOINTS])
        self._act = np.array([self.m.actuator(n).id for n in ARM_ACTUATORS])
        kp = self.m.actuator_gainprm[self._act, 0]
        kv = -self.m.actuator_biasprm[self._act, 2]
        self._ff = kv / kp  # velocity feed-forward: ctrl = q_ref + (kv/kp) qd_ref -> PD tracking
        self._flange = self.m.site("attachment_site").id
        self._flange_body = self.m.body("flange").id
        self.tcp = np.zeros(6)
        self._T_tcp = np.eye(4)
        # payload = everything on the flange (gripper, blades, camera) unless set_payload() says otherwise
        mujoco.mj_forward(self.m, self.d)
        self.payload_mass = float(self.m.body_subtreemass[self._flange_body])
        self.payload_cog = self._subtree_cog_in_flange()
        self._jac = np.zeros((3, self.m.nv))
        self.gripper: Optional["SimRobotiqGripper"] = None
        self.callbacks: List[Callable[["SimRobot"], None]] = []  # called after every control tick
        self.command_log: List[tuple] = []  # (time, text): urx-style record of every command
        self._motion: Optional[Iterator] = None
        self.reset(init_q)

    # ------------------------------------------------------------ state
    def reset(self, q: Optional[Sequence[float]] = None):
        try:
            mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        except KeyError:
            mujoco.mj_resetData(self.m, self.d)
        if q is not None:
            self.d.qpos[self._qadr] = q
        self._q_ref = self.d.qpos[self._qadr].copy()
        self._qd_ref = np.zeros(6)
        self._v_tcp = np.zeros(6)
        self.d.ctrl[self._act] = self._q_ref
        self._motion = None
        mujoco.mj_forward(self.m, self.d)

    @property
    def time(self) -> float:
        return float(self.d.time)

    def _say(self, text: str):
        if self.command_log and self.command_log[-1][1] == text:
            return  # a controller streaming the same command: log it once
        self.command_log.append((self.time, text))
        if self.verbose:
            print(f"[{self.time:8.3f} s] {text}")

    def _subtree_cog_in_flange(self) -> np.ndarray:
        b = self._flange_body
        R = self.d.site_xmat[self._flange].reshape(3, 3)
        return R.T @ (self.d.subtree_com[b] - self.d.site_xpos[self._flange])

    def flange_pose(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self._flange].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self._flange]
        return T

    # ------------------------------------------------------------ time
    def _tick(self):
        if self._motion is not None:
            try:
                self._q_ref, self._qd_ref = next(self._motion)
            except StopIteration:
                self._motion = None
                self._qd_ref = np.zeros(6)
        self.d.ctrl[self._act] = self._q_ref + self._ff * self._qd_ref
        self._payload_feedforward()
        if self.gripper is not None:
            self.gripper._tick(self.dt)
        for _ in range(self.nsub):
            mujoco.mj_step(self.m, self.d)
        for cb in self.callbacks:
            cb(self)

    def _payload_feedforward(self):
        """Hold the declared payload's weight with joint torques, as the UR controller does from
        set_payload (an undeclared load, e.g. a grasped book, still makes the arm sag a little)."""
        R = self.d.site_xmat[self._flange].reshape(3, 3)
        p = self.d.site_xpos[self._flange] + R @ self.payload_cog
        mujoco.mj_jac(self.m, self.d, self._jac, None, p, self._flange_body)
        self.d.qfrc_applied[self._dadr] = self._jac[:, self._dadr].T @ np.array([0, 0, GRAVITY * self.payload_mass])

    def sleep(self, seconds: float):
        """Advance the simulation by `seconds` (the real robot: time.sleep)."""
        for _ in range(max(1, int(round(seconds / self.dt)))):
            self._tick()

    def _wait(self, settle: float = 0.05, timeout: float = 60.0):
        t_end = self.time + timeout
        while self._motion is not None:
            self._tick()
            if self.time > t_end:
                raise RobotException("motion timed out")
        t_still = 0.0
        while t_still < settle and self.time < t_end:  # let the servos settle like the real arm
            self._tick()
            t_still = t_still + self.dt if np.max(np.abs(self.d.qvel[self._dadr])) < 2e-3 else 0.0

    # ------------------------------------------------------------ kinematics
    def _fk_flange(self, q) -> np.ndarray:
        d = self._ik_data
        d.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.m, d)
        T = np.eye(4)
        T[:3, :3] = d.site_xmat[self._flange].reshape(3, 3)
        T[:3, 3] = d.site_xpos[self._flange]
        return T

    def ik(self, T_flange: np.ndarray, q_seed: Sequence[float], iters: int = 50, tol: float = 1e-7) -> tuple:
        """Damped least-squares IK for the flange pose; returns (q, error)."""
        d, m = self._ik_data, self.m
        q = np.array(q_seed, float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err = np.inf
        for _ in range(iters):
            d.qpos[self._qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[self._flange].reshape(3, 3)
            e_p = T_flange[:3, 3] - d.site_xpos[self._flange]
            e_r = R @ mat_to_rotvec(R.T @ T_flange[:3, :3])
            e = np.r_[e_p, e_r]
            err = np.linalg.norm(e)
            if err < tol:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self._flange)
            J = np.vstack([jp[:, self._dadr], jr[:, self._dadr]])
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        seed = np.asarray(q_seed, float)
        q = seed + (q - seed + np.pi) % (2 * np.pi) - np.pi  # same pose, joint branch nearest the seed
        return q, err

    def _tcp_to_flange(self, T_tcp_world: np.ndarray) -> np.ndarray:
        return T_tcp_world @ np.linalg.inv(self._T_tcp)

    # ------------------------------------------------------------ urx API: state
    def getj(self, wait: bool = False) -> List[float]:
        return [float(v) for v in self.d.qpos[self._qadr]]

    def get_pose(self, wait: bool = False, _log: bool = True) -> np.ndarray:
        """TCP pose as a 4x4 matrix (urx returns a math3d.Transform)."""
        return self.flange_pose() @ self._T_tcp

    def getl(self, wait: bool = False, _log: bool = True) -> List[float]:
        return [float(v) for v in mat_to_pose(self.get_pose())]

    def get_tcp_force(self, wait: bool = True) -> List[float]:
        """External wrench on the tool [Fx, Fy, Fz, Tx, Ty, Tz] (N, N m): base frame, moments about
        the TCP, payload (set_payload) compensated, as the UR10e's built-in F/T sensor reports it
        (check the sign convention against the real controller once)."""
        R = self.d.site_xmat[self._flange].reshape(3, 3)
        f_sensor = R @ self.d.sensor("flange_force").data      # wrist -> tool, world frame
        t_sensor = R @ self.d.sensor("flange_torque").data
        w_payload = np.array([0.0, 0.0, -GRAVITY * self.payload_mass])
        r_cog = R @ self.payload_cog
        f_ext = -f_sensor - w_payload
        t_ext_flange = -t_sensor - np.cross(r_cog, w_payload)
        r_tcp = R @ self._T_tcp[:3, 3]
        t_ext_tcp = t_ext_flange - np.cross(r_tcp, f_ext)
        return [float(v) for v in np.r_[f_ext, t_ext_tcp]]

    def get_force(self, wait: bool = True) -> float:
        """Norm of get_tcp_force() over all six components, exactly as python-urx computes it."""
        return float(np.linalg.norm(self.get_tcp_force(wait)))

    def is_running(self) -> bool:
        return True

    def is_program_running(self) -> bool:
        return self._motion is not None

    # ------------------------------------------------------------ urx API: setup
    def set_tcp(self, tcp):
        self.tcp = np.asarray(tcp, float)
        self._T_tcp = pose_to_mat(self.tcp)
        self._say(f"rob.set_tcp({_fmt(self.tcp)})")

    def set_payload(self, weight: float, cog=None):
        self.payload_mass = float(weight)
        if cog is not None:
            self.payload_cog = np.asarray(cog, float)
        self._say(f"rob.set_payload({weight:.3f}, {_fmt(self.payload_cog)})")

    # ------------------------------------------------------------ urx API: motion
    def movej(self, joints, acc: float = 0.1, vel: float = 0.05, wait: bool = True, relative: bool = False,
              threshold=None):
        q0 = self._q_ref.copy()
        q1 = np.asarray(joints, float) + (q0 if relative else 0.0)
        self._say(f"rob.movej({_fmt(q1)}, acc={acc:g}, vel={vel:g}{', relative=True' if relative else ''})")
        dq = q1 - q0
        self._v_tcp = np.zeros(6)
        total, s = _trapezoid(np.max(np.abs(dq)), vel, acc)  # leading joint sets the timing
        lead = np.max(np.abs(dq)) or 1.0

        def motion():
            t, prev = 0.0, q0
            while t < total:
                t += self.dt
                q = q0 + dq * (s(t) / lead)
                yield q, (q - prev) / self.dt
                prev = q
            yield q1, np.zeros(6)
        self._motion = motion()
        if wait:
            self._wait()

    def movel(self, tpose, acc: float = 0.01, vel: float = 0.01, wait: bool = True, relative: bool = False,
              threshold=None):
        current = np.asarray(self.getl(), float)
        target = np.asarray(tpose, float) + (current if relative else 0.0)  # urx adds element-wise
        self._say(f"rob.movel({_fmt(target)}, acc={acc:g}, vel={vel:g}{', relative=True' if relative else ''})")
        T0 = self._fk_flange(self._q_ref) @ self._T_tcp
        T1 = pose_to_mat(target)
        q_end, err = self.ik(self._tcp_to_flange(T1), self._q_ref)
        if err > 1e-5:
            raise RobotException(f"movel target {_fmt(target)} is out of reach (IK error {err:.2e})")
        dp = T1[:3, 3] - T0[:3, 3]
        dr = mat_to_rotvec(T0[:3, :3].T @ T1[:3, :3])  # tool-frame rotation
        length = np.linalg.norm(dp)
        if length < 1e-6:  # pure rotation: acc/vel act as rad/s^2, rad/s
            length = np.linalg.norm(dr)
        total, s = _trapezoid(length, vel, acc)
        self._v_tcp = np.zeros(6)
        self._motion = self._cartesian(lambda t: _interp(T0, dp, dr, s(t) / length if length else 1.0), total)
        if wait:
            self._wait()

    def _cartesian(self, pose_at: Callable[[float], np.ndarray], duration: float):
        q, t = self._q_ref.copy(), 0.0
        while t < duration:
            t += self.dt
            q_new, _ = self.ik(self._tcp_to_flange(pose_at(t)), q, iters=8)
            yield q_new, (q_new - q) / self.dt
            q = q_new
        yield q, np.zeros(6)

    def translate(self, vect, acc: float = 0.01, vel: float = 0.01, wait: bool = True, command: str = "movel"):
        pose = np.asarray(self.getl(), float)
        pose[:3] += np.asarray(vect, float)
        self.movel(pose, acc=acc, vel=vel, wait=wait)

    def translate_tool(self, vect, acc: float = 0.01, vel: float = 0.01, wait: bool = True):
        T = self.get_pose()
        pose = np.asarray(self.getl(), float)
        pose[:3] += T[:3, :3] @ np.asarray(vect, float)
        self.movel(pose, acc=acc, vel=vel, wait=wait)

    def up(self, z: float = 0.05, acc: float = 0.01, vel: float = 0.01):
        self.translate((0, 0, z), acc, vel)

    def down(self, z: float = 0.05, acc: float = 0.01, vel: float = 0.01):
        self.translate((0, 0, -z), acc, vel)

    def speedl(self, velocities, acc: float, min_time: float):
        """TCP velocity [vx, vy, vz, wx, wy, wz] (base frame) for min_time, then decelerate (URScript speedl)."""
        v_goal = np.asarray(velocities, float)
        self._say(f"rob.speedl({_fmt(v_goal)}, acc={acc:g}, min_time={min_time:g})")
        T = self._fk_flange(self._q_ref) @ self._T_tcp
        v_now = self._v_tcp

        def motion():
            nonlocal T, v_now
            q, t = self._q_ref.copy(), 0.0
            while True:
                goal = v_goal if t < min_time else np.zeros(6)
                dv = goal - v_now
                step = acc * self.dt
                v_now = v_now + (dv if np.linalg.norm(dv) <= step else dv / np.linalg.norm(dv) * step)
                self._v_tcp = v_now
                if t >= min_time and np.linalg.norm(v_now) < 1e-9:
                    break
                T = T.copy()
                T[:3, 3] += v_now[:3] * self.dt
                T[:3, :3] = rotvec_to_mat(v_now[3:] * self.dt) @ T[:3, :3]
                q_new, _ = self.ik(self._tcp_to_flange(T), q, iters=8)
                yield q_new, (q_new - q) / self.dt
                q, t = q_new, t + self.dt
            yield q, np.zeros(6)
        self._motion = motion()

    def speedj(self, velocities, acc: float, min_time: float):
        qd_goal = np.asarray(velocities, float)
        self._say(f"rob.speedj({_fmt(qd_goal)}, acc={acc:g}, min_time={min_time:g})")

        def motion():
            q, qd, t = self._q_ref.copy(), self._qd_ref.copy(), 0.0
            while True:
                goal = qd_goal if t < min_time else np.zeros(6)
                qd = qd + np.clip(goal - qd, -acc * self.dt, acc * self.dt)
                if t >= min_time and np.max(np.abs(qd)) < 1e-9:
                    break
                q = q + qd * self.dt
                yield q, qd
                t += self.dt
            yield q, np.zeros(6)
        self._motion = motion()

    def stopl(self, acc: float = 0.5):
        self._say(f"rob.stopl(acc={acc:g})")
        self._stop(acc)

    def stopj(self, acc: float = 1.5):
        self._say(f"rob.stopj(acc={acc:g})")
        self._stop(acc)

    def stop(self):
        self.stopj()

    def _stop(self, acc: float):
        def motion():
            q, qd = self._q_ref.copy(), self._qd_ref.copy()
            while np.max(np.abs(qd)) > 1e-9:
                qd = qd - np.clip(qd, -acc * self.dt, acc * self.dt)
                q = q + qd * self.dt
                yield q, qd
            yield q, np.zeros(6)
        self._v_tcp = np.zeros(6)
        self._motion = motion()

    def close(self):
        self._say("rob.close()")


def _interp(T0, dp, dr, a):
    T = T0.copy()
    T[:3, 3] = T0[:3, 3] + a * dp
    T[:3, :3] = T0[:3, :3] @ rotvec_to_mat(a * dr)
    return T


def _fmt(v) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in np.asarray(v, float)) + "]"


# ---------------------------------------------------------------- gripper
class SimRobotiqGripper:
    """Robotiq_Two_Finger_Gripper-compatible 2F-140 (python-urx: urx.robotiq_two_finger_gripper).

    Registers as on the real gripper: position 0 (open) .. 255 (closed), speed and force 0..255.
    Speed maps to 30..250 mm/s and force to 10..125 N per finger (2F-140 data sheet).
    """

    Q_CLOSED = 0.7  # driver knuckle angle (rad) at position 255

    def __init__(self, robot: SimRobot, payload: float = 0.85, speed: int = 255, force: int = 50,
                 socket_host: str = "127.0.0.1", socket_port: int = 63352, socket_name: str = "gripper_socket"):
        self.robot = robot
        robot.gripper = self
        m = robot.m
        self._act = m.actuator("fingers").id
        self._jnt = robot.d.joint("left_driver_joint")  # live data view (qpos, qvel)
        self._target = float(robot.d.ctrl[self._act])
        self._ramp = self._target
        self.speed, self.force = int(speed), int(force)
        self._apply_force()

    # registers
    def _apply_force(self):
        newton = 10.0 + 115.0 * np.clip(self.force, 0, 255) / 255.0
        torque = newton * 24.0 / 124.0  # calibrated: 24 N m on the driver ~ 124 N per finger
        self.robot.m.actuator_forcerange[self._act] = [-torque, torque]

    @property
    def jaw_speed(self) -> float:
        return (30.0 + 220.0 * np.clip(self.speed, 0, 255) / 255.0) / 1000.0  # m/s

    def _tick(self, dt: float):
        rate = 255.0 * self.jaw_speed / 0.140
        self._ramp += float(np.clip(self._target - self._ramp, -rate * dt, rate * dt))
        self.robot.d.ctrl[self._act] = self._ramp

    # urx API
    def gripper_action(self, value: int, timeout: float = 2.0):
        """Move to `value` (0 open .. 255 closed) and wait until the fingers stop (target reached
        or object detected). python-urx sends the same command and sleeps a fixed 2 s."""
        self.set_position(value, _log=False)
        self.robot._say(f"gripper.gripper_action({int(value)})")
        t_end = self.robot.time + timeout
        still = 0.0
        while self.robot.time < t_end:
            self.robot._tick()
            moving = abs(self._jnt.qvel[0]) > 2e-3 or abs(self._ramp - self._target) > 0.5
            still = 0.0 if moving else still + self.robot.dt
            if still > 0.05:
                break

    def open_gripper(self):
        self.gripper_action(0)

    def close_gripper(self):
        self.gripper_action(255)

    # Robotiq register access (real gripper: socket 63352, see real_ur.RobotiqSocket)
    def set_position(self, value: int, _log: bool = True):
        """Non-blocking position command (rq_move / SET POS)."""
        self._target = float(np.clip(value, 0, 255))
        if _log:
            self.robot._say(f"gripper.set_position({int(value)})")

    def set_force(self, force: int):
        self.force = int(force)
        self._apply_force()
        self.robot._say(f"gripper.set_force({self.force})")

    def set_speed(self, speed: int):
        self.speed = int(speed)
        self.robot._say(f"gripper.set_speed({self.speed})")

    def get_position(self) -> int:
        """Actual finger position register gPO (0..255)."""
        return int(round(np.clip(self._jnt.qpos[0] / self.Q_CLOSED * 255.0, 0, 255)))

    def get_requested_position(self) -> int:
        return int(round(self._target))

    def get_object_status(self) -> int:
        """gOBJ: 0 moving, 1 stopped on an object while opening, 2 ... while closing, 3 at position."""
        if abs(self._jnt.qvel[0]) > 2e-3 or abs(self._ramp - self._target) > 0.5:
            return 0
        if abs(self.get_position() - self._target) <= 3:
            return 3
        return 2 if self._target > self.get_position() else 1
