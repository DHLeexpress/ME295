"""Real-time calibrated RGB-D: every ZED frame carries the camera's pose in the robot base frame at the moment the
image was exposed.

    poses = RTDEPoses("192.168.0.2")                 # read-only RTDE (ur_rtde), sampled at 250 Hz in a thread
    with ZedCamera() as cam:
        stream = CalibratedStream(cam, poses, WristCamera.active())
        frame = stream.read()                        # RGBDFrame, T_base_cam at the image timestamp
        pts, rgb = frame.points_base()               # point cloud in the robot base frame

The robot pose is interpolated at the ZED image timestamp (host clock), so a frame taken while the arm moves is
placed where the camera was when it was exposed, not where it is when the depth is ready (NEURAL depth adds
~30-80 ms). meta: robot speed at exposure, pose skew to the nearest sample, frame age.

RTDE reports the TCP pose for the TCP set on the controller; tcp= must be that TCP (the hand-eye calibration was
solved at TCP = flange). check_tcp() compares the controller's pose with nominal forward kinematics (they agree to
~2 mm when the TCP really is the flange) to catch a wrong assumption.
"""
from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np

from .camera import RGBDFrame
from .safety import _rotvec, _slerp_R, fk_frames
from .wrist import WristCamera


def _pose_to_mat(p) -> np.ndarray:
    from shear_agent.robot import pose_to_mat
    return pose_to_mat(p)


class RTDEPoses:
    """Flange pose from the UR controller over RTDE, read-only, buffered for interpolation."""

    def __init__(self, host: str, tcp: Sequence[float] = (0, 0, 0, 0, 0, 0), rate: float = 250.0,
                 buffer_s: float = 5.0):
        import rtde_receive
        self.r = rtde_receive.RTDEReceiveInterface(host, rate)
        self.T_tcp_inv = np.linalg.inv(_pose_to_mat(tcp))
        self.rate = rate
        self.buf = deque(maxlen=int(rate * buffer_s))
        self.lock = threading.Lock()
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        t_end = time.time() + 2.0
        while not self.buf and time.time() < t_end:
            time.sleep(0.01)
        if not self.buf:
            raise RuntimeError(f"no RTDE data from {host}")

    def _run(self):
        period = 1.0 / self.rate
        while not self._stop:
            t0 = time.time()
            try:
                p = self.r.getActualTCPPose()
                q = self.r.getActualQ()
            except Exception:  # noqa: BLE001
                time.sleep(period)
                continue
            T = _pose_to_mat(p) @ self.T_tcp_inv
            with self.lock:
                self.buf.append((0.5 * (t0 + time.time()), T, np.asarray(q, float)))
            time.sleep(max(0.0, period - (time.time() - t0)))

    def latest(self) -> Tuple[float, np.ndarray, np.ndarray]:
        with self.lock:
            return self.buf[-1]

    def at(self, t: float) -> Tuple[np.ndarray, dict]:
        """Flange pose at host time t (linear / slerp between the bracketing samples)."""
        with self.lock:
            buf = list(self.buf)
        ts = [b[0] for b in buf]
        k = bisect.bisect_left(ts, t)
        if k == 0 or k == len(buf):
            j = 0 if k == 0 else len(buf) - 1
            return buf[j][1], {"skew_s": abs(ts[j] - t), "speed": self._speed(buf, j), "q": buf[j][2]}
        (t0, T0, q0), (t1, T1, q1) = buf[k - 1], buf[k]
        s = (t - t0) / max(t1 - t0, 1e-9)
        T = np.eye(4)
        T[:3, :3] = _slerp_R(T0[:3, :3], T1[:3, :3], s)
        T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
        return T, {"skew_s": min(t - t0, t1 - t), "speed": self._speed(buf, k), "q": (1 - s) * q0 + s * q1}

    @staticmethod
    def _speed(buf, j, half_window: int = 5):
        """TCP speed over ~10 samples (40 ms at 250 Hz): one 4 ms step is dominated by encoder jitter."""
        if len(buf) < 2:
            return 0.0
        a, b = buf[max(0, j - half_window)], buf[min(len(buf) - 1, j + half_window)]
        return float(np.linalg.norm(b[1][:3, 3] - a[1][:3, 3]) / max(b[0] - a[0], 1e-9))

    def check_tcp(self) -> Tuple[float, float]:
        """(mm, deg) between the controller's flange pose (given tcp=) and nominal FK of the joints. ~2 mm / 0.4 deg
        is the controller's kinematic calibration; much more means the TCP set on the robot is not tcp=."""
        _, T, q = self.latest()
        F = fk_frames(q)[-1]
        return 1e3 * float(np.linalg.norm(F[:3, 3] - T[:3, 3])), float(np.degrees(np.linalg.norm(_rotvec(F[:3, :3].T @ T[:3, :3]))))

    def close(self):
        self._stop = True
        self.thread.join(timeout=1.0)
        try:
            self.r.disconnect()
        except Exception:  # noqa: BLE001
            pass


class SimPoses:
    """The same interface on a SimRobot (frames and poses are simultaneous in the simulator)."""

    def __init__(self, rob):
        self.rob = rob

    def latest(self):
        return self.rob.time, self.rob.flange_pose(), np.asarray(self.rob.getj(), float)

    def at(self, t):
        return self.rob.flange_pose(), {"skew_s": 0.0, "speed": 0.0, "q": np.asarray(self.rob.getj(), float)}

    def check_tcp(self):
        return 0.0, 0.0

    def close(self):
        pass


class CalibratedStream:
    """Camera (anything with .grab() -> RGBDFrame whose stamp is host time) + pose source + hand-eye calibration."""

    def __init__(self, camera, poses, wrist: Optional[WristCamera] = None, latency: float = 0.0):
        self.camera, self.poses = camera, poses
        self.wrist = wrist or WristCamera.active()
        self.latency = latency          # extra exposure-to-timestamp delay, if ever measured (s)
        self.n = 0
        self.t_start = time.time()
        self._times = deque(maxlen=30)

    def read(self) -> RGBDFrame:
        f = self.camera.grab()
        t_img = f.stamp - self.latency
        T_flange, info = self.poses.at(t_img)
        f = f.with_pose(self.wrist.T_base_cam(T_flange))
        now = time.time()
        self._times.append(now)
        self.n += 1
        f.meta = {**(f.meta or {}), "T_base_flange": T_flange, "robot_speed": info["speed"], "pose_skew_s": info["skew_s"],
                  "q": info["q"], "frame_age_s": now - f.stamp if f.stamp > 1e9 else 0.0, "fps": self.fps}
        return f

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-9)
