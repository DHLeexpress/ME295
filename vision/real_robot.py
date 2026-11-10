"""The UR10e through ur_rtde with the urx-style calls the grasp loop uses (getj, getl, get_tcp_force, speedl, stopl,
movej, set_tcp, sleep), and a background grasp estimator on the live calibrated stream.

The controller's TCP is left at the flange (what the hand-eye calibration assumes and what RTDEPoses reads); set_tcp()
is a software TCP: getl() reports it and speedl() twists given for it are converted to the flange
(v_flange = v_tcp - w x r), so nothing is changed in the robot's installation.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Sequence

import numpy as np

from .camera import RGBDFrame
from .tabletop import TabletopParams, TopDownGrasp, plan_topdown_grasp, segment_objects


def _pm(p):
    from shear_agent.robot import pose_to_mat
    return pose_to_mat(p)


def _mp(T):
    from shear_agent.robot import mat_to_pose
    return mat_to_pose(T)


class RTDERobot:
    def __init__(self, host: str, control: bool = True):
        import rtde_receive
        self.rr = rtde_receive.RTDEReceiveInterface(host)
        self.rc = None
        if control:
            import rtde_control
            self.rc = rtde_control.RTDEControlInterface(host)
            off = np.asarray(self.rc.getTCPOffset(), float)
            if np.linalg.norm(off) > 1e-4:
                raise RuntimeError(f"the robot's TCP is {off.tolist()}, not the flange: set the TCP to zero on the "
                                   f"pendant (the hand-eye calibration and the stream assume the flange)")
        self.tcp = (0.0,) * 6
        self.T_tcp = np.eye(4)
        self.sleep = time.sleep

    def set_tcp(self, tcp: Sequence[float]):
        self.tcp = tuple(float(v) for v in tcp)
        self.T_tcp = _pm(self.tcp)

    def flange_pose(self) -> np.ndarray:
        return _pm(self.rr.getActualTCPPose())

    def getl(self):
        return [float(v) for v in _mp(self.flange_pose() @ self.T_tcp)]

    def getj(self):
        return list(self.rr.getActualQ())

    def get_tcp_force(self):
        return list(self.rr.getActualTCPForce())

    def speedl(self, v, acc=0.5, min_time=0.04):
        v = np.asarray(v, float)
        r = self.flange_pose()[:3, :3] @ self.T_tcp[:3, 3]
        vf = np.r_[v[:3] - np.cross(v[3:], r), v[3:]]
        self.rc.speedL(vf.tolist(), float(acc), float(min_time))

    def stopl(self, acc=0.5):
        self.rc.speedStop(float(acc))

    def movej(self, q, acc=0.3, vel=0.2, wait=True):
        self.rc.moveJ(list(map(float, q)), float(vel), float(acc), not wait)

    def is_program_running(self):
        return self.rc is not None and self.rc.isProgramRunning()

    def close(self):
        if self.rc is not None:
            try:
                self.rc.speedStop(1.0)
                self.rc.stopScript()
            finally:
                self.rc.disconnect()
        self.rr.disconnect()


class AsyncGraspEstimator:
    """Reads the calibrated stream continuously in a thread and keeps the newest grasp estimate (tracking the object
    nearest to the previous one). perceive() never blocks: it returns the newest estimate (stamp = image time)."""

    def __init__(self, stream, params: TabletopParams = TabletopParams(), yaw_hint: Callable[[], float] = lambda: None,
                 z_table: Optional[float] = None):
        self.stream, self.params, self.yaw_hint, self.z_table = stream, params, yaw_hint, z_table
        self.latest: Optional[TopDownGrasp] = None
        self.frame: Optional[RGBDFrame] = None
        self.n = 0
        self.prev = None
        self.error: Optional[str] = None
        self._stop = False
        self.t_last = time.time()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop:
            try:
                f = self.stream.read()
                objs, zt = segment_objects(f, f.meta.get("T_base_flange"), self.params)
                self.frame = f
                self.t_last = time.time()
                if not objs or zt is None:
                    continue
                o = min(objs, key=lambda o: np.linalg.norm(o.center[:2] - self.prev)) if self.prev is not None else objs[0]
                self.prev = o.center[:2].copy()
                self.latest = plan_topdown_grasp(o, self.z_table, prefer_yaw=self.yaw_hint(), stamp=f.stamp)
                self.n += 1
            except Exception as e:  # noqa: BLE001
                self.error = repr(e)
                time.sleep(0.05)

    def perceive(self) -> Optional[TopDownGrasp]:
        return self.latest

    def stale(self, max_age: float = 0.5) -> bool:
        return time.time() - self.t_last > max_age

    def close(self):
        self._stop = True
        self.th.join(timeout=2.0)
