"""The same task code on the real UR10e: python-urx for the arm, the Robotiq URCap socket for the gripper.

    from shear_agent.robot.real_ur import connect
    rob, gripper = connect("192.168.1.10")     # urx.Robot + RobotiqSocket (URCap, port 63352)

rob is a plain urx.Robot with one addition, rob.sleep = time.sleep, so loops written against
SimRobot (which advances simulated time in sleep()) run unchanged. RobotiqSocket offers the
methods of SimRobotiqGripper / urx's Robotiq_Two_Finger_Gripper plus non-blocking register
access. Not tested on hardware from this repository: before the first run, check the TCP,
payload and the sign convention of rob.get_tcp_force() on the teach pendant, and keep speeds low.
Needs `pip install urx` (python-urx; its math3d dependency may need pinning on new Pythons).
"""
from __future__ import annotations

import socket
import time


def connect(host: str, tcp=(0, 0, 0.292, 0, 0, 0), payload: float = 1.0, cog=(0, 0, 0.12),
            speed: int = 255, force: int = 0):
    import urx  # python-urx; only needed for the real robot

    rob = urx.Robot(host)
    rob.sleep = time.sleep
    rob.set_tcp(tcp)
    rob.set_payload(payload, cog)
    time.sleep(0.2)  # let the controller process the setup (python-urx README)
    gripper = RobotiqSocket(host, speed=speed, force=force)
    gripper.activate()
    return rob, gripper


class RobotiqSocket:
    """Robotiq 2F-85/140 through the Robotiq URCap socket server (port 63352).

    Registers: ACT activate, GTO go-to, POS/PRE requested/actual position, SPE speed, FOR force,
    OBJ object status (0 moving, 1 object while opening, 2 object while closing, 3 at position).
    """

    def __init__(self, host: str, port: int = 63352, timeout: float = 2.0, speed: int = 255, force: int = 0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.speed, self.force = int(speed), int(force)

    def _send(self, text: str) -> str:
        self.sock.sendall((text + "\n").encode())
        return self.sock.recv(1024).decode().strip()

    def _set(self, **registers):
        reply = self._send("SET " + " ".join(f"{k} {int(v)}" for k, v in registers.items()))
        if reply != "ack":
            raise RuntimeError(f"Robotiq refused {registers}: {reply!r}")

    def _get(self, register: str) -> int:
        reply = self._send(f"GET {register}")
        return int(reply.split()[-1])

    def activate(self, timeout: float = 5.0):
        if self._get("STA") != 3:
            self._set(ACT=0)
            self._set(ACT=1)
            t_end = time.time() + timeout
            while self._get("STA") != 3:
                if time.time() > t_end:
                    raise RuntimeError("Robotiq gripper did not activate")
                time.sleep(0.05)
        self._set(SPE=self.speed, FOR=self.force)

    # same methods as SimRobotiqGripper / Robotiq_Two_Finger_Gripper
    def set_position(self, value: int):
        self._set(POS=max(0, min(255, int(value))), SPE=self.speed, FOR=self.force, GTO=1)

    def set_force(self, force: int):
        self.force = max(0, min(255, int(force)))
        self._set(FOR=self.force)

    def set_speed(self, speed: int):
        self.speed = max(0, min(255, int(speed)))
        self._set(SPE=self.speed)

    def gripper_action(self, value: int, timeout: float = 2.0):
        self.set_position(value)
        t_end = time.time() + timeout
        time.sleep(0.05)
        while self.get_object_status() == 0 and time.time() < t_end:
            time.sleep(0.02)

    def open_gripper(self):
        self.gripper_action(0)

    def close_gripper(self):
        self.gripper_action(255)

    def get_position(self) -> int:
        return self._get("POS")

    def get_requested_position(self) -> int:
        return self._get("PRE")

    def get_object_status(self) -> int:
        return self._get("OBJ")

    def close(self):
        self.sock.close()
