"""Controller interface for the contact-rich phase of a task.

A controller maps an Observation to a Command at a fixed rate. Commands use only what the
real UR10e + Robotiq accept, so the same loop runs in MuJoCo and on the robot:

    tcp_velocity      -> rob.speedl(v, acc, min_time)            (urx / URScript)
    gripper_position  -> gripper.set_position(p)                  (Robotiq SET POS)
    gripper_force     -> gripper.set_force(f)                     (Robotiq SET FOR)

The rule-based ShearMarginGrasp (margin.py) is the placeholder; MPC, MPPI and learned
(vision-language-force-action) policies plug in by implementing Controller.act().
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class Observation:
    t: float
    q: np.ndarray                    # rob.getj()               joint angles (rad)
    tcp_pose: np.ndarray             # rob.getl()               [x, y, z, rx, ry, rz] base frame
    tcp_force: np.ndarray            # rob.get_tcp_force()      external wrench on the tool (N, N m)
    gripper_position: int            # Robotiq gPO              0 open .. 255 closed
    gripper_request: int             # Robotiq gPR
    gripper_object: int              # Robotiq gOBJ             0 moving, 1/2 object, 3 at position
    gripper_force: int               # Robotiq force register   0..255
    tactile: Optional[Dict[str, Any]] = None       # {"left"/"right": TactileFrame} shear-sensing skin
    images: Optional[Dict[str, np.ndarray]] = None  # e.g. {"wrist_rgb": HxWx3}
    privileged: Dict[str, Any] = field(default_factory=dict)  # sim-only ground truth; empty on the robot


@dataclass
class Command:
    tcp_velocity: Optional[np.ndarray] = None   # [vx, vy, vz, wx, wy, wz] (m/s, rad/s), base frame
    gripper_position: Optional[int] = None
    gripper_force: Optional[int] = None
    done: bool = False
    phase: str = ""


class Controller(ABC):
    """Observation -> Command at `rate_hz`."""

    rate_hz: float = 50.0
    name: str = "controller"

    def reset(self, obs: Observation) -> None:  # noqa: B027  (optional hook)
        pass

    @abstractmethod
    def act(self, obs: Observation) -> Command:
        ...


def observe(rob, gripper, skins=None, images: Optional[Callable[[], Dict[str, np.ndarray]]] = None,
            privileged: Optional[Callable[[], Dict[str, Any]]] = None) -> Observation:
    """Build an Observation from urx-compatible robot/gripper objects (+ optional skin/camera)."""
    tactile = None
    if skins:
        tactile = {side: skin.read(rob.d) for side, skin in skins.items()}
    return Observation(
        t=float(rob.time) if hasattr(rob, "time") else time.monotonic(),  # sim time / wall time on the robot
        q=np.asarray(rob.getj(), float),
        tcp_pose=np.asarray(rob.getl(), float),
        tcp_force=np.asarray(rob.get_tcp_force(), float),
        gripper_position=int(gripper.get_position()),
        gripper_request=int(gripper.get_requested_position()),
        gripper_object=int(gripper.get_object_status()),
        gripper_force=int(gripper.force),
        tactile=tactile,
        images=images() if images else None,
        privileged=privileged() if privileged else {},
    )


def apply(rob, gripper, cmd: Command, period: float, acc: float = 0.5) -> None:
    """Send a Command: speedl for the arm (kept alive slightly longer than one period so that
    successive commands blend), Robotiq registers for the gripper."""
    if cmd.gripper_force is not None and int(cmd.gripper_force) != int(gripper.force):
        gripper.set_force(int(cmd.gripper_force))
    if cmd.gripper_position is not None and int(cmd.gripper_position) != gripper.get_requested_position():
        gripper.set_position(int(cmd.gripper_position))
    if cmd.tcp_velocity is not None:
        rob.speedl(list(np.asarray(cmd.tcp_velocity, float)), acc=acc, min_time=2.0 * period)


def run(controller: Controller, rob, gripper, skins=None, max_time: float = 30.0,
        images=None, privileged=None, on_step: Optional[Callable[[Observation, Command], None]] = None
        ) -> List[Command]:
    """Closed loop at controller.rate_hz until the controller reports done (or max_time)."""
    period = 1.0 / controller.rate_hz
    obs = observe(rob, gripper, skins, images, privileged)
    controller.reset(obs)
    t_end = obs.t + max_time
    history = []
    while True:
        cmd = controller.act(obs)
        history.append(cmd)
        if on_step:
            on_step(obs, cmd)
        if cmd.done or obs.t > t_end:
            rob.speedl([0, 0, 0, 0, 0, 0], acc=0.5, min_time=period)
            break
        apply(rob, gripper, cmd, period)
        rob.sleep(period)
        obs = observe(rob, gripper, skins, images, privileged)
    return history
