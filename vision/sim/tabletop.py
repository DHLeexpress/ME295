"""Tabletop grasp scene: the UR10e + 2F-140 + blades + wrist ZED (assembly/scene.xml) on a table at z = 0 (the real
table: the calibration board lies at z = 5-7 mm in the base frame), one random object in front of the robot.

    obj = sample_object(np.random.default_rng(0))
    env = TabletopEnv(obj)              # robot at the top-down view configuration Q_VIEW, gripper open
    frame = env.grab()                  # calibrated RGB-D (perception's belief of the camera pose)
    env.metrics()                       # lifted / held / pushed / collisions (simulation ground truth)
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shear_agent.robot import SimRobot, SimRobotiqGripper  # noqa: E402

from ..camera import RGBDFrame  # noqa: E402
from ..sim_zed import SimZedCamera, add_wrist_zed  # noqa: E402
from ..wrist import WristCamera  # noqa: E402

SCENE = ROOT / "assembly" / "scene.xml"
TABLE_Z = 0.0
WORKSPACE_CENTER = (0.10, -0.50)
WORKSPACE_HALF = (0.10, 0.07)          # objects are placed (and must be fully seen) inside centre +- half
# Top-down view configuration: the ray through pixel (0.5, 0.25) of the wrist ZED is vertical and hits
# WORKSPACE_CENTER from 0.70 m; then the whole workspace (objects up to 16 cm tall) projects into the image band the
# gripper leaves free (v < 0.45 H). Elbow up, IK seeded with calibration pose 16 (the posture the real robot ran).
# vision.look.view_pose(wrist, (0.10, -0.50, 0.0065), distance=0.7, pixel=(0.5, 0.25), yaw=0) -> IK:
Q_VIEW = np.radians([74.288, -122.705, 111.369, -85.508, -87.078, 164.463])
PALETTE = {"red": (0.75, 0.10, 0.10), "blue": (0.10, 0.25, 0.70), "green": (0.10, 0.55, 0.20),
           "yellow": (0.90, 0.78, 0.15), "orange": (0.92, 0.45, 0.08), "purple": (0.45, 0.15, 0.55),
           "white": (0.92, 0.92, 0.90), "black": (0.07, 0.07, 0.08)}
DEPTH_MODELS = {"exact": {}, "noise": {"disparity_noise_px": 0.3},
                "fatten": {"disparity_noise_px": 0.3, "edge_fatten_px": 2}}


@dataclass
class ObjectSpec:
    shape: str                       # box | cylinder | sphere
    size: Tuple[float, ...]          # box: half extents; cylinder: (radius, half height); sphere: (radius,)
    mass: float
    friction: float
    colour: str
    xy: Tuple[float, float]
    yaw: float

    @property
    def half_height(self) -> float:
        return {"box": lambda s: s[2], "cylinder": lambda s: s[1], "sphere": lambda s: s[0]}[self.shape](self.size)

    @property
    def grasp_width(self) -> float:
        """True width across the narrowest horizontal axis."""
        return {"box": lambda s: 2 * min(s[0], s[1]), "cylinder": lambda s: 2 * s[0],
                "sphere": lambda s: 2 * s[0]}[self.shape](self.size)

    def to_dict(self):
        return asdict(self)


def sample_object(rng: np.random.Generator, center=WORKSPACE_CENTER, spread=WORKSPACE_HALF) -> ObjectSpec:
    shape = str(rng.choice(["box", "cylinder", "sphere"], p=[0.45, 0.35, 0.20]))
    if shape == "box":
        a, b = sorted(rng.uniform([0.020, 0.025], [0.045, 0.070]))
        size = (float(a), float(b), float(rng.uniform(0.03, 0.08)))
    elif shape == "cylinder":
        size = (float(rng.uniform(0.022, 0.045)), float(rng.uniform(0.03, 0.08)))
    else:
        size = (float(rng.uniform(0.030, 0.045)),)
    return ObjectSpec(shape=shape, size=size, mass=float(rng.uniform(0.05, 0.5)), friction=float(rng.uniform(0.6, 1.0)),
                      colour=str(rng.choice(list(PALETTE))),
                      xy=(float(center[0] + rng.uniform(-1, 1) * spread[0]), float(center[1] + rng.uniform(-1, 1) * spread[1])),
                      yaw=float(rng.uniform(-np.pi, np.pi)))


def build(obj: ObjectSpec, wrist: Optional[WristCamera] = None) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for b in ("table", "box"):
        spec.delete(spec.body(b))
    spec.geom("floor").pos = [0.0, 0.0, -0.75]
    mat = spec.add_material()
    mat.name, mat.rgba = "plywood", [0.80, 0.64, 0.42, 1.0]
    top = spec.worldbody.add_geom()
    top.name, top.type, top.material = "tabletop", mujoco.mjtGeom.mjGEOM_BOX, "plywood"
    top.size, top.pos = [0.75, 0.55, 0.02], [0.15, -0.45, TABLE_Z - 0.02]
    top.friction = [0.5, 0.005, 0.0001]
    top.conaffinity = 3
    body = spec.worldbody.add_body()
    body.name = "object"
    body.pos = [obj.xy[0], obj.xy[1], TABLE_Z + obj.half_height + 0.001]
    body.quat = [np.cos(obj.yaw / 2), 0.0, 0.0, np.sin(obj.yaw / 2)]
    body.add_freejoint().name = "object"
    g = body.add_geom()
    g.name = "object"
    g.type = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
              "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}[obj.shape]
    g.size = list(obj.size) + [0.0] * (3 - len(obj.size))
    g.mass = obj.mass
    g.friction = [obj.friction, 0.005, 0.0001]
    g.rgba = [*PALETTE[obj.colour], 1.0]
    g.conaffinity = 3
    add_wrist_zed(spec, wrist)
    return spec.compile()


class TabletopEnv:
    """Simulated cell for one grasp episode. The camera is mounted at the TRUE calibrated pose; perception uses
    `wrist_belief` (default the same; pass a perturbed one to emulate a hand-eye calibration error)."""

    def __init__(self, obj: ObjectSpec, depth: str = "noise", scale: float = 0.5, seed: int = 0,
                 wrist_belief: Optional[WristCamera] = None, q_start=Q_VIEW, grip_speed: int = 200):
        self.obj = obj
        self.wrist = WristCamera.active(warn=False)
        self.wrist_belief = wrist_belief or self.wrist
        self.m = build(obj, self.wrist)
        self.rob = SimRobot(model=self.m, verbose=False)
        mujoco.mj_setConst(self.m, self.rob.d)
        self.gripper = SimRobotiqGripper(self.rob, speed=grip_speed, force=0)
        self.cam = SimZedCamera(self.m, scale=scale, seed=seed, **DEPTH_MODELS[depth])
        m = self.m
        self.body = m.body("object").id
        self.geom = m.geom("object").id
        self.pads = {g for g in range(m.ngeom) if "_pdms_" in (m.geom(g).name or "")}
        self.left = {g for g in self.pads if m.geom(g).name.startswith("left")}
        self.right = self.pads - self.left
        fl = m.body("flange").id

        def under(b):
            while b > 0:
                if b == fl:
                    return True
                b = m.body_parentid[b]
            return False
        self.hard = {g for g in range(m.ngeom) if under(m.geom_bodyid[g]) and g not in self.pads}
        self.table = m.geom("tabletop").id
        self.rob.reset(q_start)
        self.gripper.set_position(0, _log=False)
        self.gripper._ramp = 0.0
        self.rob.sleep(0.4)
        self.p0 = self.rob.d.xpos[self.body].copy()
        self.max_hard_force = 0.0
        self.min_table_clear = np.inf
        self.pushed_before_close = 0.0
        self.closed = False
        self.rob.callbacks.append(self._tick)

    # ------------------------------------------------------------------ sensing
    def grab(self) -> RGBDFrame:
        return self.cam.grab(self.rob.d).with_pose(self.wrist_belief.T_base_cam(self.rob.flange_pose()))

    # ------------------------------------------------------------------ disturbances
    def move_object(self, dxy):
        """An external push: the object jumps by dxy (m) on the table."""
        j = self.m.joint("object").qposadr[0]
        self.rob.d.qpos[j:j + 2] += np.asarray(dxy, float)
        self.p0[:2] += np.asarray(dxy, float)
        mujoco.mj_forward(self.m, self.rob.d)

    # ------------------------------------------------------------------ metrics (ground truth)
    def _tick(self, rob):
        d, m = rob.d, self.m
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom1, c.geom2
            if (a in self.hard and (b == self.geom or b == self.table)) or (b in self.hard and (a == self.geom or a == self.table)):
                mujoco.mj_contactForce(m, d, i, f6)
                self.max_hard_force = max(self.max_hard_force, abs(f6[0]))
        if not self.closed:
            self.pushed_before_close = max(self.pushed_before_close, float(np.linalg.norm(d.xpos[self.body][:2] - self.p0[:2])))

    def pad_contacts(self) -> Tuple[bool, bool]:
        d = self.rob.d
        l = r = False
        for i in range(d.ncon):
            c = d.contact[i]
            o = c.geom2 if c.geom1 == self.geom else c.geom1 if c.geom2 == self.geom else None
            if o is None:
                continue
            l |= o in self.left
            r |= o in self.right
        return l, r

    def lifted(self) -> float:
        return float(self.rob.d.xpos[self.body][2] - self.p0[2])

    def metrics(self) -> dict:
        l, r = self.pad_contacts()
        return {"lift_m": self.lifted(), "held": bool(l and r), "success": bool(l and r and self.lifted() > 0.05),
                "pushed_before_close_mm": 1e3 * self.pushed_before_close,
                "hard_contact_max_N": float(self.max_hard_force)}

    def close(self):
        self.cam.close()
