"""Scripted expert for the pick-up task (privileged: knows the designated book's pose) and a demo recorder.

The motions are run_librarian.py's: approach 10 cm in front of the spines with the pads opened to the book + 2 x 3 mm,
slide the blades 20 mm past the spines beside the book (guarded at 15 N), then the contact phase with the simulator's
no-skin controller FixedForceGrasp (close at a fixed force register, pull the book out, lift 8 cm, hold). For demo
diversity the approach speed, a via point and the pre-shaping timing are randomised; the insertion line is exact.

The recorder samples the episode at the policy rate. The action label at sample k is the arm's servo reference and
the gripper's position request at sample k + 1 (what a teleoperator's command stream would contain), so a policy
trained on it is executed with LibrarianPickEnv.step(), which servos to exactly such targets.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from shear_agent.control import FixedForceGrasp, run

from . import scene as _scene
from .env import HORIZONTAL, LibrarianPickEnv

INSERT_DEPTH = 0.020   # TCP 20 mm behind the spines (run_librarian.py)
LIFT = 0.08


def _pose(p, r=HORIZONTAL):
    return [float(v) for v in (*p, *r)]


class Recorder:
    """Samples observations + servo references at the policy rate via the robot's tick callbacks."""

    def __init__(self, env: LibrarianPickEnv, images: bool = True):
        self.env, self.images = env, images
        self.period = 1.0 / env.cfg.control_hz
        self.next_t = env.rob.time
        self.frames: List[dict] = []
        self.refs: List[np.ndarray] = []
        env.rob.callbacks.append(self._tick)

    def _tick(self, rob):
        if rob.time + 1e-9 < self.next_t:
            return
        self.next_t += self.period
        self.env.update_metrics()
        obs = self.env.observe() if self.images else {"joints": np.asarray(rob.getj())}
        obs["phase"] = getattr(self, "phase", "")
        self.frames.append(obs)
        self.refs.append(np.r_[rob._q_ref.copy(), self.env.gripper.get_requested_position() / 255.0])

    def detach(self):
        if self._tick in self.env.rob.callbacks:
            self.env.rob.callbacks.remove(self._tick)

    def episode(self) -> Dict[str, np.ndarray]:
        """Arrays of length T (observations) with actions[k] = reference at k + 1."""
        refs = np.array(self.refs)
        actions = np.vstack([refs[1:], refs[-1:]])
        f = self.frames
        out = {
            "joints": np.array([o["joints"] for o in f], np.float32),
            "gripper": np.array([o["gripper"] for o in f], np.float32),
            "tcp_pose": np.array([o["tcp_pose"] for o in f], np.float32),
            "tcp_force": np.array([o["tcp_force"] for o in f], np.float32),
            "actions": actions.astype(np.float32),
            "t": np.array([o["t"] for o in f], np.float32),
            "phase": np.array([o["phase"] for o in f]),
        }
        if "images" in f[0]:
            for cam in f[0]["images"]:
                out[f"image_{cam}"] = np.stack([o["images"][cam] for o in f])
        if "tactile" in f[0]:
            for side in ("left", "right"):
                out[f"tactile_{side}"] = np.stack([o["tactile"][side] for o in f]).astype(np.float32)
        return out


def run_expert(env: LibrarianPickEnv, rng: Optional[np.random.Generator] = None, recorder: Optional[Recorder] = None,
               say=lambda s: None) -> bool:
    """Drive the env with the scripted expert. Returns True if the env reports success."""
    rng = rng or np.random.default_rng()
    rob, gripper, info = env.rob, env.gripper, env.info
    c = info.center
    open_reg = env.ee.register_for_width(env.open_width)
    pre = np.array([c[0], _scene.SPINE_Y - 0.10, c[2]])
    insert = np.array([c[0], _scene.SPINE_Y + INSERT_DEPTH, c[2]])

    def phase(name):
        if recorder is not None:
            recorder.phase = name
        say(name)

    phase("approach")
    rob.sleep(float(rng.uniform(0.0, 0.4)))       # reaction time
    gripper.set_position(open_reg)                # pre-shape while moving
    p0 = np.asarray(rob.getl()[:3])
    if rng.random() < 0.5:                        # random via point between start and pre-insert
        via = 0.5 * (p0 + pre) + rng.uniform(-0.04, 0.04, 3)
        rob.movel(_pose(via), acc=0.5, vel=float(rng.uniform(0.10, 0.16)))
    rob.movel(_pose(pre), acc=0.5, vel=float(rng.uniform(0.10, 0.16)))
    while gripper.get_object_status() == 0:       # the jaw must be open before the blades go in
        rob.sleep(0.05)
    rob.sleep(float(rng.uniform(0.1, 0.4)))

    phase("insert")
    rob.movel(_pose(insert), acc=0.2, vel=float(rng.uniform(0.025, 0.035)), wait=False)
    rob.sleep(0.1)
    while rob.is_program_running():
        if rob.get_force() > 15.0:
            rob.stopl()
            say("unexpected contact while inserting")
            return False
        rob.sleep(0.02)
    rob.sleep(0.3)

    phase("grasp")
    ctrl = FixedForceGrasp(force=env.cfg.grip_force, pull_dist=2 * info.half[1] + 0.04, lift=LIFT,
                           pull_speed=float(rng.uniform(0.035, 0.045)), lift_speed=float(rng.uniform(0.025, 0.035)))
    hist = run(ctrl, rob, gripper, max_time=25.0)
    phase("hold")
    t_end = rob.time + env.cfg.hold_time + 0.5
    while rob.time < t_end and not env.flags["success"]:
        rob.sleep(0.05)
        env.update_metrics()
    return bool(env.flags["success"]) and bool(hist) and hist[-1].phase == "done"
