"""Turn a remote VLA's outputs into LibrarianPickEnv actions [6 UR joint targets, gripper 0..1].

JointChunkPolicy     a model fine-tuned on this robot with openpi's UR recipe (state = joints + gripper, actions =
                     absolute joint targets + gripper): execute the first `replan` actions of each chunk.
DroidRetargetPolicy  the zero-shot generalist pi05_droid, trained on a Franka + Robotiq 2F-85 (DROID). The UR10e's
                     TCP is mapped onto a *virtual* Franka (inverse kinematics of baseline/embodiment cards) to give
                     the model the joint state it expects; each predicted DROID step (desired joints = measured + 0.2 v,
                     DROID's own convention) is mapped back through the virtual Franka's forward kinematics to a TCP
                     target, which the UR10e reaches by its own inverse kinematics. The TCP is the point between the
                     sensor pads (assembly card), so a different blade or skin changes only the card.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..embodiment import EndEffector, FixedTCPArm, se3


def _obs_images(obs):
    return obs["images"]["ext_cam"], obs["images"]["wrist_cam"]


class JointChunkPolicy:
    name = "pi05_ft"

    def __init__(self, client, replan: int = 5):
        self.client, self.replan = client, replan
        self.queue: List[np.ndarray] = []

    def reset(self):
        self.queue = []

    def act(self, obs, env=None) -> np.ndarray:
        if not self.queue:
            ext, wrist = _obs_images(obs)
            out = self.client.infer({
                "observation/ext_image": ext, "observation/wrist_image": wrist,
                "observation/state": np.r_[obs["joints"], obs["gripper"]].astype(np.float32),
                "prompt": obs["prompt"],
            })
            self.queue = [np.asarray(a, float) for a in out["actions"][: self.replan]]
        return self.queue.pop(0)


class DroidRetargetPolicy:
    name = "pi05_droid"
    MAX_JOINT_DELTA = 0.2     # rad per 15 Hz step at |v| = 1 (droid/robot_ik/robot_ik_solver.py)
    DROID_MEAN_Q = np.array([0.016, 0.267, -0.017, -2.026, -0.033, 2.345, 0.083])   # pi05_droid norm stats (state mean)

    def __init__(self, client, ee: EndEffector, open_loop: int = 8,
                 virtual_card: str = "virtual_franka_droid.toml"):
        self.client, self.ee, self.open_loop = client, ee, open_loop
        self.franka = FixedTCPArm.load(virtual_card)
        self.T_wF = self.franka.base_in_world
        self.T_Fw = se3.inv(self.T_wF)
        self.reset()

    def reset(self):
        self.chunk: Optional[np.ndarray] = None
        self.k = 0
        self.qF = self.DROID_MEAN_Q.copy()
        self.ik_fail = 0
        self.max_ik_err = 0.0

    def _ur_tcp(self, q_ur, g) -> np.ndarray:
        qg = self.ee.gripper.q_from_register(g * self.ee.gripper.register_max)
        return self.ee.arm.fk(q_ur) @ self.ee.tcp(qg, "jaw"), qg

    def virtual_joints(self, q_ur, g) -> np.ndarray:
        """Franka joints whose DROID TCP coincides with the UR10e's pad-centre TCP (jaw axes)."""
        T_world, _ = self._ur_tcp(q_ur, g)
        q, err = self.franka.ik(self.T_Fw @ T_world, self.qF, iters=100, q_rest=self.DROID_MEAN_Q, w_rest=0.05)
        self.max_ik_err = max(self.max_ik_err, err)
        self.qF = q
        return q

    def act(self, obs, env=None) -> np.ndarray:
        q_ur, g = np.asarray(obs["joints"], float), float(obs["gripper"][0])
        qF = self.virtual_joints(q_ur, g)
        if self.chunk is None or self.k >= self.open_loop:
            ext, wrist = _obs_images(obs)
            out = self.client.infer({
                "observation/exterior_image_1_left": ext, "observation/wrist_image_left": wrist,
                "observation/joint_position": qF.astype(np.float32),
                "observation/gripper_position": np.array([g], np.float32),
                "prompt": obs["prompt"],
            })
            self.chunk, self.k = np.asarray(out["actions"], float), 0
        a = self.chunk[self.k]
        self.k += 1
        v = np.clip(a[:7], -1.0, 1.0)
        grip = 1.0 if a[7] > 0.5 else 0.0                      # DROID binarises the gripper command
        qF_des = np.clip(qF + self.MAX_JOINT_DELTA * v, self.franka.arm.q_min, self.franka.arm.q_max)
        T_des = self.T_wF @ self.franka.tcp(qF_des)
        _, qg = self._ur_tcp(q_ur, g)
        q_des, err = self.ee.arm.ik(T_des, q_ur, T_tool=self.ee.tcp(qg, "jaw"), iters=100)
        if err > 1e-3:                                         # unreachable for the UR10e: hold
            self.ik_fail += 1
            q_des = q_ur
        return np.r_[q_des, grip]


# DROID state distribution (pi05_droid norm stats, assets/droid/norm_stats.json): 1st / 99th percentile per joint
DROID_Q01 = np.array([-0.828, -0.840, -0.843, -2.773, -1.843, 1.172, -2.047])
DROID_Q99 = np.array([0.900, 1.385, 0.692, -0.454, 1.732, 3.467, 2.198])


def check_retarget(virtual_card: str = "virtual_franka_droid.toml", book_x=(-0.16, 0.0, 0.16), book_z: float = 0.535,
                   spine_y: float = 0.82, verbose: bool = True) -> dict:
    """Run the pick-up task's TCP path (start -> pre-insert -> insert -> pull -> lift, horizontal approach) through
    the virtual Franka's IK: is every pose reachable, and how many joints leave DROID's q01-q99 range?
    This is how base_in_world in the card was chosen."""
    ee = EndEffector.load()
    fr = FixedTCPArm.load(virtual_card)
    hor = se3.rotvec_to_mat([-np.pi / 2, 0.0, 0.0])
    qg = ee.q_for_width(0.046)
    jaw = se3.inv(ee.tcp(qg)) @ ee.tcp(qg, "jaw")
    out = {}
    for xb in book_x:
        path = [(0.0, 0.66, 0.66)] + [(xb, y, book_z) for y in np.linspace(spine_y - 0.10, spine_y + 0.02, 5)] \
            + [(xb, y, book_z) for y in np.linspace(spine_y + 0.02, spine_y - 0.19, 8)] \
            + [(xb, spine_y - 0.19, z) for z in np.linspace(book_z, book_z + 0.08, 4)]
        q = DroidRetargetPolicy.DROID_MEAN_Q.copy()
        errs, outside = [], []
        for p in path:
            T = se3.inv(fr.base_in_world) @ se3.make(hor, p) @ jaw
            q, err = fr.ik(T, q, iters=100, q_rest=DroidRetargetPolicy.DROID_MEAN_Q, w_rest=0.05)
            errs.append(err)
            outside.append(int(np.sum((q < DROID_Q01) | (q > DROID_Q99))))
        out[xb] = {"max_ik_err": float(max(errs)), "joints_outside": outside}
        if verbose:
            print(f"book x {xb:+.2f} m: max IK error {max(errs):.1e}, joints outside DROID q01-q99 along the path "
                  f"{outside}")
    return out


if __name__ == "__main__":
    check_retarget()
