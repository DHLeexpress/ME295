"""Librarian pick-up environment for VLA policies: UR10e + 2F-140 + blade/PDMS fingers at shelf A.

Task: "pick up the <colour> book" = pull the designated book sideways-clear of the shelf and lift it, held between
the two PDMS pads. One step = 1/control_hz s. Actions are what a VLA fine-tuned with openpi's UR recipe outputs:

    action = [q_1 .. q_6 (rad, absolute joint targets), g (0 open .. 1 closed, Robotiq position request / 255)]

executed like URScript servoj (linear joint interpolation over the step, joint speed capped). The gripper squeezes
with a fixed force register (default 128 ~ 67 N per finger, the simulator's fixed-grip baseline): a vision-only VLA
has no force channel. Observations: two 224 x 224 RGB cameras (ext_cam third-person, wrist_cam), joints, gripper,
TCP pose (from baseline.embodiment), wrist F/T, and the PDMS skin (logged for the tactile follow-up, unused by the
baseline). Metrics come from simulation ground truth (see metrics()).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shear_agent.robot import PDMSSkin, SimRobot, SimRobotiqGripper, SkinSensor, pose_to_mat  # noqa: E402

from ..embodiment import EndEffector, se3  # noqa: E402
from . import scene as _scene  # noqa: E402

HORIZONTAL = (-np.pi / 2, 0.0, 0.0)      # tool z -> +y (into shelf A), blade width vertical (run_librarian.py)
START_TCP = (0.0, 0.66, 0.66)            # run_librarian.py start pose
START_SEED = (-1.97, -1.8, 2.0, -0.2, 1.17, 3.14)
CLEARANCE = 0.003                        # air between each pad and the book while inserting (run_librarian.py)


class ServoSimRobot(SimRobot):
    """SimRobot + URScript-style servoj for streaming joint targets (a VLA's action chunks)."""

    max_joint_speed = 1.0   # rad/s cap per joint (UR10e allows 2-3.1 rad/s; a policy's jumps are clipped here)

    def servoj(self, q, t: float):
        q1 = np.asarray(q, float)
        q0 = self._q_ref.copy()
        n = max(1, int(round(t / self.dt)))
        dq = q1 - q0
        cap = self.max_joint_speed * n * self.dt
        clipped = bool(np.any(np.abs(dq) > cap + 1e-12))
        dq = np.clip(dq, -cap, cap)

        def motion():
            for k in range(1, n + 1):
                yield q0 + dq * (k / n), dq / (n * self.dt)
            while True:          # hold the target until the next command (servoj semantics)
                yield q0 + dq, np.zeros(6)
        self._motion = motion()
        return clipped


@dataclass
class EnvConfig:
    control_hz: float = 10.0
    image_size: int = 224
    max_time: float = 40.0          # s of simulated time per episode
    grip_force: int = 128           # Robotiq force register (0..255 -> 10..125 N per finger)
    grip_speed: int = 128
    hold_time: float = 1.0          # success needs the lifted book held this long
    lift_height: float = 0.03       # book bottom above the board (m) to count as lifted
    render: bool = True
    skin: bool = True
    ee_card: str = "assembly_ur10e_2f140_dohyun_v1.toml"
    cameras: tuple = ("ext_cam", "wrist_cam")
    pdms_mu: Optional[float] = None   # PDMS-cover friction override (model default 1.0)
    protective_stop: float = 150.0    # N on the wrist F/T: the episode ends as a failure (UR protective stop)
    keep_frames: bool = False         # keep (ext | wrist) frames for a video


class LibrarianPickEnv:
    def __init__(self, variation: Optional[_scene.Variation] = None, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self.variation = variation or _scene.nominal()
        self.m, self.info = _scene.build(self.variation)
        if cfg.pdms_mu is not None:
            for g in range(self.m.ngeom):
                if "_pdms_" in (self.m.geom(g).name or ""):
                    self.m.geom_friction[g, 0] = cfg.pdms_mu
        self.ee = EndEffector.load(cfg.ee_card)
        self.rob = ServoSimRobot(model=self.m, verbose=False)
        mujoco.mj_setConst(self.m, self.rob.d)
        self.gripper = SimRobotiqGripper(self.rob, speed=cfg.grip_speed, force=cfg.grip_force)
        # pads open to the book + 2 x 3 mm; in gaps too narrow for that (blade + skin + 3 mm + 1 mm spare > gap) the
        # blades are centred in the gaps instead (only the narrow-gap test suite has such gaps)
        gap = min(self.variation.gap_left, self.variation.gap_right)
        blade = self.ee.tool.thickness
        clearance = CLEARANCE if gap - blade >= CLEARANCE + 0.001 else max(0.0005, 0.5 * (gap - blade))
        self.open_width = 2 * self.info.half[0] + 2 * clearance
        self.tcp = self.ee.tcp_pose6(width=self.open_width)
        self.rob.set_tcp(self.tcp)
        self.renderer = mujoco.Renderer(self.m, cfg.image_size, cfg.image_size) if cfg.render else None
        self.skins: Dict[str, SkinSensor] = {}
        if cfg.skin:
            self.skins = {s: SkinSensor(PDMSSkin(self.m, s), rate_hz=20.0) for s in ("left", "right")}
            for sk in self.skins.values():
                self.rob.callbacks.append(sk.tick)
        d, m = self.rob.d, self.m
        self._book = m.body(self.info.designated).id
        self._book_geom = m.geom(self.info.designated).id
        self._others = [m.body(n).id for n in self.info.others]
        self._upper_others = [m.body(n).id for n in self.info.others if n in _scene.UPPER]
        self._pad_geoms = {s: {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(f"{s}_pdms_")}
                           for s in ("left", "right")}
        grip_bodies = set()
        for b in range(m.nbody):  # every body below the flange (gripper, blades, camera)
            p = b
            while p > 0:
                if m.body(p).name == "flange":
                    grip_bodies.add(b)
                    break
                p = m.body_parentid[p]
        self._grip_bodies = grip_bodies
        self._hard_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in grip_bodies
                            and g not in self._pad_geoms["left"] | self._pad_geoms["right"]}
        self._world_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] not in grip_bodies}
        self._armed = False
        self.rob.callbacks.append(self._safety_tick)   # 500 Hz, like the UR controller's force monitoring
        self.reset()

    def _safety_tick(self, rob):
        """UR-style protective stop: checked every control tick (2 ms), not only at the policy rate."""
        if not self._armed or self.flags["protective_stop"]:
            return
        wrench = float(np.linalg.norm(rob.get_tcp_force()[:3]))
        self.max_wrench = max(self.max_wrench, wrench)
        if wrench > self.cfg.protective_stop:
            self.flags["protective_stop"] = True
            self.t_flag["protective_stop"] = self.elapsed
            rob.stopj(acc=3.0)

    # ------------------------------------------------------------------ episode
    def start_q(self) -> np.ndarray:
        off = np.asarray(self.variation.start_offset, float)
        T = pose_to_mat([*(np.asarray(START_TCP) + off[:3]), *HORIZONTAL])
        T[:3, :3] = se3.rotvec_to_mat(off[3:]) @ T[:3, :3]
        q, err = self.rob.ik(T @ np.linalg.inv(pose_to_mat(self.tcp)), START_SEED, iters=300)
        if err > 1e-6:
            raise RuntimeError(f"start pose unreachable (IK error {err:.1e})")
        return q

    def reset(self):
        self._armed = False
        self.rob.reset(self.start_q())
        self.gripper.set_position(0, _log=False)
        self.gripper._ramp = 0.0
        self.rob.sleep(0.3)  # settle
        d = self.rob.d
        self.t0 = self.rob.time
        self.book0 = d.xpos[self._book].copy()
        self.others0 = d.xpos[self._others].copy()
        self.n_steps = 0
        self.n_clipped = 0
        self.flags = {k: False for k in ("contact", "moved", "extracted", "lifted", "success", "dropped",
                                         "protective_stop", "wrong_book")}
        self.frames = []
        self.t_flag: Dict[str, float] = {}
        self.t_hold = None
        self.max_other_disp = 0.0
        self.max_wrench = 0.0
        self.max_hard_contact = 0.0
        self.max_pad_force = 0.0
        self.max_tilt = 0.0
        self.min_dist_target = np.inf     # closest TCP approach to the designated book's spine centre (m)
        self.min_dist_other = np.inf      # ... to any other upper-tier book's spine centre
        self.trace = []                   # per update: t, TCP xyz, book xyz, gripper position, |wrist force|
        self._armed = True
        return self.observe()

    @property
    def elapsed(self) -> float:
        return self.rob.time - self.t0

    def done(self) -> bool:
        return self.flags["success"] or self.flags["protective_stop"] or self.elapsed >= self.cfg.max_time

    # ------------------------------------------------------------------ observation
    def render(self, cam: str) -> np.ndarray:
        self.renderer.update_scene(self.rob.d, camera=cam)
        return self.renderer.render().copy()

    def gripper_state(self) -> float:
        return self.gripper.get_position() / 255.0

    def observe(self) -> dict:
        rob = self.rob
        q = np.asarray(rob.getj(), float)
        obs = {
            "t": self.elapsed,
            "joints": q,
            "gripper": np.array([self.gripper_state()]),
            "tcp_pose": np.asarray(rob.getl(), float),                 # urx pose of the TCP set from the card
            "tcp_force": np.asarray(rob.get_tcp_force(), float),
            "prompt": self.info.instruction,
        }
        if self.renderer is not None:
            obs["images"] = {c: self.render(c) for c in self.cfg.cameras}
            if self.cfg.keep_frames:
                self.frames.append(np.concatenate([obs["images"][c] for c in self.cfg.cameras], axis=1))
        if self.skins:
            fr = {s: sk.read() for s, sk in self.skins.items()}
            obs["tactile"] = {s: np.concatenate([f.normal[..., None], f.shear], axis=-1) for s, f in fr.items()}
        return obs

    # ------------------------------------------------------------------ action
    def step(self, action) -> dict:
        a = np.asarray(action, float)
        period = 1.0 / self.cfg.control_hz
        self.n_clipped += int(self.rob.servoj(a[:6], period))
        self.gripper.set_position(int(round(np.clip(a[6], 0.0, 1.0) * 255)), _log=False)
        self.rob.sleep(period)
        self.n_steps += 1
        self.update_metrics()
        return self.observe()

    # ------------------------------------------------------------------ metrics (simulation ground truth)
    def pad_forces(self) -> Dict[str, float]:
        """Normal force between each pad and the designated book (N)."""
        m, d = self.m, self.rob.d
        f6 = np.zeros(6)
        out = {"left": 0.0, "right": 0.0}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self._book_geom not in (g1, g2):
                continue
            other = g2 if g1 == self._book_geom else g1
            for s in out:
                if other in self._pad_geoms[s]:
                    mujoco.mj_contactForce(m, d, i, f6)
                    out[s] += abs(f6[0])
        return out

    def hard_contact_force(self) -> float:
        """Largest normal force between a non-skin part of the gripper (blade body, finger links, camera) and the
        world (books, shelf): the gripper hitting things."""
        m, d = self.m, self.rob.d
        f6 = np.zeros(6)
        worst = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in self._hard_geoms and c.geom2 in self._world_geoms) or \
               (c.geom2 in self._hard_geoms and c.geom1 in self._world_geoms):
                mujoco.mj_contactForce(m, d, i, f6)
                worst = max(worst, abs(f6[0]))
        return worst

    def update_metrics(self):
        d = self.rob.d
        t = self.elapsed
        p = d.xpos[self._book]
        R = d.xmat[self._book].reshape(3, 3)
        half = self.info.half
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        self.max_tilt = max(self.max_tilt, tilt)
        # extent of the (possibly tilted) box along y and z
        ext = np.abs(R) @ half
        back_y = p[1] + ext[1]
        bottom_z = p[2] - ext[2]
        pads = self.pad_forces()
        held = min(pads.values()) > 1.0
        self.max_pad_force = max(self.max_pad_force, max(pads.values()))
        disp = np.linalg.norm(d.xpos[self._others] - self.others0, axis=1)
        self.max_other_disp = max(self.max_other_disp, float(disp.max(initial=0.0)))
        tcp = np.asarray(self.rob.getl()[:3])
        self.trace.append([t, *tcp, *p, self.gripper_state(), float(np.linalg.norm(self.rob.get_tcp_force()[:3]))])
        spine = lambda b: d.xpos[b] - np.array([0.0, self.m.geom_size[self.m.body_geomadr[b]][1], 0.0])
        self.min_dist_target = min(self.min_dist_target, float(np.linalg.norm(tcp - spine(self._book))))
        if self._upper_others:
            self.min_dist_other = min(self.min_dist_other,
                                      min(float(np.linalg.norm(tcp - spine(b))) for b in self._upper_others))
        self.max_hard_contact = max(self.max_hard_contact, self.hard_contact_force())

        def mark(k, cond):
            if cond and not self.flags[k]:
                self.flags[k] = True
                self.t_flag[k] = t
        mark("contact", held)
        mark("moved", self.book0[1] - p[1] > 0.02)
        extracted = back_y < _scene.SHELF_FRONT_Y - 0.002
        mark("extracted", extracted)
        lifted = extracted and bottom_z > _scene.BOARD_TOP + self.cfg.lift_height
        mark("lifted", lifted)
        if lifted and held and tilt < 30.0:
            self.t_hold = t if self.t_hold is None else self.t_hold
            if t - self.t_hold >= self.cfg.hold_time:
                mark("success", True)
        else:
            self.t_hold = None
        if self.flags["lifted"] and not held and bottom_z < _scene.BOARD_TOP:
            mark("dropped", True)
        moved_y = self.others0[:, 1] - d.xpos[self._others][:, 1]
        mark("wrong_book", bool(np.any(moved_y > 0.05)))   # another book pulled > 5 cm toward the robot

    def insertion(self) -> dict:
        """Where the TCP crossed the plane of the spines (from the trace): lateral offset from the designated book's
        centre (mm) and the deepest TCP point behind the spines (mm). The pads have 3 mm of air on each side."""
        if len(self.trace) < 2:
            return {}
        tr = np.asarray(self.trace)
        spine_y = self.book0[1] - self.info.half[1]
        out = {"insert_depth_mm": 1e3 * float(tr[:, 2].max() - spine_y)}
        k = np.where((tr[:-1, 2] < spine_y) & (tr[1:, 2] >= spine_y))[0]
        if len(k):
            out["insert_lateral_mm"] = 1e3 * float(tr[k[0] + 1, 1] - tr[k[0] + 1, 4])
            out["insert_height_mm"] = 1e3 * float(tr[k[0] + 1, 3] - tr[k[0] + 1, 6])
        return out

    def metrics(self) -> dict:
        d = self.rob.d
        disp = np.linalg.norm(d.xpos[self._others] - self.others0, axis=1)
        v = self.variation
        return {
            "success": bool(self.flags["success"]),
            **{f"stage_{k}": bool(self.flags[k]) for k in ("contact", "moved", "extracted", "lifted", "dropped")},
            "protective_stop": bool(self.flags["protective_stop"]),
            "wrong_book": bool(self.flags["wrong_book"]),
            "open_width_mm": 1e3 * float(self.open_width),
            "min_tcp_target_mm": 1e3 * float(self.min_dist_target),
            "min_tcp_other_mm": 1e3 * float(self.min_dist_other),
            "approached_target": bool(self.min_dist_target < min(self.min_dist_other, 0.05)),
            **{f"t_{k}": float(self.t_flag[k]) for k in self.t_flag},
            "time": float(self.elapsed),
            "steps": int(self.n_steps),
            "clipped_steps": int(self.n_clipped),
            "book_disp_m": float(np.linalg.norm(d.xpos[self._book] - self.book0)),
            "book_tilt_max_deg": float(self.max_tilt),
            "others_max_disp_mm": 1e3 * float(self.max_other_disp),
            "others_moved_gt_5mm": int(np.sum(disp > 0.005)),
            "others_moved_gt_20mm": int(np.sum(disp > 0.02)),
            "wrist_force_max_N": float(self.max_wrench),
            "hard_contact_max_N": float(self.max_hard_contact),
            "pad_force_max_N": float(self.max_pad_force),
            "designated": v.designated, "color": v.color, "gap_left_mm": 1e3 * v.gap_left,
            "gap_right_mm": 1e3 * v.gap_right, "mass_kg": float(self.info.mass),
            "thickness_mm": 2e3 * float(self.info.half[0]), "distractor": v.distractor,
            "instruction": self.info.instruction,
            **self.insertion(),
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
