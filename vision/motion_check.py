"""Checks for slow, safe joint moves on the real UR10e (numpy + MuJoCo; no robot connection needed).

    plan = plan_joint_move(q0, q1, v_max=0.01)      # leading-joint speed for <= v_max everywhere, + checks
    plan["ok"], plan["vel"], plan["duration"], plan["min_self_dist"], plan["table"]

- speed: a UR movej interpolates linearly in joint space, all joints synchronised. Along that path the speed of every
  checked point (joint frames, flange, gripper + ZED envelope corners) is computed per unit of path parameter; the
  leading joint's speed is chosen so that the fastest point never exceeds v_max (ramps only slow it down).
- self-collision: minimum distance between the gripper + ZED and the arm's links (wrist 3 excluded: it carries
  them), and between non-adjacent links, on the MuJoCo collision geometry (fingers open), sampled along the path.
- table: vision.safety SafetyFilter.check_path (tool points and joint frames above the table).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .safety import SafetyConfig, SafetyFilter, fk_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARM_BODIES = ("base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


@lru_cache(maxsize=1)
def _model():
    from .sim_zed import add_wrist_zed
    spec = mujoco.MjSpec.from_file(str(ROOT / "assembly" / "scene.xml"))
    for b in ("table", "box"):
        spec.delete(spec.body(b))
    add_wrist_zed(spec)
    m = spec.compile()
    fl = m.body("flange").id

    def under(b):
        while b > 0:
            if b == fl:
                return True
            b = m.body_parentid[b]
        return False
    coll = lambda g: m.geom_contype[g] > 0 or m.geom_conaffinity[g] > 0 or m.geom(g).name == "zed_cam_housing"
    tool = [g for g in range(m.ngeom) if under(m.geom_bodyid[g]) and coll(g)]
    arm_idx = {m.body(n).id: i for i, n in enumerate(ARM_BODIES)}
    arm = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_idx and coll(g)]
    pairs = [(a, t) for a in arm if arm_idx[m.geom_bodyid[a]] <= 5 for t in tool]          # not wrist_3 (carrier)
    pairs += [(a, b) for i, a in enumerate(arm) for b in arm[i + 1:]
              if abs(arm_idx[m.geom_bodyid[a]] - arm_idx[m.geom_bodyid[b]]) >= 2]
    qadr = [m.joint(n).qposadr[0] for n in ARM_JOINTS]
    return m, pairs, qadr


def pair_distances(q) -> np.ndarray:
    """Distance (m, capped at 0.10) of every checked geom pair (gripper / ZED vs arm, non-adjacent links)."""
    m, pairs, qadr = _model()
    d = mujoco.MjData(m)
    d.qpos[qadr] = q
    mujoco.mj_forward(m, d)
    return np.array([mujoco.mj_geomDistance(m, d, a, b, 0.10, None) for a, b in pairs])


def self_distance(q) -> float:
    """Smallest distance (m) between the gripper / ZED and the arm, or between non-adjacent links (fingers open)."""
    return float(pair_distances(q).min())


def point_speed_per_unit(q0, q1, steps: int = 200, safety: SafetyFilter = None) -> np.ndarray:
    """|dp/ds| (m per unit path parameter) of every checked point, max over points, at each sample."""
    sf = safety or SafetyFilter()
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)

    def pts(q):
        F = fk_frames(q)
        return np.vstack([F[1:, :3, 3], sf.tool_points(F[-1])])
    out = []
    ss = np.linspace(0.0, 1.0, steps)
    for s in ss:
        e = 1e-4
        a, b = pts(q0 + (q1 - q0) * max(0.0, s - e)), pts(q0 + (q1 - q0) * min(1.0, s + e))
        out.append(np.linalg.norm(b - a, axis=1).max() / (min(1.0, s + e) - max(0.0, s - e)))
    return np.array(out)


def plan_joint_move(q0, q1, v_max: float = 0.01, z_table: float = 0.0065, min_self: float = 0.02,
                    steps: int = 200, acc: float = 0.05) -> dict:
    sf = SafetyFilter(SafetyConfig(z_table=z_table))
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dq = q1 - q0
    lead = float(np.abs(dq).max())
    if lead < 1e-6:
        return {"ok": True, "vel": 0.0, "duration": 0.0, "lead_rad": 0.0, "msg": "already there"}
    g = point_speed_per_unit(q0, q1, steps, sf)
    G = float(g.max())                                  # m per unit s at the worst sample
    vel = v_max * lead / G                              # leading joint rad/s so that G * ds/dt <= v_max
    ok_t, rep, idx = sf.check_path(q0, q1, steps=steps)
    # relative self-collision check: a pair that is already close at the start (e.g. the wrist-2 capsule and the
    # gripper base, 12 mm in every pose) may not get closer; every other pair keeps min_self
    base = pair_distances(q0)
    allowed = np.minimum(min_self, base - 0.003)
    D = np.array([pair_distances(q0 + dq * s) for s in np.linspace(0, 1, steps // 2)])
    msd = float(D.min())
    self_ok = bool(np.all(D >= allowed[None, :]))
    ok = ok_t and self_ok
    return {"ok": ok, "vel": vel, "acc": acc, "lead_rad": lead, "duration": lead / vel + vel / acc,
            "path_len_max_m": float(np.trapezoid(g, np.linspace(0, 1, len(g)))), "min_self_dist": msd,
            "table": rep.message if ok_t else f"BLOCKED at sample {idx}: {rep.message}",
            "self_ok": self_ok, "msg": "ok" if ok else ("table" if not ok_t else "self-collision risk")}
