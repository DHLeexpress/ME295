"""Check an assembly card against the MuJoCo model the simulator uses.

    python -m baseline.embodiment.check                                  # default assembly card
    python -m baseline.embodiment.check assembly_ur10e_2f140_dohyun_v2.toml

For random arm configurations and gripper openings it compares the card's forward kinematics (numpy only) with
MuJoCo's: flange, fingertips, sensor pads, pad free faces, TCP and pad gap. Run it after changing a blade, the skin,
the coupling, or the scene; it fails if any frame is off by more than 1e-6 m / 1e-6 rad.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from . import se3  # noqa: E402
from .assembly import SIDES, EndEffector  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GRIPPER_JOINTS = [("left_driver_joint", 1), ("left_follower_joint", 1), ("left_inner_knuckle_joint", -1),
                  ("right_driver_joint", -1), ("right_follower_joint", 1), ("right_inner_knuckle_joint", -1)]
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
              "wrist_3_joint")


def mj_pose(d, kind: str, idx: int) -> np.ndarray:
    if kind == "site":
        return se3.make(d.site_xmat[idx].reshape(3, 3), d.site_xpos[idx])
    return se3.make(d.xmat[idx].reshape(3, 3), d.xpos[idx])


def set_config(m, d, q_arm, q_grip):
    for n, qi in zip(ARM_JOINTS, q_arm):
        d.qpos[m.joint(n).qposadr[0]] = qi
    for n, k in GRIPPER_JOINTS:
        d.qpos[m.joint(n).qposadr[0]] = k * q_grip
    mujoco.mj_kinematics(m, d)


def frame_error(A: np.ndarray, B: np.ndarray) -> tuple:
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(np.linalg.norm(se3.mat_to_rotvec(A[:3, :3].T @ B[:3, :3])))


def run_check(ee: EndEffector, n: int = 200, seed: int = 0, verbose: bool = True) -> dict:
    mj = ee.mjcf
    m = mujoco.MjModel.from_xml_path(str(ROOT / mj["scene"]))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    flange = m.site(mj["flange_site"]).id
    tip = {s: m.body(mj["fingertip_body"][s]).id for s in SIDES}
    pad = {s: m.body(mj["pad_body"][s]).id for s in SIDES}
    face = {s: m.site(mj["pad_site"][s]).id for s in SIDES}
    box = {s: m.geom(mj["pad_body"][s]).id for s in SIDES}
    worst = {k: [0.0, 0.0] for k in ("flange", "fingertip", "pad", "pad_face", "tcp")}
    worst_gap = 0.0
    for i in range(n):
        q_arm = np.array(ee.arm.q_home) + rng.uniform(-1.2, 1.2, 6)
        q_grip = rng.uniform(0.0, ee.gripper.q_closed) if i else 0.0
        set_config(m, d, q_arm, q_grip)
        ours = ee.fk(q_arm, q_grip)
        checks = [("flange", mj_pose(d, "site", flange), ours["flange"])]
        for s in SIDES:
            checks += [("fingertip", mj_pose(d, "body", tip[s]), ours[f"{s}_fingertip"]),
                       ("pad", mj_pose(d, "body", pad[s]), ours[f"{s}_pad"])]
            # MuJoCo's surface site keeps the pad axes (+z into the layer); ours flips to the outward normal
            mj_face = mj_pose(d, "site", face[s]) @ se3.make(se3.rot_y(np.pi))
            checks.append(("pad_face", mj_face, ours[f"{s}_pad_face"]))
        mid = 0.5 * (d.site_xpos[face["left"]] + d.site_xpos[face["right"]])
        checks.append(("tcp", se3.make(ours["flange"][:3, :3], mid), ours["tcp"]))
        for name, A, B in checks:
            ep, er = frame_error(A, B)
            worst[name] = [max(worst[name][0], ep), max(worst[name][1], er)]
        gap_mj = mujoco.mj_geomDistance(m, d, box["left"], box["right"], 0.5, None)
        worst_gap = max(worst_gap, abs(gap_mj - ee.width(q_grip)))
    # frames must agree to 1e-6; the gap is compared with mj_geomDistance (convex collision, ~1e-6 m accurate)
    ok = all(v[0] < 1e-6 and v[1] < 1e-6 for v in worst.values()) and worst_gap < 1e-5
    if verbose:
        print(ee.describe())
        print(f"\nFK vs MuJoCo ({m.nq} qpos, {n} random arm/gripper configurations):")
        for k, (ep, er) in worst.items():
            print(f"  {k:10s} max position error {ep:.2e} m, max rotation error {er:.2e} rad")
        print(f"  pad gap    max error {worst_gap:.2e} m (MuJoCo distance between the two pad boxes)")
        print("PASS" if ok else "FAIL")
    return {"ok": ok, "worst": worst, "worst_gap": worst_gap}


def tool_thickness(ee: EndEffector, n: int = 9) -> tuple:
    """Blade + proud skin thickness through the pad (rays along z on an n x n grid over the pad footprint, STL in
    mm): (median, max) in m. This is what has to fit beside a book together with the clearances."""
    import trimesh
    mesh = trimesh.load(str(ROOT / ee.tool.mesh))
    pc = ee.sensor.mount[:3, 3] * 1e3
    half = 0.45 * np.array(ee.sensor.size) * 1e3
    xs, ys = np.meshgrid(np.linspace(-half[0], half[0], n), np.linspace(-half[1], half[1], n))
    origins = np.c_[xs.ravel() + pc[0], ys.ravel() + pc[1], np.full(xs.size, -200.0)]
    loc, idx, _ = mesh.ray.intersects_location(origins, np.tile([0.0, 0.0, 1.0], (len(origins), 1)))
    th = [np.ptp(loc[idx == i, 2]) for i in range(len(origins)) if np.sum(idx == i) >= 2]
    return float(np.median(th)) * 1e-3, float(np.max(th)) * 1e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("card", nargs="?", default="assembly_ur10e_2f140_dohyun_v1.toml")
    ap.add_argument("-n", type=int, default=200)
    args = ap.parse_args()
    ee = EndEffector.load(args.card)
    res = run_check(ee, n=args.n)
    try:
        med, mx = tool_thickness(ee)
        print(f"\nblade + skin thickness through the pad (from {ee.tool.mesh}): median {med * 1e3:.2f} mm, "
              f"max {mx * 1e3:.2f} mm")
    except Exception as e:  # trimesh missing or mesh not found: informational only
        print(f"(blade thickness not computed: {e})")
    for w in (0.030, 0.040, 0.046, 0.050):
        q = ee.q_for_width(w)
        print(f"pad gap {w * 1e3:4.0f} mm: register {ee.register_for_width(w):3d}, TCP {np.round(ee.tcp_pose6(q=q)[:3], 4)} m")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
