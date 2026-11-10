#!/usr/bin/env python3
"""Offline IK for a calibration session's recorded tool poses -> elbow-up joint targets for
`ur10e_hand_eye_calibration.py --replay` (written to SESSION_DIR/replay_joints_elbow_up.npy).

For every T_base_ee in SESSION_DIR/poses.npz: finds all UR10e IK solutions (Drake IK from many
seeds), keeps the elbow-up ones (elbow joint above the shoulder->wrist line), picks the one in
the same configuration family as a reference configuration (default: a working grasp plan's
pregrasp joints) and closest to the previous pose, and checks it by FK.

Frame convention (checked against a working grasp plan's ur_joints_rad / flange_pose_ur pair to
0.17mm / 0.06deg): controller pose = RotZ(pi) @ FK(URDF tool0). The recorded T_base_ee came from
urx get_pose() with the controller's TCP at the tool flange (tool0).

Run in the leap_retarget env:
    python solve_replay_joints.py captures_ur10e/<session>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import Solve
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "ur10e_description/ur10e.urdf"
DEFAULT_REF_PLAN = ROOT / "springgrasp/runs/can_round3/plans/grasp_00.plan.json"
NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
RZ = RigidTransform(RotationMatrix.MakeZRotation(np.pi))  # controller base <-> URDF base_link

plant = MultibodyPlant(0.0)
parser = Parser(plant)
parser.package_map().Add("ur_description", str(ROOT / "ur10e_description"))
parser.AddModels(str(URDF))  # the URDF itself welds base_link to world
plant.Finalize()
ctx = plant.CreateDefaultContext()
IDX = [plant.GetJointByName(n).position_start() for n in NAMES]
LO = np.array([plant.GetJointByName(n).position_lower_limits()[0] for n in NAMES])
HI = np.array([plant.GetJointByName(n).position_upper_limits()[0] for n in NAMES])
TOOL0 = plant.GetFrameByName("tool0")


def _set(qj):
    q = np.zeros(plant.num_positions())
    q[IDX] = qj
    plant.SetPositions(ctx, q)


def fk(qj, frame="tool0"):
    _set(qj)
    return RZ @ plant.CalcRelativeTransform(ctx, plant.world_frame(), plant.GetFrameByName(frame))


def elbow_height(qj):
    """Height of the elbow joint above the shoulder->wrist line (>0: elbow up)."""
    sh, el, wr = (fk(qj, f).translation() for f in ("upper_arm_link", "forearm_link", "wrist_1_link"))
    t = np.dot(el - sh, wr - sh) / np.dot(wr - sh, wr - sh)
    return (el - (sh + t * (wr - sh)))[2]


def wrap(q):
    return (q + np.pi) % (2 * np.pi) - np.pi


def all_solutions(T_ctrl, rng):
    X = RZ.inverse() @ RigidTransform(T_ctrl)
    seeds = ([np.array([a, b, c, -1.5, e, 0.0]) for a in (-2.5, 0.0, 1.5, 3.0) for b in (-2.0, -1.0)
              for c in (-1.8, 1.8) for e in (-1.5, 1.5)] + [rng.uniform(-np.pi, np.pi, 6) for _ in range(40)])
    sols = []
    for s in seeds:
        ik = InverseKinematics(plant, ctx, with_joint_limits=True)
        ik.AddPositionConstraint(TOOL0, [0, 0, 0], plant.world_frame(), X.translation() - 1e-5, X.translation() + 1e-5)
        ik.AddOrientationConstraint(plant.world_frame(), X.rotation(), TOOL0, RotationMatrix(), 1e-4)
        q0 = np.zeros(plant.num_positions())
        q0[IDX] = s
        ik.prog().SetInitialGuess(ik.q(), q0)
        r = Solve(ik.prog())
        if not r.is_success():
            continue
        w = wrap(r.GetSolution(ik.q())[IDX])
        if not any(np.allclose(w, o, atol=2e-3) for o in sols):
            sols.append(w)
    return sols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--ref-plan", type=Path, default=DEFAULT_REF_PLAN,
                    help="a grasp *.plan.json whose pregrasp joints define the configuration family to stay in")
    args = ap.parse_args()

    T_list = np.load(args.session_dir / "poses.npz")["T_base_ee"]
    ref = np.array(json.loads(args.ref_plan.read_text())["pregrasp"]["ur_joints_rad"])
    print(f"reference configuration (deg) {np.round(np.degrees(ref)).astype(int).tolist()}: "
          f"elbow {'UP' if elbow_height(ref) > 0 else 'DOWN'}")
    rng = np.random.default_rng(0)
    chain, prev = [], ref.copy()
    for i, T in enumerate(T_list, 1):
        sols = all_solutions(T, rng)
        ups = [w for w in sols if elbow_height(w) > 0]
        if not ups:
            raise SystemExit(f"pose {i}: no elbow-up IK solution ({len(sols)} solutions total)")
        same = [w for w in ups if np.sign(w[4]) == np.sign(wrap(ref)[4]) and abs(wrap(w[0] - ref[0])) < np.radians(90)]
        w = min(same or ups, key=lambda w: np.abs(wrap(w - prev)).max())
        q = prev + wrap(w - prev)  # the 2*pi version closest to the previous joints
        q = np.where(q > HI, q - 2 * np.pi, np.where(q < LO, q + 2 * np.pi, q))
        X = fk(q)
        perr = np.linalg.norm(X.translation() - T[:3, 3]) * 1000
        rerr = np.degrees(R.from_matrix(X.rotation().matrix().T @ T[:3, :3]).magnitude())
        assert perr < 0.5 and rerr < 0.1 and elbow_height(q) > 0, f"pose {i}: check failed"
        print(f"pose {i:2d}: {len(sols)} solutions, {len(ups)} elbow-up; chose {'same family' if same else 'OTHER family'} "
              f"{np.round(np.degrees(q)).astype(int).tolist()}  FK err {perr:.2f}mm/{rerr:.3f}deg  "
              f"largest joint move {np.degrees(np.abs(q - prev).max()):.0f}deg")
        chain.append(q)
        prev = q
    out = args.session_dir / "replay_joints_elbow_up.npy"
    np.save(out, np.array(chain))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
