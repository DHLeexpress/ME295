#!/usr/bin/env python3
"""Meshcat animation of a calibration replay: the UR10e (with the LEAP hand mounted) stepping
through SESSION_DIR/replay_joints_elbow_up.npy, joint-interpolated between poses like a joint
movej. Also shows the ChArUco board at the calibrated T_base_board, the wrist camera frame
(T_base_ee @ T_ee_cam_wrist) and a line from the camera to the board center.

Run in the leap_retarget env:
    python visualize_replay.py captures_ur10e/<session>
then open the printed URL and press play in Meshcat's Animations panel.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from pydrake.geometry import Box, Cylinder, MeshcatVisualizer, Rgba, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "execution"))
from visualize_arm_and_hand import (  # noqa: E402
    LEAP_URDF, UR10E_DESCRIPTION_ROOT, UR10E_URDF, compute_X_flange_leapbase, compute_X_leapbase_F)

NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
RZ = RigidTransform(RotationMatrix.MakeZRotation(np.pi))  # controller base <-> URDF base_link
BOARD_SIZE = (9 * 0.115, 4 * 0.115)  # ChArUco: 9x4 squares of 115mm


def triad(meshcat, path, length=0.06, radius=0.003):
    for axis, rgba in zip(range(3), (Rgba(1, 0, 0, 1), Rgba(0, 1, 0, 1), Rgba(0, 0, 1, 1))):
        X = RigidTransform([0, 0, 0])
        if axis == 0:
            X = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2, 0, 0])
        elif axis == 1:
            X = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2, 0])
        else:
            X = RigidTransform([0, 0, length / 2])
        meshcat.SetObject(f"{path}/{'xyz'[axis]}", Cylinder(radius, length), rgba)
        meshcat.SetTransform(f"{path}/{'xyz'[axis]}", X)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--theta", type=float, default=0.0, help="hand mount rotation (same as the grasp pipeline)")
    ap.add_argument("--move-s", type=float, default=2.0, help="animation seconds per move")
    ap.add_argument("--hold-s", type=float, default=1.0, help="animation seconds held at each pose")
    args = ap.parse_args()

    Q = np.load(args.session_dir / "replay_joints_elbow_up.npy")
    T_poses = np.load(args.session_dir / "poses.npz")["T_base_ee"]
    calib = np.load(args.session_dir / "hand_eye_calibration_ur10e_result.npz")
    T_ee_cam, T_base_board = calib["T_ee_cam_wrist"], calib["T_base_board"]

    # hand mount, same composition as visualize_full_grasp_pipeline.py
    tb = DiagramBuilder()
    tplant, _ = AddMultibodyPlantSceneGraph(tb, time_step=0.0)
    tleap = Parser(tplant).AddModels(LEAP_URDF)[0]
    tplant.Finalize()
    tctx = tplant.GetMyContextFromRoot(tb.Build().CreateDefaultContext())
    q0 = np.zeros(tplant.num_positions(tleap))
    q0[0] = 1.0
    tplant.SetPositions(tctx, tleap, q0)
    X_flange_leapbase = compute_X_flange_leapbase(compute_X_leapbase_F(tplant, tctx, tleap), args.theta)

    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().Add("ur_description", UR10E_DESCRIPTION_ROOT)
    ur = parser.AddModels(UR10E_URDF)[0]
    leap = parser.AddModels(LEAP_URDF)[0]
    plant.WeldFrames(plant.GetFrameByName("flange", ur), plant.GetFrameByName("leap_hand_base", leap), X_flange_leapbase)
    plant.Finalize()
    vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(context)
    ur_idx = [plant.GetJointByName(n, ur).position_start() for n in NAMES]

    # board (controller base frame -> Drake world) and a thin plate for it
    X_W_board = RZ.inverse() @ RigidTransform(T_base_board)
    meshcat.SetObject("/board", Box(BOARD_SIZE[0], BOARD_SIZE[1], 0.005), Rgba(0.95, 0.95, 0.95, 0.9))
    meshcat.SetTransform("/board", X_W_board @ RigidTransform([BOARD_SIZE[0] / 2, BOARD_SIZE[1] / 2, 0]))
    board_center_W = (X_W_board @ RigidTransform([BOARD_SIZE[0] / 2, BOARD_SIZE[1] / 2, 0])).translation()
    triad(meshcat, "/camera")
    meshcat.SetObject("/camera_to_board", Cylinder(0.002, 1.0), Rgba(1, 0.6, 0, 1))

    def set_arm(q):
        x = np.zeros(plant.num_positions())
        x[ur_idx] = q
        plant.SetPositions(pctx, x)

    def camera_pose(q):
        set_arm(q)
        X_W_ee = plant.CalcRelativeTransform(pctx, plant.world_frame(), plant.GetFrameByName("tool0", ur))
        return X_W_ee @ RigidTransform(T_ee_cam)

    def beam(X_W_cam):  # thin cylinder from the camera to the board center
        a, b = X_W_cam.translation(), board_center_W
        d = b - a
        L = np.linalg.norm(d)
        z = d / L
        x = np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0])
        x /= np.linalg.norm(x)
        R = RotationMatrix(np.column_stack([x, np.cross(z, x), z]))
        return RigidTransform(R, (a + b) / 2), L

    # check the joint targets reproduce the recorded tool poses
    for i, (q, T) in enumerate(zip(Q, T_poses), 1):
        set_arm(q)
        X = RZ @ plant.CalcRelativeTransform(pctx, plant.world_frame(), plant.GetFrameByName("tool0", ur))
        err = np.linalg.norm(X.translation() - T[:3, 3]) * 1000
        X_cam = camera_pose(q)
        dist = np.linalg.norm(board_center_W - X_cam.translation())
        view = np.degrees(np.arccos(np.clip((X_cam.rotation().matrix()[:, 2]) @ (board_center_W - X_cam.translation()) / dist, -1, 1)))
        print(f"pose {i:2d}: tool0 vs recorded {err:.2f}mm, camera {dist*100:.0f}cm from board center, "
              f"board center {view:.0f}deg off the camera's optical axis")

    # record: hold at each pose, joint-linear moves between them
    fps = 30
    vis.StartRecording(set_transforms_while_recording=False)
    rec = vis.get_mutable_recording()
    frames, t = [], 0.0
    for i in range(len(Q)):
        if i > 0:
            for k in range(1, int(args.move_s * fps) + 1):
                frames.append(Q[i - 1] + (Q[i] - Q[i - 1]) * k / (args.move_s * fps))
        frames += [Q[i]] * int(args.hold_s * fps)
    for k, q in enumerate(frames):
        t = k / fps
        context.SetTime(t)
        set_arm(q)
        diagram.ForcedPublish(context)
        fr = rec.frame(t)
        X_cam = camera_pose(q)
        rec.SetTransform(fr, "/camera", X_cam)
        Xb, L = beam(X_cam)
        rec.SetTransform(fr, "/camera_to_board", Xb)
    vis.StopRecording()
    vis.PublishRecording()
    print(f"\n{len(Q)} poses, {len(frames) / fps:.0f}s animation -- open the URL above and press play "
          f"(Animations panel). The orange line runs from the wrist camera to the board center. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
