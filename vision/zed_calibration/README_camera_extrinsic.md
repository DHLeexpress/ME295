# UR10e wrist ZED Mini: camera extrinsic + how to recalibrate

## The current extrinsic

File: `calibration/captures_ur10e/20261013_121056/hand_eye_calibration_ur10e_result.npz`
(this is the one pinned in `calibration/active_calibration.txt`; every script in the pipeline reads
that file to decide which calibration is active).

Keys in the npz:

| key | meaning |
|---|---|
| `T_ee_cam_wrist` | 4x4, camera pose in the tool frame: `p_ee = T_ee_cam_wrist @ p_cam` |
| `T_base_board` | 4x4, where the ChArUco board was during that session (robot base frame) |
| `num_poses` | 18 |

Value (metres):

```
T_ee_cam_wrist =
[[-0.99852  0.05090  0.01925  0.02460]
 [-0.05441 -0.93420 -0.35257  0.10156]
 [ 0.00004 -0.35309  0.93559  0.01439]
 [ 0.       0.       0.       1.     ]]
```

Frames, exactly:
- **ee** = the UR controller's TCP frame with the TCP set at the tool flange (tool0), as returned by
  `urx` `robot.get_pose()`. Base = the controller's base frame. (If you work from the ROS/Drake URDF
  instead: controller pose = `RotZ(pi) @ FK(URDF tool0)`; verified to 0.17 mm / 0.06 deg.)
- **cam** = the ZED Mini **left** camera, ZED SDK default IMAGE coordinate system (x right, y down,
  z forward), i.e. the frame `MEASURE.XYZ` point clouds come in. Camera: ZED Mini SN 18982088,
  HD720, factory intrinsics fx 770.9, fy 769.5, cx 621.1, cy 362.8.
- To put a point cloud in the robot base frame: `p_base = T_base_ee @ T_ee_cam_wrist @ p_cam`
  (`perception/fuse_utils.py: fuse_to_base_frame`, `load_calibration`, `newest_calibration`).

Quality of this calibration: 18 poses, board-pose consistency 3.4 +/- 2.1 mm across poses. It
differs by ~0.8 deg from the 2026-10-09 calibration, which is what a bumped camera looks like; if
clouds from different views stop overlapping by more than ~5 mm, recalibrate.

## Recalibrating automatically

Target: the Boston Dynamics Spot ChArUco board, 9x4 squares, 115 mm squares, 90 mm markers,
DICT_4X4_50, legacy pattern. Tape it flat on the table where the arm can see it from many angles.

Environment: python with `urx` + `math3d` (clone https://github.com/Caltech-AMBER/python-urx),
`opencv-contrib` (aruco), `pyzed` (ZED SDK), numpy. Robot pendant in Remote mode, IP 192.168.0.2
(`--robot-ip` to override).

1. **Automatic capture + solve** (the arm drives itself through the recorded poses, captures
   wherever the board is detected, then solves AX = XB with Tsai):
   ```
   cd calibration
   python ur10e_hand_eye_calibration.py --replay captures_ur10e/20261009_100545 --no-hand
   ```
   `captures_ur10e/20261009_100545/replay_joints_elbow_up.npy` holds elbow-up joint targets for
   its 18 poses, so the arm moves with plain joint moves (no controller IK surprises). Useful flags:
   `--confirm-each` (press Enter before every move the first time), `--vel/--acc`, `--settle`.
   Omit `--no-hand` only if the LEAP hand is mounted and its API is installed (it holds the fingers
   out of the camera's view).
   Output: a new `captures_ur10e/<timestamp>/` with `poses.npz`, the images, and
   `hand_eye_calibration_ur10e_result.npz`. The script prints per-pose board consistency; expect
   ~3 mm. Outliers (one pose far from the rest) can be dropped by rerunning the math on the session
   without them.

2. **Make it the active calibration**: write its path into `calibration/active_calibration.txt`
   (one line, path to the npz). Nothing else needs to change.

3. **Re-solve from saved poses only** (no robot/camera): `python ur10e_hand_eye_calibration.py
   captures_ur10e/<session>` recomputes from that session's `poses.npz`.

4. **If the board moves to a new place** the replay poses may not see it: do a manual session once
   (`python ur10e_hand_eye_calibration.py`, freedrive the arm, press `y` per pose, 15-20 poses from
   varied angles and distances), then `python solve_replay_joints.py captures_ur10e/<session>`
   (needs Drake + `ur10e_description/ur10e.urdf`) to create its `replay_joints_elbow_up.npy`, and
   from then on `--replay` that session. `visualize_replay.py` (Drake Meshcat) previews the replay.

## Files in this bundle

```
calibration/ur10e_hand_eye_calibration.py   capture (manual or --replay) + solve
calibration/solve_replay_joints.py          offline elbow-up IK -> replay_joints_elbow_up.npy
calibration/visualize_replay.py             Meshcat preview of a replay (needs the repo's execution/ models)
calibration/active_calibration.txt          pointer to the active result
calibration/captures_ur10e/20261013_121056/ current calibration (poses.npz + result)
calibration/captures_ur10e/20261009_100545/ replay source (poses.npz + replay_joints_elbow_up.npy + result)
perception/fuse_utils.py                    how the calibration is loaded and applied to clouds
perception/capture_pointclouds.py           the 3-view scan script (same frames; hand-hold helper the calibration imports)
ur10e_description/ur10e.urdf                UR10e URDF used by solve_replay_joints.py (meshes: Universal_Robots_ROS2_Description)
```
