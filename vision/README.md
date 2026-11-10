# vision/: wrist ZED Mini → calibrated RGB-D → what to grasp → grasp controllers

Everything vision-related in ME295. It includes the original ZED hand-eye calibration bundle
(`zed_calibration/`, copied from the lab's `ZED/` folder) and runs both in the MuJoCo cell and on the real UR10e.

```
vision/
  camera.py        Intrinsics, RGBDFrame: one frame type for the real and the simulated camera (ZED IMAGE frame)
  wrist.py         WristCamera: the hand-eye calibration (zed_calibration/ active result) + intrinsics; flange_pose(rob)
  zed.py           ZedCamera (ZED SDK / pyzed), UVCStereo (raw stream, no SDK)
  stream.py        CalibratedStream: each frame + the robot pose at its exposure time (RTDE, 250 Hz, read-only)
  sim_zed.py       the calibrated ZED Mini in MuJoCo (real intrinsics, calibrated mount)
  safety.py        table / workspace safety filter: joint-path checks + barrier filter on every velocity command
  tabletop.py      table plane, objects on it, top-down parallel-jaw grasp
  controllers/     grasp ablation: scripted (open loop), PBVS (closed loop), MPC (OSQP)   [docs/controllers.md]
  real_robot.py    RTDERobot (urx-style calls on ur_rtde) + background grasp estimator
  sim/tabletop.py  simulated tabletop cell (random objects) + the top-down view configuration Q_VIEW
  look.py, shelf.py, pipeline.py   bookshelf perception for the librarian task (run_librarian.py --perception)
  charuco.py       the calibration board, for a live calibration check
  scripts/         zed_check.py, zed_live.py, grasp_ablation.py, grasp_real.py, eval_sim_perception.py
  tests/           python -m unittest discover -s vision/tests -t .   (simulator only, ~2 s)
  zed_calibration/ original calibration bundle: README_camera_extrinsic.md, calibration/ (scripts + results),
                   perception/, ur10e_description/
  zed_mini_wrist.json   snapshot of the active calibration + intrinsics (for CI / the simulator)
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate       # Python 3.11 (baseline/ needs tomllib)
pip install -r requirements.txt                          # + vision: opencv-contrib-python ur_rtde osqp scipy
# ZED SDK Python API (needs read access to /usr/local/zed; on this PC: sudo usermod -aG zed,video $USER, log in again)
python /usr/local/zed/get_python_api.py                  # installs pyzed into the active environment
```

## 1. Real-time calibrated vision
```bash
python vision/scripts/zed_check.py --robot     # camera + intrinsics + fps per depth mode; robot TCP check; base-frame cloud
python vision/scripts/zed_live.py              # live: RGB with base axes / table grid / board outline, depth, top view
python vision/scripts/zed_live.py --grasp      # + objects and the planned grasp (no motion)
```
Every frame is `RGBDFrame` with `T_base_cam = T_base_flange(t_image) @ T_flange_cam`. The flange pose comes from
RTDE at 250 Hz and is interpolated at the image's exposure time. Read-only: these scripts never command the robot.

Calibration checks:
- The projected 10 cm grid must lie flat on the table, and the projected board outline (cyan) on the board.
- With the ChArUco board in view, the HUD shows the live board pose against the calibration session's
  `T_base_board`, in mm and degrees.
- The robot's TCP must be the flange (the calibration was solved that way). `RTDEPoses.check_tcp()` compares the
  controller with nominal forward kinematics: ~2 mm is normal, more means another TCP is set.

Recalibrating: `zed_calibration/README_camera_extrinsic.md`. Then point `calibration/active_calibration.txt` at the
new result and run `zed_check.py --snapshot`. Nothing else changes.

## 2. Top-down grasp, zero-shot, three controllers
**View configuration** (`sim/tabletop.py` `Q_VIEW`, degrees): `[74.3, -122.7, 111.4, -85.5, -87.1, 164.5]`.
- The camera is 0.70 m above the workspace centre (0.10, −0.50) and looks straight down.
- The whole workspace (±10 cm × ±7 cm, objects up to 16 cm tall) falls in the image band the gripper leaves free.
- It is elbow up, seeded from calibration pose 16, which the robot has already run. The safety filter passes it,
  with the lowest gripper point 135 mm above the table.

```bash
python vision/scripts/grasp_real.py                      # DRY RUN: look, plan, safety-check the plan; no motion
python vision/scripts/grasp_real.py --controller mpc --execute       # typed "yes"; speed scale 0.5
python vision/scripts/grasp_ablation.py                  # the simulated ablation (below)
```

Simulated ablation: 30 random objects per cell (boxes, cylinders, spheres, 40–90 mm wide, 60–160 mm tall,
50–500 g), stereo noise model, table at z = 0. Results: `results/vision/grasp_ablation/summary.md`.

| success (30 objects each) | nominal | calib 0.75°, 2 mm | calib2 2°, 4 mm | moved 4 cm | fatten (stereo edges) |
|---|---|---|---|---|---|
| scripted (open loop) | 30/30 | 30/30 | 30/30 | **18/30** | 30/30 |
| PBVS (closed loop) | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| MPC (OSQP, funnel) | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |

| goal error mm, median [95th] | nominal | calib | calib2 | moved | fatten |
|---|---|---|---|---|---|
| scripted | 1.3 [3.1] | 7.2 [9.4] | 18.3 [23.7] | 40.0 [41.9] | 1.2 [2.3] |
| PBVS | 1.2 [2.8] | 4.4 [7.7] | 11.4 [15.8] | 1.0 [2.7] | 1.3 [2.0] |
| MPC | 1.4 [3.0] | 4.3 [7.0] | 11.3 [16.2] | 1.1 [2.8] | 1.1 [2.5] |

Gripper-body contact (median over all episodes): scripted 3.9 N, PBVS 0.3 N, MPC 0.1 N. Each grasp takes ~9–10 s.
- **Closing the loop pays when the world or the calibration is wrong.** It recovers a moved object (40 → 1 mm) and
  roughly halves a hand-eye rotation error, because the error scales with the camera's distance and the closed-loop
  controllers look again from closer.
- **The 18 mm per-side jaw clearance absorbs the rest.** Even 18 mm of open-loop error still succeeds on these
  objects: at 2°, the jaws close on the object's edge and pull it centred.
- **PBVS and MPC tie here.** MPC adds speed and acceleration limits and the approach funnel as constraints, not
  logic, which matters once obstacles or tighter clearances enter.
- Videos of the `moved` case: `results/vision/media/grasp_{scripted,pbvs,mpc}_moved.mp4`. Keyframes:
  `grasp_moved_keyframes.jpg`.

Bookshelf perception (`eval_sim_perception.py`, the VLA baseline's held-out suites, 30 cases each):
- With exact or noisy depth, it picks the instructed book in 90/90 cases. Centre error is within ±1 mm, thickness
  +1 mm, gaps −1.4 mm, spine plane 0.0 mm.
- With 2 px stereo edge fattening, the 7 mm gaps read ~3 mm narrower and the narrow suite drops to 9/30. Measure the
  real ZED's edge behaviour before relying on narrow gaps.
- `scripts_sim/run_librarian.py --perception sim` runs the whole librarian task from vision. The book is shelved
  2.4 mm from the slot centre.

## Known limits
- **No ZED SDK access yet.** Depth on hardware and the `pyzed` path are untested until this account can read
  `/usr/local/zed`. The UVC stream works, and the sim gripper silhouette roughly matches the real view (fingertips
  within ~2–4°). That check used a raw, unrectified frame, so it is not a precision test.
- **The calibration was solved with the LEAP hand mounted** (2026-10-13). If the camera bracket moved when the
  2F-140 went on, recalibrate. The live board check shows it.
- **The gripper model is the 2F-140 + blade v1 + PDMS skin.** Another finger set needs its envelope
  (`safety.ENVELOPE_BOXES`) and width / TCP table (`tabletop.GripperSpec`).
- **The simulator's ZED is an ideal stereo camera with a simple noise model.** Real NEURAL depth blurs edges:
  measure with `zed_check.py` before trusting the 18 mm jaw clearance.
- **One object at a time.** `segment_objects` returns all of them, largest first; the grasp scripts take the first.
