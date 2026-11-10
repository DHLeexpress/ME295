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
  real_robot.py    RTDERobot (urx-style calls on ur_rtde, hard speed caps) + background grasp estimator
  grippers.py      finger profiles: ME295 blade + skin fingers, bare 2F-140 (widths, TCP, tips, envelope)
  motion_check.py  slow joint moves: speed of every point, table and self-collision along the path
  sim/tabletop.py  simulated tabletop cell (random objects) + the top-down view configuration Q_VIEW
  look.py, shelf.py, pipeline.py   bookshelf perception for the librarian task (run_librarian.py --perception)
  charuco.py       the calibration board, for a live calibration check
  scripts/         zed_check.py, zed_live.py, real_session.py (real robot), grasp_ablation.py, grasp_real.py,
                   eval_sim_perception.py
  zed_calibration_live/  live calibration check / recapture (side folder; zed_calibration/ stays as delivered)
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

## 3. Real robot: slow, checked picks (`scripts/real_session.py`)
```bash
python vision/scripts/real_session.py status                       # read-only: remote mode, joints, TCP, safety
python vision/scripts/real_session.py goto-view --v-max 0.04 [--execute]
sg zed -c "python vision/scripts/real_session.py observe"          # objects on the table (read-only)
sg zed -c "python vision/scripts/real_session.py pick-all --pick bottle:0.177:-0.465 --pick apple:0.265:-0.597 \
          --pick cylinder:-0.010:-0.490 --v-max 0.04 --execute"   # per-object grip force, wrist-ZED video
python vision/scripts/real_session.py rehearse                     # the same sequence in MuJoCo, from the real joints
```
Nothing moves without `--execute`.

**Safety layers:**
1. Joint moves are planned so that the fastest point of the arm, gripper and camera stays at 95 % of `--v-max`
   (`motion_check.py`). They are checked against the table and for self-collision on the MuJoCo collision model
   (relative to the start pose).
2. Every `speedL` twist passes the barrier filter (`safety.py`): table, workspace, and a cap on the speed of every
   tool point. `RTDERobot.speedl` adds a hard cap.
3. A monitor thread stops everything if the flange moves faster than 2.5 × `--v-max` or the safety mode leaves
   NORMAL.
4. Before each pick: two looks 1.5 s apart must agree (nobody is placing objects), nothing tall may reach into the
   view, and the pre-grasp, grasp and lift poses are checked with IK.

**Gripper:** `--gripper bare` (default on the real robot) is the plain 2F-140 with standard fingertips
(`grippers.py`). `blade` is ME295's skin fingers, the simulator's model.

**2026-11-09:** at ≤ 4 cm/s, with the scripted (open-loop) controller, all three picks succeeded. Each object was
lifted 5 cm, held 2 s and put back.

| object | force per finger | why |
|---|---|---|
| bottle | 15 N | assumed full, 0.55 kg, μ 0.35 |
| apple | 10 N | the gripper's minimum; needs 1.6 N |
| cylinder | 10 N | thin-walled container |

Videos stay local (`results/vision/real_session/`, git-ignored); a 2× edit is `docs/media/real_picks_wrist.mp4`.

**Findings:**
- **The gripper's fingertips were measured with the camera.** The mounted gripper had no blade fingers, so the blade
  model would have closed the jaws 7 cm too high.
- **A grey textureless cone has no stereo depth** at 0.75 m. It needs RGB segmentation or a closer look.
- **A `moveJ` sent while `ur_rtde` is still in `speedL` mode is ignored.** Call `speedStop()` first.
- **Someone arranged objects during one attempt.** That is why the scene check exists.

## Known limits
- **ZED SDK 5.4 needs the `zed` group.** Until you log in again after being added, run with `sg zed -c ...`.
  `pyzed` installs into the venv with `/usr/local/zed/get_python_api.py`. Pin OpenCV < 5: 5.0's Python API dropped
  `calibrateHandEye`.
- **Calibration.** The 2026-10-13 hand-eye calibration (solved with the LEAP hand) was re-checked on 2026-11-09 with
  the 2F-140: 15 new poses agree to 6.1 ± 0.9 mm / 0.32°, so it is kept
  (`zed_calibration_live/captures/20261109_205947/VALIDATION.md`). The SDK now reports fx = 726.8 px, against 770.9
  in the bundle's README.
- **Finger sets.** Two profiles exist in `grippers.py` (blade skin fingers, bare 2F-140). Another finger set needs
  its own profile: widths, TCP, tips and envelope.
- **Only the scripted controller has run on hardware.** PBVS and MPC are validated in the simulator only, so their
  near-object behaviour on real depth, with the occlusion guard, is untested.
- **The simulator's ZED is an ideal stereo camera with a simple noise model.** Real NEURAL depth blurs edges:
  measure with `zed_check.py` before trusting the 18 mm jaw clearance.
- **One object at a time.** `segment_objects` returns all of them, largest first; the grasp scripts take the first.
