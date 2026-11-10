# Grasp controllers for the ablation: what and why (short survey)

Goal: from a top-down view of the table, grasp one unknown object with the UR10e + 2F-140 (blade fingers), zero-shot
(no learning), so that perception, calibration and control can be tested on hardware separately from the VLA work.
The ablation keeps everything fixed except the approach strategy.

*Sources: well-established references, cited from memory; no new literature search was run for this note. Check
venues and years before citing them in a report.*

## Shared pipeline (not ablated)
- **Perception**: single-view RGB-D, table plane fit plus clusters on the plane, top-view footprint (minimum-area
  rectangle). This is the classic geometric tabletop segmentation used as the front end of point-cloud grasp
  planners. Learned 6-DoF grasp detectors could replace it later; their recipe is "point cloud, grasp poses, open-loop
  execution": GPD (ten Pas et al., IJRR 2017), Dex-Net 2.0 (Mahler et al., RSS 2017), GraspNet-1Billion / AnyGrasp
  (Fang et al., CVPR 2020 / T-RO 2023), Contact-GraspNet (Sundermeyer et al., ICRA 2021).
- **Grasp**: top-down parallel-jaw grasp across the narrow side of the footprint, at a fixed depth below the top,
  never so low that the blade tips reach the table.
- **Safety**: every velocity command passes a control-barrier-function filter: the tool points' height above the
  table, the workspace box, and speed caps (Ames et al., "Control Barrier Functions: Theory and Applications", ECC
  2019). Discrete moves are checked as sampled joint paths, like the lab's `ta_utils.joint_path_safety_filter`.
- **Gripper**: open to the width + 2 × 18 mm, close at a fixed Robotiq force, lift 10 cm, hold 1 s.

## Ablated: how the arm gets to the grasp
| | controller | idea | reference point |
|---|---|---|---|
| 1 | **scripted (open loop)** | look once, plan, execute a time-parametrised path to pre-grasp and down | sense-plan-act, as Dex-Net / GPD grasps are executed |
| 2 | **PBVS (closed loop)** | re-estimate at 5 Hz, filter, servo the TCP to the estimate (align, then descend); freeze when the object gets too close to see | position-based visual servoing (Chaumette & Hutchinson, IEEE RAM 2006); closed-loop grasping that stops updating at the camera's minimum range (GG-CNN, Morrison et al., RSS 2018; Viereck et al., CoRL 2017) |
| 3 | **MPC (OSQP)** | linear receding-horizon MPC on the TCP: speed and acceleration limits, never below the grasp height, a soft approach funnel; re-planned on every estimate | constrained visual-servoing MPC (Allibert, Courtial & Chaumette, T-RO 2010) in its simplest position-based form; OSQP (Stellato et al., Math. Prog. Comp. 2020) |

Candidates left out for now: IBVS (image-space servoing; the gripper hides the object at the end, so it would need
the same freeze); sampling MPC / MPPI (Williams et al., ICRA 2017), which pays off once contact or the skin is in the
model; and learned closed-loop policies (the VLA baseline in `baseline/`).

## Design notes from the simulator
- **The wrist ZED looks across the gripper.** Its optical axis meets the tool axis 0.284 m from the flange, where
  the pads are. The fingers fill the lower ~45 % of the image, so the view configuration puts the workspace in the
  free band.
- **Closing the loop needs an occlusion guard.** Below ~10 cm above the grasp, the fingers hide part of the object.
  The partial view shifts the centroid by 2–3 cm and halves the width. The goal filter therefore rejects estimates
  that lose points or change the object's size, and freezes after two rejections. Without the guard, PBVS ended
  11 mm off target and pushed the object.
- **A hand-eye rotation error tilts the table** as perception sees it (about 1 cm per 0.5° across the view), so the
  table is fitted as a plane, not as a single height.
