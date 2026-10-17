# Draft project proposal (v0, Oct 16 2026). Locked version due with LR2 (Nov 1)

**Working title.** Shear-Aware Agentic Grasping: a parallel-jaw gripper with a thin tactile skin that feels shear, squeezes just enough, and pulls books off a shelf.

**Team.** Dohyun ([@DHLeexpress](https://github.com/DHLeexpress)): robot/ROS 2 integration, controllers, agent layer, experiments. Jihoon ([@jhahn2](https://github.com/jhahn2)): tactile sensor, acquisition, strain/shear estimation, calibration. Shared: testbed, evaluation, writing.

## Problem
Parallel-jaw grippers are the workhorse of real deployments, yet they cannot sense contact state. They know the jaw position and a commanded force, not how close each contact is to slipping. Classical grasp analysis (friction cones, grasp quality) assumes contact forces that ordinary grippers cannot observe, and vision cannot see them. That is especially true in confined, occluded spaces such as a tightly packed bookshelf. Camera-based tactile fingertips do measure shear, but they are thick, and a dexterous hand is bulky. Our tactile skin is thin and measures the shear-strain field at the contact in real time. That is the variable classical theory needs and the gripper lacks.

## Research question
Does real-time shear sensing on a thin parallel-jaw finger, used **(A)** inside a friction-margin grip controller and **(C)** as tools an LLM agent can call, make **(B)** confined-space manipulation such as pulling one book out of a packed shelf more reliable and gentler than fixed-force, slip-reactive, LLM-prior and vision-only baselines?

## Hypotheses
- **H1 (A).** Holding the margin Q ≥ Q\* with measured shear matches the failure rate of a high fixed force at a much lower mean grip force, with fewer and shorter slips than slip-reactive control.
- **H2 (B).** Tactile events (contact, tilt onset, incipient slip, jamming against neighbours) raise book-extraction success and reduce disturbance to neighbouring books, relative to a scripted vision-only tilt-and-grasp.
- **H3 (C).** The same LLM agent detects and recovers from more failures when it can call tactile tools than when it only has vision tools.

## Approach
- **A: control.** Q = min_i (1 − ‖f̂_t,i‖ / (μ̂_i f̂_n,i)), from our internal grasp-quality slide.
  - f̂_t comes from the skin's measured shear, calibrated against the UR10e wrist F/T sensor or a load cell.
  - f̂_n comes from the gripper force/current.
  - μ̂ starts from a prior (an LLM guess, DeliGrasp-style) and is lowered online at slip onsets.
  - Command the gentlest closure with Q ≥ Q\*, plus a slip reflex.
- **B: application.** A shelf testbed with adjustable packing, and a book set that varies thickness, weight and cover friction. The primitives are *tilt* (pull the top edge), *pinch* (fingers into the gaps) and *pull*.
- **C: agent.** An LLM agent with tool calls: `contact_state`, `shear_margin`, `incipient_slip`, `set_margin`, `execute_skill`. It chooses the primitive sequence, watches Q while pulling, and triggers recovery (regrasp, re-tilt, push a neighbour aside).

## Baselines and comparisons
| ID | Method | Role |
|---|---|---|
| B0 | Robotiq fixed force (low / high) | gripper default |
| B1 | normal-force / motor-current threshold | "force but no shear" |
| B2 | slip-reactive regrasp (Dong et al. 2017 style) | classical tactile |
| B3 | DeliGrasp-style LLM-prior force, no touch | agentic, open loop |
| B4 | scripted vision-only tilt → grasp → pull (Trixi-style) | task baseline for B |
| B5 | our agent with vision-only tools | ablation for C |
| S | vision-only Diffusion Policy vs RDP-style visuo-tactile policy, ~50 teleop demos | scaled-up learned comparator (stretch) |

**Metrics:** extraction success; peak and mean grip force; slip events and slip distance; neighbour-book displacement; time to extract; failure-detection and recovery rate; agent latency and number of LLM calls.

## Plan (one row per lab report)
| Weeks | Report | Milestone |
|---|---|---|
| 1–2 | LR1, Oct 18 | literature, repo/env, toy evaluation harness, this draft |
| 3–4 | LR2, Nov 1 | lock question; shear → force calibration; gripper re-command latency; 1-page proposal |
| 5–6 | LR3, Nov 15 | A on hardware (one-sided): shear-margin vs B0–B2 lift-and-hold |
| 7–8 | LR4, Nov 29 | two-sided sensing; agent tools on hardware; end-to-end book extraction; failure modes |
| 9–10 | LR5 | B3–B5 ablations (S if feasible), reliability audit, write-up |

## Risks and fallbacks
- **Second sensing channel arrives late.** Run A and C with one sensed finger (symmetric-contact assumption), and run B with one sensed finger.
- **About 50 ms loop.** Pull slowly; compute strain on the sensing PC and stream features instead of images. The toy latency sweep in `results/toy_benchmark.md` shows the cost.
- **Normal strain is hard to recover from the skin's raw images.** Use gripper force for f_n and use the skin for shear only.
- **Robotiq exposes a force limit + position, not torque control.** Measure the re-command latency; fall back to position-increment squeezing.
- **Finger thickness vs. book gaps.** Measure it and choose shelf packing levels accordingly.
- **Skin interface wears under fast motion.** Use speed limits and keep spare skins.
