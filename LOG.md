# Running log

Newest first. One entry per working session: **what we did → result → next**. Bottlenecks are tagged **[block]**.

## 2026-10-16 (Fri) · Dohyun
- Created the repo. Python ≥3.10 with numpy + matplotlib only. Raw sensor data, recordings and internal lab material are git-ignored.
- Wrote the [related-work note](docs/related_work.md) (3 core papers + 9 adjacent) and the [draft proposal](docs/proposal_draft.md) for the A+B+C direction.
- Implemented the friction-margin grasp quality Q and the gentlest-closure helper from our internal grasp-quality slide (`src/shear_agent/grasp_quality.py`), with unit tests.
- Built a toy 1-D slip simulator to set up the evaluation harness before hardware time: 4 controllers (fixed 15 N, fixed 60 N, slip-reactive, shear-margin) on 500 random books, plus a latency sweep.
  - Result (synthetic): shear-margin had 0 % failed pulls at 16.7 N mean grip, vs 60 N for the strong fixed grip. Mean slip was 1.2 mm vs 3.9 mm for slip-reactive.
  - First version (no slip reflex, fixed 130 ms lead) had more failed pulls than slip-reactive at ≥150 ms latency. Two lessons, both now in the controller:
    1. During gross slip, measured shear saturates at the friction limit and under-reports the load, so a slip reflex is needed.
    2. The feed-forward lead must match the loop delay (sensing latency + gripper lag).
  - Next: replace the toy plant with logged hardware data.
- Defined the agent's tactile tool schemas (`contact_state`, `shear_margin`, `incipient_slip`, `set_margin`, `execute_skill`). An offline dry run works: pull → incipient slip → raise Q\* → regrasp.
  - Wrote `scripts/check_llm_api.py` (one real tool-use round trip). **[block]** API key not verified yet.
- Made the hero figure (`scripts/make_hero.py`) and wrote Lab Report 1.

## 2026-10-15 (Thu) · Dohyun
- Reviewed our internal lab deck (tactile gripper v1, pipeline timing, open questions) and the course lab guideline.
- Literature scan: slip detection and grip-force control; LLM-parameterized grasping; visuo-tactile policies and tactile foundation models; book and shelf manipulation.
- Brainstormed 5 directions:
  - A: shear-margin control
  - B: book extraction
  - C: agent with tactile tools
  - D: visuo-tactile imitation learning
  - E: pretrained optical tactile encoders
- **Chose A+B+C.** D stays as a stretch comparison. E is dropped because of the domain gap from optical tactile images.
  - B follows our last discussion with Prof. Burdick: a thin, shear-sensing parallel-jaw finger suits pulling a book from a shelf, where a full hand is bulky.
- The literature scan and repo scaffolding were AI-assisted (Claude).

## Context carried in from earlier lab work (from the internal deck)
- **Tactile gripper v1:** thin tactile skin on one Robotiq 2F-140 finger, with a dummy finger on the other side.
- **Real-time pipeline:** sensor frame → strain (e_xx, e_yy, e_xy, shape) → ROS → UR10e PC, ~50 ms per cycle (~16 fps).
  - Link test: a 10×10 double matrix goes at ~20 000 msg/s, a single .bmp at ~5 fps. Images are the slow part.
- **UR10e tests:** pushing tests on flat, inclined and compliant surfaces. Shear-only tactile control of the UR10e is in progress.
- **[block]** The new acquisition hardware for double-sided sensing has shipped, ETA ~2 weeks per the latest update.
