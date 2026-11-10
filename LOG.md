# Running log

Newest first. One entry per working session: **what we did → result → next**. Bottlenecks are tagged **[block]**.

## 2026-11-09 (Mon) · Dohyun
- **Explainers for the VLA results table.** A 60 s paper-reel explainer with voice-over and sim footage; the results
  table and the failure-mode table are built on screen as they are read (`baseline/results/media/reel_results_60s.mp4`,
  source in `baseline/reel/`). Two figures: one test case per suite
  (`fig_suites`) and key frames per outcome (`fig_outcomes`), plus a 6-clip GIF. The clips are high-resolution re-runs
  of evaluation test cases (`evaluate.py --media`, `sim/media.py`); they are not counted in the results.
- **Correction:** most "jams" are **push-ins**. In 12 of the 17 episodes, the blade tips push the book 55-64 mm deeper
  into the shelf and the pads never touch it (8 of the 12 are 44-45 mm books). So the inner-face skin is blind to this
  failure; a blade-tip sensor or the F/T during insertion would see it. Only 5 are true jams (pads squeeze, book
  stuck). `analyze_failures.py`, `RESULTS.md` §3.2-3.4.
- **Next:** unchanged: skin-set grip force on the heavy / low-friction suite.

## 2026-11-08 (Sun) · Dohyun
- **VLA baseline in the MuJoCo cell (`baseline/`).** Literature check (arXiv, Nov 2026) → π0.5 (openpi) is the open VLA
  to beat: it wins most independent 2026 comparisons after fine-tuning and is the backbone and vision-only baseline
  of the 2026 tactile-VLA papers (e.g. TacVLA: LoRA on `pi05_base` vs fine-tuned π0.5). `baseline/docs/SOTA_survey.md`.
- **Modular kinematics.** One TOML card per component, as in `model/DESCRIPTION.md` (UR10e → coupling → 2F-140 → blade
  v1 → PDMS skin); FK matches MuJoCo to 1e-15 m. TCP between the pads: 292.3 mm from the flange at a 40 mm grasp,
  moves 23.7 mm as the jaw closes; blade + skin 3.5 mm thick. A new blade is a new card.
- **Result** ("pick up the {colour} book", success %): zero-shot π0.5-DROID 0 / 0 (nominal / randomized; hits the
  shelf, UR protective stop). π0.5 fine-tuned with LoRA on 251 scripted demos: 100 / 72 / 73 / 43 % (nominal /
  randomized / narrow gaps / heavy + μ 0.5); privileged expert 100 / 100 / 100 / 57 %. Fine-tuned failures: jams
  beside 44-45 mm books (it does not size the jaw to the book), shoving neighbours in narrow gaps, slips under the
  fixed 67 N grip. `baseline/RESULTS.md`.
- **Next:** same backbone + same episodes (skin maps and F/T already in the dataset) with touch: first the skin setting
  the grip force on the heavy / low-friction suite.

## 2026-10-17 (Sat) · Dohyun
- **Repo.** Created the public repo [DHLeexpress/ME295](https://github.com/DHLeexpress/ME295) and pointed the local repo at it (`origin`). Next: add Jihoon (@jhahn2) as a collaborator.
- **Hero figure.** Redrew it so the book comes out *sideways*, not lifted. Real shelves leave no headroom, so the gripper approaches horizontally, slides the thin fingers into the gaps, pinches and pulls. Added a top-view inset of the gap showing book | skin | finger | next book, with the shear direction.
- **API handling for a public repo.**
  - Keys: `.env*` is git-ignored (template `.env.example`). A pre-commit hook (`git config core.hooksPath .githooks`) blocks key-shaped strings and `.env` files; tested on a throwaway clone. CI runs the tests, an offline agent episode and a key scan.
  - Client: dependency-free `LLMClient` (Anthropic by default, OpenAI-compatible optional) with a 60 s timeout, up to 4 retries with backoff on 429/5xx/529 that honour Retry-After, token and turn caps, and token-usage reporting. The key is never printed or put into errors.
  - Robot safety: every tool call is schema-validated before it reaches the robot backend (skill enum, 0 ≤ Q\* ≤ 0.8, no unknown arguments).
  - Agent loop + `MockLLM`: `scripts/run_agent.py --mock` runs the full episode offline. Added `insert` to the skills.
  - 23 unit tests pass.
- **Report.** Made a 2-page Word version of Lab Report 1 (Times New Roman) with the revised hero.
- **[block]** API key still not verified (`scripts/check_llm_api.py`).

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
