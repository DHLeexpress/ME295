# Which VLA is the state of the art to compete against? (survey, 8 Nov 2026)

**Answer: π0.5 (Physical Intelligence, openpi), fine-tuned on our robot's demonstrations.** Optional second
baseline: GR00T N1.7 (NVIDIA). This is the model this folder runs (`pi05_librarian_lora`).

Why, in one paragraph: several open models released in 2026 beat π0.5 on benchmarks *their own authors* chose
(G0.5, MolmoAct2, X-VLA, Xiaomi-Robotics-0, LingBot-VLA 2.0). In *independent* 2026 evaluations, fine-tuned π0.5
still wins most head-to-head comparisons on single-arm robots, and it is the backbone and the vision-only baseline of
nearly every 2026 tactile/force-VLA paper, i.e. of the rivals we compete with. A reviewer will expect "fine-tuned
π0.5 (vision only)" as the baseline that our skin has to beat; anything else invites "why not π0.5?".

Verification key: **✓** = checked against the primary source in this session (arXiv abstract/HTML, GitHub, source
code); **S** = from the survey's reading of the primary source, not re-checked. Dates are arXiv v1.

## 1. Open-weight candidates

| Model | arXiv (v1) | Weights / license | Released checkpoints (action space) | Why (not) the baseline |
|---|---|---|---|---|
| **π0.5** (Physical Intelligence) | 2504.16054 (22 Apr 2025) | openpi: code Apache-2.0, weights Gemma terms ✓ | `pi05_base` (pre-training mix includes a single-arm **UR5e**: `assets/ur5e` norm stats ✓), `pi05_droid` (Franka, 7 joint velocities + gripper, 15 Hz ✓), `pi05_libero` ✓ | **Primary.** Wins most independent tests (§2); base of the tactile rivals (§4); LoRA fine-tuning > 22.5 GB ✓ |
| GR00T N1.7 (NVIDIA) | no paper (N1: 2503.14734) S | Apache-2.0 code, NVIDIA Open Model License S | base, `-DROID`, `-LIBERO`, `-SimplerEnv` S | Optional second: different architecture; leads on SO-101 in RoboColosseum ✓ |
| MolmoAct2 (AI2) | 2605.02881 (4 May 2026) S | Apache-2.0, data released S | `-DROID` (Franka joints), `-SO100_101`, `-LIBERO` S | Fully open; loses to π0.5 on all four arms in RoboColosseum ✓ |
| X-VLA (Tsinghua AIR et al.) | 2510.10274 (11 Oct 2025) S | Apache-2.0 S | LIBERO / WidowX / Google robot / CALVIN / RoboTwin (EE6D) S | Strong on SimplerEnv (S); behind π0.5 on RoboTwin-2.0 leaderboard, LIBERO-Plus, real robots S |
| G0.5 (Galaxea) | 2608.11739 (12 Aug 2026) S | **non-commercial**, gated S | base, droid, libero, so101, robotwin S | Best self-reported LIBERO/RoboTwin; 16.0% vs π0.5 84.0% on Franka in RoboColosseum ✓ |

Not shortlisted (S): LingBot-VLA 1.0/2.0 (bimanual focus), Xiaomi-Robotics-0 and Hy-Embodied-0.5 (no single-arm
results), RDT2 (zero-shot only with its own gripper/camera), Qwen-VLA (no weights found); open *world-action models*
(DreamZero 2602.15922, Cosmos3-Nano-Policy, FLUX 3 Action) top DROID-style leaderboards but are 7-16B parameters with no
independent fine-tuning evidence yet. Closed / not reproducible: Gemini Robotics 1.5 (2510.03342) and 2, π\*0.6
(2511.14759) and π0.7 (2604.15483) (hosted API only, no weights), GR-3 (2507.15493), GR00T N2 (previewed) — S.

## 2. Independent head-to-head evidence (2026)

| Evaluation | Result | |
|---|---|---|
| **RoboColosseum** (2611.02497, 2 Nov 2026), fine-tune track, Table 2, mean full-success over 5 tasks | Franka (DROID): **π0.5 84.0**, MolmoAct2 54.4, GR00T N1.7 42.4, LingBot-VLA v2 40.8, G0.5 16.0. Bimanual YAM: **π0.5 93.6**, G0.5 75.2, MolmoAct2 49.6. SO-101: **GR00T N1.7 82.4**, π0.5 77.6 | ✓ |
| RoboTwin 2.0 official leaderboard (updated 30 Oct 2026), clean / randomized | π0.5 70.66 / 46.0 > X-VLA 67.96 / 20.95 > Xiaomi-R0 62.86 / 18.22 | S |
| LIBERO-Plus (robustness) | π0.5 85.7 > Cosmos Policy 82.2 > X-VLA 71.4 (2603.22078 T4); π0.5 85.3 > GR00T N1.7 75.1 (2609.25562 T3) | S |
| ArmnetBench (2607.24481 T5) / RoboDojo-Real (2608.09892 T4) | π0.5 47.6 > GR00T 29.4 > MolmoAct2 18.9 / π0.5 22.9 > X-VLA 7.6 > GR00T 5.9 | S |
| Where π0.5 is *not* best | SimplerEnv (X-VLA, GR00T N1.6), RoboCasa with frozen VLM (GR00T N1.7 64.1 vs π0.5 34.9, 2608.23478), SO-101 (GR00T) | S |
| LIBERO | saturated (~97-99% for all top models) — not discriminative | S |

## 3. Zero-shot use on a UR10e

No released open checkpoint targets a UR10e. `pi05_droid` (a widely used open zero-shot generalist; RoboArena 1608,
behind the 14B world-action model DreamZero at 1735, S) outputs
Franka joint-velocity commands (`desired_q = measured_q + 0.2 v` per 15 Hz step, DROID's `robot_ik_solver.py` ✓).
We map them through a virtual Franka (`baseline/policies/adapters.py`); the result is a floor, not a fair baseline.
Base checkpoints are documented for fine-tuning only (openpi README ✓).

## 4. The rivals: tactile / force VLAs and what they compare against

| Paper | arXiv (v1) | Built on | Vision-only baseline | Pull / extraction-like task: result (theirs vs baseline) | |
|---|---|---|---|---|---|
| **TacVLA** | 2603.12665 (13 Mar 2026) | **openpi `pi05_base` + LoRA** | **fine-tuned π0.5** | in-box picking 70% vs 10%; disassembly avg 83.75% vs 63.75% (Table II) | ✓ |
| TaF-VLA | 2601.20321 (28 Jan 2026) | π0.5 + tactile-force adapter | π0.5 | power-bank extraction from a friction-fit slot 66.7 vs 33.3 | exists ✓, numbers S |
| FTP-1 | 2606.13102 (11 Jun 2026) | π0.5 + tactile expert | π0.5 | flip book 85 vs 65; pull-out key (sim) 48 vs 38 | S |
| SimpleTouch | 2610.02784 (2 Oct 2026) | π0.5 + tactile expert | π0.5, FTP-1 | pull-out key (sim) 62 vs 36 (π0.5) | S |
| Visible Touch | 2609.14156 (12 Sep 2026) | π0.5 (`pi05_droid` + LoRA), miniVLA | plain π0.5 / miniVLA | contact rendered into the image; ~+30 pts for π0.5 on 4 real tasks; sim on LIBERO with **MuJoCo** contact forces | ✓ abstract, details S |
| ForceDelta-VLA | 2609.18242 (16 Sep 2026) | ForceVLA | π0.5, TA-VLA | plug removal 80 vs 50 | S |
| Earlier: ForceVLA 2505.22159, Tactile-VLA 2507.09160, VLA-Touch 2507.17294 (GelSight on a **2F-140**), TA-VLA 2509.07962 | 2025 | π0 / RDT | π0 / RDT | Tactile-VLA charger pull-out 90 vs 40 | S |

Take-aways for us: (1) in 2026 the rival's baseline is *fine-tuned vision-only π0.5*, usually LoRA on `pi05_base`;
(2) nobody pulls books out of a packed shelf with a VLA; (3) adding touch naively can hurt (ForceLDM 2611.03294 reports
raw torque input lowering GR00T 20.0 → 17.5, S), so the comparison must keep data and backbone identical.

## 5. Prior work on pulling books out of shelves (no VLA)

Tomizawa et al., ICRA 2003 (flat fingers slide into the gaps beside a book — closest to our blades); Prats et al. 2008
(UJI librarian robot, vision/force); Morales et al. 2007 (tilting with tactile sensing); Trixi (2210.10110, tilts the
book from the top; hook dropped for lack of headroom); LAX-RAY / SLAX-RAY (2011.11696, 2201.08968); pre-grasp pivoting
(2305.04654, sim); tactile slip detection while a UR5e slides a book (2303.00935); ViSk (2410.17246, bookshelf
retrieval 7/10 vs 4/10, skin tore on book edges); force-aware retraction from clutter (2508.19476). Benchmarks place
books but none covers tight extraction (RLBench, LIBERO, VLABench `select_book`, BEHAVIOR-1K); DexVerse
`TakeBookOffShelf` (2607.08751) supports a UR10e. (S)

## 6. What this baseline therefore does

- Model: openpi π0.5, `pi05_base` weights, LoRA (`gemma_2b_lora` + `gemma_300m_lora`), the same recipe TacVLA uses as
  its vision-only rival, and openpi's UR recipe (joint state + gripper, absolute joint targets → delta actions).
- Data: scripted-expert demonstrations in our MuJoCo cell (`baseline/data/`), with the skin and F/T recorded but not
  given to the model, so the tactile follow-up can be trained on *the same episodes and backbone*.
- Also reported: zero-shot `pi05_droid` (floor) and the privileged scripted expert (ceiling).

## References (first author et al., title, year, arXiv)

Black+ π0 (2024) 2410.24164 · Physical Intelligence π0.5 (2025) 2504.16054 · Pertsch+ FAST (2025) 2501.09747 ·
NVIDIA GR00T N1 (2025) 2503.14734 · Zheng+ X-VLA (2025) 2510.10274 · Fang+ MolmoAct2 (2026) 2605.02881 · Liu+ G0.5
(2026) 2608.11739 · Nguyen+ RoboColosseum (2026) 2611.02497 · Zhang+ (TacVLA) Contact-Aware Tactile Fusion for Robust
VLA Manipulation (2026) 2603.12665 · Huang+ TaF-VLA (2026) 2601.20321 · Yuan+ FTP-1 (2026) 2606.13102 · Yang+
SimpleTouch (2026) 2610.02784 · Dogan+ Visible Touch (2026) 2609.14156 · Dong+ ForceDelta-VLA (2026) 2609.18242 · Yu+
ForceVLA (2025) 2505.22159 · Huang+ Tactile-VLA (2025) 2507.09160 · Bi+ VLA-Touch (2025) 2507.17294 · Zhang+ TA-VLA
(2025) 2509.07962 · Khazatsky+ DROID (2024) 2403.12945 · Atreya+ RoboArena (2025) 2506.18123 · Wieczorek+ Trixi (2022)
2210.10110 · Pattabiraman+ ViSk (2024) 2410.17246 · Physical Intelligence π\*0.6 (2025) 2511.14759 · Gemini Robotics
1.5 (2025) 2510.03342.
