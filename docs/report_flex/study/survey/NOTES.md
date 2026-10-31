# Who uses MuJoCo flex well: survey notes (flex-usage-survey, 2026-10-30)

Scope: MuJoCo >= 3.0 flex (flexcomp) usage relevant to a 51 x 71 x 1.5 mm PDMS layer (E 1 MPa, nu 0.45/0.49) bonded to a
gripper blade on a UR10e + 2F-140. Our parameters are in the shared context. All sources were opened (HTTP 200, checked
in `urls_status.txt`; local copies under `raw/`, treated as untrusted data and never executed). Every number below is
either quoted from a source (URL/section given) or produced by a script here:

| script | output | what |
|---|---|---|
| `official_examples.py` | `official_examples.json`, `official_examples_table.txt` | settings + 0.5 s stepping of all 3.17.0 `model/flex` examples |
| `checks.py` | `checks.json`, `checks.log` | C1 StVK vs Stable Neo-Hookean (`elastic3d=1`) patch test; C2 integrators; C3 edge equality + young |
| `regimes.py` | `regimes.json` | CFL-like number h c_p / l_min, ours vs others |
| `analysis_s2.py` | `analysis_s2.json` | F1 ratio by load case; our layer's thickness modes, omega*h, vertex masses |

Note that "verified" below means "I opened the source and it says so" or "my script reproduced it on 3.17.0". Claims
made by third parties about their own runs are NOT re-run by me and are marked "(their measurement)".

## 0. Headline findings

1. **Nobody has reported F1 (StVK lambda doubled).** Issue searches ("poisson", "flex lame", "flex lambda", "StVK
   volumetric", "flex poisson ratio stiffness", "lambda twice", "uniaxial strain flex": `raw/issues1.txt`,
   `raw/issues_recheck_s2.txt`) find nothing. The only hint is PR #3621 (Stable Neo-Hookean): "The SNH energy uses
   conventional Young/Poisson Lame parameters ... The previous 3D StVK volumetric normalization is preserved on the
   default path." (https://github.com/google-deepmind/mujoco/pull/3621).
2. **Why everyone misses it** (`analysis_s2.json`, block A): with lambda' = 2 lambda the uniaxial-STRESS modulus changes
   by only x1.0163 at nu = 0.45 (x1.0317 at 0.4, x1.0577 at 0.3, x1.0 at nu = 0), whereas the confined (uniaxial-strain)
   modulus changes by x1.8182 (x1.9608 at 0.49) and the bulk modulus by x1.9310. All validations I found are bending /
   unconfined / nu = 0 tests (MuJoCo's ElasticEnergySolid uses poisson 0; badinkajink's block check is at nu = 0;
   MetalSim's and discussion #1777's are cantilevers). A thin bonded layer under normal load is the worst case.
3. **The fix already exists inside 3.17 (spec-only):** `mjsFlex.elastic3d = 1` (experimental simplified Stable
   Neo-Hookean, changelog 3.15.0 item 5) gives exact patch ratios: uniaxial 1.000000, shear 1.000000, hydrostatic
   0.999997 at nu = 0.45 and 0.49, while default StVK gives 1.818179 / 1.0 / 1.931032 (nu 0.45) and 1.960781 / 1.0 /
   1.986574 (nu 0.49) (`checks.json` C1). The F1 workaround inputs on StVK also give 1.000 (C1). SNH requires the
   discrete integrator, non-interpolated 3D flexes, is not available in MJCF (spec export errors), and its Hessian is
   PSD-projected (PR #3638). Not yet tested by me on the robot.
4. **No shipped example has an elastic `dof="full"` 3D flex on a moving body.** Of the 14 elastic 3D flexes in
   `model/flex` 3.17.0 only `gripper_trilinear.xml` (trilinear, young 1e4, nu 0.1) hangs on moving bodies
   (`official_examples.json`, field `attached_to_moving_body`). Articulated attachments for non-interpolated flex
   elasticity only arrived in 3.15.0 (changelog item 7, PR #3618), with "the existing PSD stiffness approximation; this
   does not add the geometric derivatives of an articulated attachment's Jacobian", "General flexes use the mass
   preconditioner; their convergence can be slower", and "General standard-flex attachments require CG with the
   discrete integrator". Our configuration (F4) is at the frontier of what the code supports, not a beaten path.
5. **Our regime is far outside every other user's**: CFL-like h c_p / l_min = 41.8 (11x15x3, nu .45, true lambda) to
   248.6 (41x57x5, nu .49, 2 lambda) at h = 0.5 ms vs 0.15-0.49 for the official 3D examples (`regimes.json`,
   `official_examples_table.txt`). The layer's first thickness modes are 10.45 kHz (P) and 3.15 kHz (S) at nu .45;
   omega*h = 32.8 and 9.9 at 0.5 ms (`analysis_s2.json`). roqsim measured that the discrete integrator's damping and
   frequency are within 5 % only for omega*h <= 0.3 and that it adds Rayleigh damping equal to one timestep:
   zeta_i = (damping + h) omega_i / 2 (their measurement, MuJoCo 3.14). So at 0.5 ms the layer can only be a
   quasi-static element; its dynamics are numerically over-damped. The only third-party flex with a comparable regime is
   badinkajink/hand's TPU cap (CFL ~240 with 2 lambda, 1 ms, discrete+CG), and it was dropped for cost (1,249 us/step,
   98x the rigid pads; their measurement).
6. **Contact count cap (F3) is confirmed in source**: `flexReducible()` = rigid or interpolated
   (`src/engine/engine_collision_driver.c` l. 572-575); geom-flex pairs get `mjMAXCONPAIR` = 50 (l. 1900); changelog
   3.17.0 item 10. MetalSim (MuJoCo 3.14, before this change) measured cloth falling through a box with the cap of 50 and
   resting with a rebuilt cap of 400/4000 (their measurement; mujoco_warp issue #1709).
7. **What competent robot-mounted soft-pad users actually ship**: either (a) rigid geoms with compliant `solimp`
   calibrated to data (Msornerrrr LEAP PaXini pads), or (b) one-cell `dof="trilinear"` skin pinned at its bottom nodes
   under discrete + CG at 500 us (WowSkin-Lab, after their full-node StVK/SNH variants inverted at ~3.41 s and cost
   292-1160 ms per 10 ms sample; their measurement), or (c) Winkler/lattice pads (badinkajink). None validates a flex
   against an exact elastic solution except badinkajink's nu = 0 block (19.0 vs exact 18.75 um).

## 1. Table

Settings are copied from the opened files. "-" = not set in the file (MuJoCo default). h = timestep.

| # | source | what | dim / dof | cells vs body | young / poisson / damping | radius / margin | contact solref / solimp | h / integrator / solver | pins / moving body | validation |
|---|---|---|---|---|---|---|---|---|---|---|
| O1 | mujoco 3.17.0 `model/flex/press.xml` | two blocks pressed | 3 / full | 64 verts, l_min 200 mm | 1e4 / 0.4 ; 5e4 / 0 ; 0 | 5 mm / 0 | 0.01 1 / .95 .99 .0001 | 1 ms / discrete / Newton | none / no | none |
| O2 | `floppy.xml` | 24x4x4 grid falls on cylinder | 3 / full | 0.1 m cells | 5e4 / 0.2 / 0.002 | 0 | 0.01 1 / .95 .99 .0001 | 1 ms / Euler / CG | none / no | none ("1 ms is quite close to the stability limit", #2652) |
| O3 | `gripper_trilinear.xml` | 2-finger slide gripper with soft caps | 3 / trilinear 1x1x1 | cap.obj, 218 verts | 1e4 / 0.1 / 0.05 | 1 mm | 0.02 1 / .9 .95 .001 | 2 ms / discrete / CG | caps on slide-jointed fingers / yes | none |
| O4 | `mechanisms/cantilever.xml` | quadratic cantilevers | 3 / quadratic 2x1x1 | 441 verts | 8e6-2e7 / 0.25 / 2e-3-4e-2 | 1 mm | defaults | 1 ms / discrete / Newton | pinned end / no | none in file |
| O5 | `gripper.xml`, `softbox.xml`, `sphere_radial.xml` | flex on moving bodies | 3 / full (radial) | - | young 0: edge equality | 1-10 mm | 0.02 1 / .9 .95 .001 | 2 ms implicitfast Newton; 2 ms Euler CG; 1 ms discrete CG | yes | none |
| T1 | badinkajink/hand `scripts/contact_bed_candidates.py` (+ hand-archive 20261007 compliance page) | TPU fingertip cap replacing pad spheres | 3 / full, type direct | 1 mm hex columns, two 1.5 mm layers (165 pts, 540 tets per pad) on the spherical pad (radius `H.R_PAD`, not read) | 1e7 / 0.45 / 1e-3 (beta 1 ms) | 0 | 0.002 1 / .9999 .9999 .001 .5 2, priority 1, internal false | 1 ms / discrete / CG | back layer + rim pinned; welded child of pad body | 4x4x3 mm block, 1 N: 19.0 um vs exact 18.75 (nu = 0) |
| T2 | badinkajink/hand-archive `scripts/flexcomp_skin_probe.py` (+ `.json`) | compile/cost probe of skins on a 10.55 mm tip | 2 radial; 3 full/trilinear/quadratic | 7-11 grid | shell 3e6 / 0.3 (dim 2) | 0.5 mm | 0.002 1 / .9999 .9999 .001 | 1 ms / discrete | press slider | none ("not a study"); costs 16-102 us/step |
| T3 | Msornerrrr/in-hand-rotation-mjlab `xmls/left_hand_flexpad.xml` | LEAP fingertip pads | 3 / full grid 2x3x3 | spacing 2.5 x 10 x 10 mm | 1e5 / 0.45 / - | 1.2 mm | 0.006 1 / global .99 .99 .002 | - / implicitfast / - (elliptic, impratio 100) | 9 nodes pinned per pad; on finger links | none; variant not referenced by fetched code |
| T4 | Msornerrrr `robots/leap_hand/pad_model.py` | what they ship instead: rigid convex slabs + soft solimp | no flex | - | - | - | solimp .30 .99 .003 .5 2 | - | on finger | calibrated against real PaXini pad readings (their claim) |
| T5 | Ju6276/WowSkin-Lab `wowskin_lab/core/geometry.py`, `docs/MODEL_CALIBRATION_ZH.md` | AnySkin-like magnetic skin on Panda/Robotiq fingers | 3 / trilinear (1 cell), type direct | 42 x 44 mm pad, 2 mm top skin | 3e4 / 0.35 / 1e-5 | 0.05 mm (code default) | timeconst 2 ms (code default) / from config (not fetched) | 500 us / discrete / CG (robot); 20 us Newton (bench) | 4 bottom nodes pinned; robot-mounted | none ("no convergence validation yet") |
| T6 | pulipakaa24/MetalSim `docs/research/deformables_2026-09-25.md`, `scripts/diagnostics/deformable/*.py` | Warp/C parity, Isaac comparison | 2,3 / full, trilinear | rod 0.5 x 0.02 m, 1-4 cells across | 1e5-1e6 / 0.4-0.45 | 5-20 mm | - | 0.1-1 ms explicit | rod end pinned | cantilever EI_eff/EI = 14.0 (1 across), 7.0 (2 across) (their measurement) |
| T7 | TX-Leo/HumanEgo `simulation/docs/physics_params.md`, `upgrade/physics.md` | soft bread grasped by robots | 2 or 3 / trilinear 4x2x2 | 14 cm objects | 2e4 / 0.25 / 0.03 | 1 mm | 0.004 3 / variant .95 .995 .001 .5 2 | 2 ms / discrete / - (elliptic) | free object | 40 mm cube: apparent E 6.8 kPa vs 20 kPa configured, contact sink 5.1 of 11.4 mm (their measurement) |
| T8 | cps-test-lab/roqsim `src/roqsim/flex.py`, `flex_modes.py` | framework rules for flex | any | - | example 5e5 / 0.45 | - | solref floor rule | discrete | - | measured rules: numerical damping = h, omega*h <= 0.3 (MuJoCo 3.14) |
| T9 | paulotto/hybrid-jaw-exo-sim `..._3_9_MPa_0_45.xml` | soft face skin under a jaw exoskeleton | 2 / full shell (stretch) | mesh | 3.9e6 / 0.45 / 1e-5, thickness 2 mm | 1 mm | 0.002 1.5 / .9 .999 .001 .1 6 | 30 us / implicit / - | skin on skeleton | has an E/nu <-> solref converter for edge constraints |
| T10 | dap-i/FOCUS-Hand `wonik_allegro/left_hand_soft.xml` | soft cube grasped by Allegro | 3 / trilinear | 9x9x9, 10-15 mm | 6e3 / 0.3 / 0.1 | 1 mm / 3 mm | 0.03 2 / .8 .95 .005 | 0.5 ms / implicitfast / Newton | free object | none |
| T11 | Albusgive/mujoco_learning `extend/touch/touch_flex.xml` | flex "touch" pad demo | 2 / full | 16x16 (inferred from corner pins 0 15 240 255) | 3e5 / 0 / -, elastic2d stretch, thickness 10 mm | - | 0.01 1 / .95 .99 .0001 | 0.5 ms / implicitfast | 4 corners | none; reads per-vertex contact forces (no sensor plugin) |
| T12 | EduardGilM/Thekenyos `scripts/kiwi_compression.py` | kiwi between gripper pads | 3 / full ellipsoid 7^3 | 9-12 mm cells | 1.57e6 / 0.4 / 1e-5 | 0.3 mm | CONTACT_TIME_S / .95 .99 .001 | 10 us / implicitfast / Newton | free | none |
| T13 | cloth users: dblanm/benchmarking_cloth, google-deepmind/aloha_sim Towel, mll-lab-nu scene_sharpa, SynReal | cloth / paper | 2 | - | shell plugin 8.5e4 / 0.40; 60 / 0.1 bend | 0.35-10 mm | various | 0.5-10 ms, mostly implicitfast (now an error with elasticity) | - | sim-to-real cloth benchmark (arXiv 2310.09543) |
| T14 | dfki-ric/deformable_gym `insole_fixed.xml` | gmsh insole to grasp | 3 / full + edge equality | mesh | 1e4 / 0.1 / 1e-4 | 1 mm | - | - | - | none (young + edge equality is now a compile error, C3) |

Official example settings for all 35 example files (scene.xml excluded): `official_examples_table.txt` (all 35 ran 0.5 s without divergence or warnings).

## 2. Per-source notes

### (a) Official

**MuJoCo 3.17.0 docs, local `doc/` (= https://github.com/google-deepmind/mujoco/blob/3.17.0/doc/...).**
- `XMLreference.rst` flex-elasticity: young = Young's modulus, poisson in [0, 0.5), damping = "Rayleigh's damping
  coefficient, units of time" scaling the stiffness. Nothing about the volumetric normalization (F1). Says
  non-interpolated flex elasticity supports vertices attached to articulated bodies; a vertex pinned to a moving body
  follows it and applies reactions.
- `computation/index.rst` (geIntegrators, discrete): metric M + hD + h^2 K; "flex terms are applied matrix-free inside
  the iterative solvers"; limitations: flex elasticity under PGS or noslip is a runtime error; interpolated flexes on
  arbitrary moving bodies need CG; discrete is "new (September 2026), under active development".
- `changelog.rst`: 3.13.0 item 4 adds `discrete`, removes implicit flex under implicit/implicitfast (runtime error; I
  reproduced it, `checks.json` C2: "flex elasticity is no longer integrated implicitly under integrator='implicit' and
  'implicitfast': set integrator='discrete'"); 3.14.0 item 1 experimental `ipc` flag (dim-2 only, frictionless);
  3.15.0 item 5 SNH via `mjsFlex.elastic3d = 1`; item 7 articulated vertex attachments (CG + discrete); 3.17.0 item 8
  corrected interpolated-flex shear and quadratic quadrature ("existing models ... can change behavior"); item 10
  contact cap only for rigid/interpolated flexes.
- `mjx.rst`: MJX (JAX) Flex row = "Not supported"; MJWarp: `VERTCOLLIDE`, `ELASTICITY`.
- No official flex tutorial/colab exists (`python/*.ipynb` contain no flexcomp; README colab list has none).
- Copy: discrete + CG, CG for any flex on a moving body. Avoid: relying on docs for E/nu accuracy of 3D StVK.

**`model/flex` examples** (`official_examples.json`). The full-dof elastic 3D examples (floppy, jelly, press) are blocks in
free space with shortest edges of 70-200 mm (CFL 0.15-0.49); damping 0-0.05 s; radius 0-5 mm; contact solref 0.01-0.02 s with solimp .9-.95 / .95-.99;
none is thin, stiff, or mounted on an arm. The press/floppy/jelly blocks are StVK at nu 0-0.4: they would look fine with
F1 because nothing is confined. Only `gripper_trilinear.xml` has an elastic 3D flex on moving (slide) bodies.

**MuJoCo Warp** (https://github.com/google-deepmind/mujoco_warp: README, `_src/io.py`, `_src/flex_test.py`).
README: "Flex: experimental". `io.py` rejects `dof="quadratic"` ("Quadratic flex interpolation (dof=quadratic) is not
supported.") and noslip with discrete. `flex_test.py`: all tests are parity tests against MuJoCo C (kinematics,
constraints, passive forces incl. `test_flex_3d_snh_passive`, `test_flex_attachment_jacobian_passive`), so Warp inherits
F1 by construction. Issues: #1709 (open) Warp does not apply mjMAXCONPAIR to flex-geom pairs; #1723 3D stiffness
layout changed from 21 to 24 values per element (SNH coefficients), fixed by #1724; #1859 (open) flex contacts
attributed to unrelated rigid touch/force sensors (geom = -1 sentinel) -> do not use rigid touch sensors with flex in
Warp; PR #1702 adds the discrete integrator to Warp (Sept 2026). PR #1612: Warp tactile sensor fixes (SDF/mesh tactile
sensor, not flex).

### (b) Third-party

**T1/T2 badinkajink/hand + hand-archive** (SR^2 Hand, https://github.com/badinkajink/hand). The closest analogue to us:
a 3 mm TPU cap (E 10 MPa, nu 0.45) on a fingertip, 1 mm columns, back layer and rim pinned, discrete + CG at 1 ms,
contact solref 2 ms with solimp 0.9999 (comment in the probe: MuJoCo contact stiffness scales with the contacting
body's mass, "a 5 mg vertex makes a 2 ms contact ~12 N/m: the first probe fell through the box"; d = 0.9999 gives
~1e4 N/m per vertex contact). Their compliance page (hand-archive 20261007): with explicit integration the cap
"diverges at 1 ms with up to 100 g of armature"; under discrete it runs at 1 ms; 4x4x3 mm block under 1 N: 19.0 um vs
exact 18.75 (nu = 0; my arithmetic check of the exact value: `analysis_s2.json` C); "the tool creeps over the light
vertices"; cost 1,249 us/step (98x the pads); flex has no GPU value because Warp had no discrete integrator. They chose
a non-flex skin. Copy: pin back face + rim, priority on the flex so its solref/solimp apply unmixed, solimp near 1 for
light vertices, internal="false". Avoid: nu = 0 validation (cannot see F1).

**T3/T4 Msornerrrr/in-hand-rotation-mjlab** (LEAP hand, 34 stars). `left_hand_flexpad.xml` has 2x3x3 full flex pads
(E 1e5, nu .45, 9 pinned nodes) under implicitfast; flex elasticity under implicitfast is a runtime error since 3.13
(C2), and the file is not referenced by the fetched Python. The shipped pad model (`pad_model.py`) is rigid convex
slabs with solimp (0.30, 0.99, 0.003, 0.5, 2) "a compliant layer", calibrated by replaying real episodes against PaXini
pad readings. Copy (as baseline): a calibrated solimp-compliance pad is the cheap reference to beat.

**T5 Ju6276/WowSkin-Lab**. Magnetic skin (30 kPa, nu 0.35, 2 mm) on Panda/Robotiq: one trilinear cell, 4 bottom nodes
pinned, discrete + CG 500 us, `MuJoCo >= 3.15` enforced. Their Robotiq log summary: full-node SNH inverted at ~3.41 s
with 292/508/1160 ms median per sample at 100/50/20 us; StVK at 20 us inverted at 3.41 s; reduced trilinear ~8.5 ms per
sample (their measurement, not re-run). They note SNH has no log barrier at J -> 0 (no anti-inversion guarantee) and
that "young 30 kPa, poisson 0.35 are uncalibrated; mesh, contact and timestep convergence are pending". Copy: the
trilinear reduced skin pinned at its base as the robot-mounted fallback. Avoid: assuming SNH prevents inversion.

**T6 pulipakaa24/MetalSim**. Most careful third-party measurement work on flex (Warp vs C parity, Isaac comparison):
measured "flex constraints and elasticity (young) cannot both be present" (I reproduced, C3), the 50-contact cap
dropping cloth through a box (cap 400 holds it), StVK cantilever EI_eff / EI = 14.0 at 1 cell across, 7.0 at 2 across
(their measurement; F1 can explain at most x1.03 of this at nu 0.4, `analysis_s2.json` A), explicit flex diverging at
0.5 ms for damping above 1e-4/1e-3. Copy: mesh sweeps against an analytic reference. Gap: no confined/thin-layer test.

**T7 TX-Leo/HumanEgo** (590 stars). Soft bread as trilinear flex, discrete at 2 ms. Their finding 1: contact compliance
hides the modulus (E 20 kPa configured, apparent 6.8 kPa; the platen sinks 5.1 of 11.4 mm into the contact with solref
0.004 3 on ~1 g nodes; solimp .95 .995 brings apparent E to 16.3 kPa). Directly relevant: for our 1.5 mm layer any
contact penetration is in series with ~1-10 um of elastic deflection, so the contact must be far stiffer than the layer
(our solref 1e-4 / solimp .99 does this; measure penetration explicitly in every benchmark).

**T8 cps-test-lab/roqsim**. Encodes measured MuJoCo 3.14 rules: under discrete each elastic mode is damped as if
damping were one timestep larger (zeta_i = (damping + h) omega_i / 2); discrete is accurate for omega*h <= 0.3; under
discrete with refsafe a contact time constant is capped at about 1.03 steps. Copy: compute omega*h of the pad.

**T9 paulotto/hybrid-jaw-exo-sim** (+ arXiv 2502.18749, abstract opened; values in the HTML are MathML and were not
extracted). Robot-mounted soft skin, but as a 2D stretch shell (3.9 MPa, nu .45, 2 mm) at 30 us under `implicit` (a
runtime error on >= 3.13). Converter between E/nu and edge-constraint solref. Not applicable to a 3D layer.

**T10-T14** soft objects / cloth / demos. None validates against analytic solutions except the cloth sim-to-real
benchmark (dblanm, arXiv 2310.09543). Albusgive's touch_flex shows reading `mjContact.flex`/elem/vert forces directly
(the approach of our tactile.py; dev pointer in discussion #2641: `mjContact` has a `flex` field).

Papers (abstracts opened via arXiv API, `raw/arxiv_s2/abs.xml`; no flex settings extracted): 2412.03191 soft adaptive
feet, 2502.18749 soft head + jaw, 2606.22397 tendon-driven continuum robots natively in MuJoCo validated against a
Cosserat rod, 2501.08077 HydroelasticTouch (tactile via hydroelastic contact, not flex). arXiv searches for flexcomp /
MuJoCo + tactile + soft/finger/gel (`raw/arxiv1.json`, `raw/arxiv2.json`) found no paper that builds a tactile skin
with flex and validates it. Semantic Scholar was rate-limited (`raw/s2_search.txt`).

### (c) MuJoCo issues / discussions bearing on our problems

- Accuracy / validation: discussion #2500 (soft gripper loses stiffness after topology optimisation; quagla: "standard
  piecewise linear FE ... linear elements are very mesh-dependent ... use a finer mesh"; damping = XML value times the
  Lame parameters; inverted tets with coarse meshes; `internal="false"`). Discussion #1777 (cantilever off by ~8x;
  quagla: "coarse simulations in elasticity with p1 elements are notoriously poor", use the cable plugin, which is
  validated in `test/plugin/elasticity/elasticity_test.cc`). Issue #2084 (soft gripper, E 11e6 nu .45 diverges; "intrinsic
  feature of elasticity ... reduce the timestep", contact is always compliant). None mentions the Poisson factor.
- Poisson / lambda: only PR #3621 (above) and PR #3633 (SNH strain-rate damping) match "lame"; no report of F1.
- Thin layers: no issue or discussion found (query "flex thin": nothing relevant).
- Contact limits: changelog 3.17 item 10; issues #3297/#3298/#3495 and PR #3562 (filterFlexContacts compaction bugs,
  deterministic tie-breaking); mujoco_warp #1709. Issue #1922 (touch_grid on rigid bodies gives one contact; a commenter
  notes flex contacts fixed it but were "super slow" on CPU).
- Flex on articulated robots / divergence / metric: issue #3364 (soft trilinear gripper on a rotated parent body -> NaN
  in QACC), fixed by PR #3379 (passive forces assumed world-aligned slide joints) and PR #3437 (stretch stiffness basis
  for rotated parents); PR #3617/#3618 (attachments through vertex Jacobians, CG + discrete required, PSD approximation
  without Jacobian-derivative terms, mass preconditioner); PR #3386 (effective metric in CG), PR #3439 (block
  preconditioner replaces factorization: bag.xml 39 -> 86 CG iterations; `qacc_smooth` becomes an iteration that raises
  mjWARN_INERTIA at its cap, which is the warning text we see in F4); PR #3638 (SNH Hessian projected to PSD; benchmark
  includes "7^3 grid on hinge (CG)" at 0.1 ms). Issue #2077 (attach soft to rigid: make the flexcomp a child and pin).
- Discrete / implicit: changelog 3.13 item 4 (C2); discussion #2652 (floppy at 10 ms resets; "0.001 is quite close to
  the stability limit ... use a softer material or edge constraints"; pre-discrete era).
- solref/solimp for flex: no official recommendation beyond the examples (solref 0.01-0.02, solimp .9-.95/.95-.99).
  Practitioners with light vertices use solimp d ~ 0.98-0.9999 and solref 2-4 ms (T1, T5, T7), and priority on the flex.
- The F4 warning text ("Flex stiffness is too ill-conditioned for the effective-metric block preconditioner ... relative
  residual of 3.67e-07", followed by "Inertia matrix is too close to singular ... at DOF 260") also appeared once in
  `checks.log` during `mj_forward` of the free 5x5x5, 10 mm, 1 g patch-test cube at h = 0.1 ms (no contacts, no robot);
  it did not affect the qfrc_spring ratios. So the warning is easy to trigger and is not by itself evidence of an
  indefinite metric.
- F5 (false "not rigid and has no equality constraints or passive forces" warning): reproduced on every elastic flex in
  `checks.log`; no issue found.

## 3. What to copy, what to avoid (for the PDMS layer)

Copy: discrete + CG; flex as child of the blade body, back face pinned (T1, #2077); flex priority so its contact
parameters apply; contact much stiffer than the layer and penetration reported (T7); `elastic3d = 1` via MjSpec or the
F1 input mapping for StVK (C1); mesh sweeps against exact solutions (T6); a calibrated solimp-compliance rigid pad and
a 1-cell trilinear skin as baselines (T4, T5).
Avoid: implicit/implicitfast with elasticity (error); edge equality + young (error); nu = 0 or bending-only validation;
expecting interpolated flexes to deliver more than 50 contacts per geom; MJX (no flex); rigid touch/force sensors with
flex in Warp (#1859); treating the 0.5 ms dynamics of a 1.5 mm layer as physical (omega*h >> 0.3).
