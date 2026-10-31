# compare: MuJoCo 3.17.0 flex vs the exact solutions, and the choice of cell structure

The layer is 51 x 71 x 1.5 mm, linear elastic, E = 1 MPa, nu = 0.45 (also 0.49), 965 kg/m^3, with no damping.
Every number below comes from a run made in this directory or from the input directories' result files; the script
and output file are given each time. All errors are against the exact or ground-truth solutions (GT), never against
Winkler.

GT sources:
- C1, C1f, C2 and C3 come from `../gt_fem/results.json` (finite-pad Q2 FEM).
- C4, C5, C6 and C6f come from `../gt_transform/results.json` (exact infinite bonded layer). The finite-pad effect
  there is below 2e-4.

`gt.py` is the single accessor for both.

Error definitions:
- **total** = K_app / K_exact - 1. Here K_app = contact force / indenter displacement, i.e. what the rig reads.
- **discretisation** = K_static / K_exact - 1. K_static is MuJoCo's own elastic operator on the same mesh with exact
  BCs and no contact (from `../bench_static`).
- **contact** = K_app / K_static - 1.
- **C5**: the force at 30 um, F(30 um), and the contact radius.
- **C6**: the force in the centre taxel and the mean u_z/delta of the four edge-neighbour taxels.

## 0. Consistency checks done before using the numbers

| check | result | file |
|---|---|---|
| The two GTs where they overlap: C4 (12 values) and C6/C6f (6 values) | all within the combined error bars. Max relative difference: 2.7e-5 for C4, 1.2e-3 for C6 (gt_fem is above, as a displacement FEM should be). Taxel u_z maps differ by at most 4.3e-4 delta | `check_gt.py` -> `out/check_gt.json`, `.out` |
| `rig2.py` (my copy of the dynamic rig) vs `../bench_dynamic/rig.py` | the centred 40x56x2 C4/C6/C6f runs give identical K_app (0.148 / 0.164 / 0.136 N/um) | `out/dynamic_errors.csv` |
| Static vs dynamic: does the contact rig reproduce MuJoCo's static operator? | **yes, once contact compliance and the collision type are removed.** With a plane indenter (vertex contacts), stiffest contact and flex mass x100, K_app/K_static - 1 = -0.43 / -0.43 / -0.47 % on 10x14x2 / 20x28x2 / 20x28x3. That is the expected small shortfall from below | group `sd`, `fig/fig_contact.pdf` |
| Why the dynamic "layer stiffness" in bench_dynamic exceeded static values | a **box** (any non-plane geom) collides with whole tetrahedra. With contact compliance removed (mass x100), it is +7.7 / +7.1 / +3.6 % *stiffer* than the exact face BC, because the contact constrains points inside the tets. With the robot settings the penalty compliance dominates: -13 / -14 / -21 %. So bench_dynamic's K_layer > K_static is a collision-geometry effect, not an inconsistency | same |
| Reference values quoted by the benches | bench_static used gt_fem's interim file (C6ns 0.2170 vs final 0.21658; C1 8.567 vs 8.5662). bench_dynamic used 8.59 N/um (main session) vs final 8.566. All errors here are recomputed against the final GT | `analyze_static.py`, `analyze_dynamic.py` |

No disagreement was left unexplained.

## 1. Error vs GT, separated into discretisation and contact

Full per-model and per-run tables:
- `out/static_errors.csv`: 295 static models, all cases, plus taxel-map RMS and C6 cross-talk.
- `out/dynamic_errors.csv`: 2639 dynamic runs, 1666 from bench_dynamic and 973 from here.
- `out/config_errors.csv`: per configuration, with the 95 % CI over 5 lateral indenter placements for C4, C5, C6
  and C6f.

**Static operator (no contact), nu = 0.45, relative error in %** (`out/static_errors.csv`):

| mesh (cells) | C1 | C1f | C2 | C3 | C6 | C6f | C4 (nodal) | C6 neighbour u_z/delta (exact -0.035) |
|---|---|---|---|---|---|---|---|---|
| 10x14x2 uncorrected (robot now) | +85.3 | +77.3 | +0.5 | +100.4 | +345 | +351 | -32 | +0.238 |
| 10x14x2 F1-corrected | +4.4 | +4.8 | +0.5 | +14.2 | +158 | +171 | -60 | +0.242 |
| 5x7x1 corrected (robot choice) | +6.9 | +9.2 | +0.5 | +26.1 | +371 | +398 | - | +0.495 |
| 40x56x2 corrected (rig choice) | +2.6 | +2.3 | +0.5 | +5.1 | +36.3 | +35.5 | -1.8 | +0.038 |
| quadratic 40x56x3 cells (best tried) | +0.3 | +0.3 | +0.2 | +0.7 | +5.1 | +5.1 | -2.6 | -0.024 |

At nu = 0.49:
- 5x7x1 is +22.4 (C1), +50.4 (C3) and +770 (C6).
- 40x56x2 is +7.8, +14.1 and +110.

Taxel force maps for distributed loads are good on every mesh: the C3 force-map RMS is 0.0045 to 0.009 of the
total force per taxel.

**Observed convergence order** (`analyze_static.py` -> `out/static_orders.json`). I fitted e = a dx^p + b dz^q
(dx = in-plane cell size, dz = element thickness, in mm) over all meshes with at least 2 elements through the
thickness, at nu = 0.45. The residual RMS is 1-8 %.

| element | C1 p / q | C3 p / q | C6 p / q |
|---|---|---|---|
| full (corrected) | 0.76 / 2.02 | 0.97 / 2.20 | 1.22 / 0.93 |
| trilinear | 0.98 / 1.58 | 1.01 / 1.68 | 1.17 / 1.38 |
| quadratic | 1.86 / 0.80 | 1.22 / 1.18 | 1.24 / 2.17 |
| full uncorrected | 0.81 / 0.05 (a +80 % floor = F1) | 0.94 / 0.06 | 1.23 / 0.21 |

- In-plane order is about 1 for every element type. This is what the r^-1/2 edge singularities at the platen and
  punch edges and at the bonded free edge allow (gt_fem's graded Q2 FEM sees 1.2-1.5).
- Quadratic elements do not raise the order. They lower the constant: C6 a = 0.036 vs 0.214 for linear tets.

**Contact (dynamic) part.** The contact error is set by the penalty spring in series with the layer. In 3.17 its
stiffness in N/m is proportional to the vertex mass: solref is an acceleration-space stiffness (internals NOTES
point 3). It is therefore set mainly by the elements through the thickness (`fig/fig_thickness.pdf`, right panel,
bench_dynamic group `conv`, C1 at 20x28):

| elements through thickness | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| discretisation (%) | +6.9 | +3.2 | +2.5 | +2.3 | +2.1 | +2.1 |
| contact (%) | ~0 | -15.1 | -22.8 | -28.5 | -33.2 | -37.1 |
| total (%) | +6.9 | -12.4 | -20.9 | -27.0 | -31.8 | -35.8 |

Other contact effects:
- **Box punch, placement.** For C6 the contact error depends strongly on where the punch sits relative to the mesh.
  On 40x56x2, a punch whose edges lie on grid lines (the centred, taxel-aligned case) gets 128 contacts, a
  penetration share of 0.53 and K -32 %. Shifted by a fraction of a cell, it gets 161 contacts and K +8.5 to +29 %.
  Hence the wide CIs.
- **C5 sphere.** The contact radius (0.63 mm exact) is smaller than every uniform cell, so the error is
  discretisation-dominated: +313 % at 1.275 mm cells.

## 2. How cell size matters (accuracy, cost, convergence)

Figures:
- `fig/fig_static_convergence.pdf`: error and cross-talk vs cell size.
- `fig/fig_thickness.pdf`: error vs elements through the thickness.
- `fig/fig_pareto.pdf`: error vs time per step.
- `fig/fig_solver.pdf`: PCG and CG iterations and settling time vs cell size.

1. **In-plane size.**
   - Whole- and half-pad loads (C1, C3): from 2.55 to 1.275 mm the static C3 error goes from 8.0 to 5.1 %. With
     contact, C1 changes by < 1 % (-12.4 -> -13.7 %), while the cost rises 4.4x (bench_dynamic C1 CG 1000 at nz=2:
     150 -> 671 ms/step) and contacts rise 3.8x. **In-plane refinement past ~2.5 mm does not pay for distributed
     loads**, because the contact error (-15 %) is independent of the in-plane size.
   - Localized loads: the static error only falls as h^1.2 (C6: +72 % at 2.55 mm, +36 % at 1.275 mm, +15 % at
     0.64 mm). A uniform grid cannot resolve C4, C5 or C6 within about 1 s/step.
2. **Through-thickness.**
   - Static C1 and C3 saturate after 2-3 elements: +3.2 / +2.5 / +2.3 % for 2 / 3 / 4 elements at 2.55 mm.
   - With contact, every added element makes the total *worse* by 6-9 points (table above). Each also costs
     1.3-1.4x (bench_dynamic) and raises the PCG iterations of the M + h^2 K solve: 40 / 54 / 70 / 86 / 102 at
     2.55 mm for 2-6 elements (`out/static_errors.csv`).
   - **"Overmeshing" through the thickness is real.** Thinner elements mean lighter surface vertices, which make the
     contact softer and the solve harder.
3. **Solver convergence.**
   - The M + h^2 K PCG exceeds opt.iterations = 100 at about 0.85 mm cells for any thickness, and at 2.55 mm for
     6 elements.
   - No-slip contact (elliptic cone, impratio 100) never converges at CG 100 on any mesh: the frac_steps_at_iter_cap
     column is 1.0 in bench_dynamic.
   - With CG 1000 the median is 480-800 iterations and the force settles to 1 % within 1 ms (mesh 2.55 mm or finer)
     or 27-200 ms (coarse meshes).
   - Frictionless contact converges in 15-70 iterations (`fig/fig_solver.pdf`).
4. **Time step vs stiffness.** h·omega_max = 22-916, set by dz (bench_static). Only the implicit discrete integrator
   can run this layer, and stiff contact (time constant < 2h) is stable only if the constraint solver converges.

## 3. Choice of cell structure, with confirmation runs

Confirmation runs, all made here:
- `rig2.py` + `plan.py` + `sweep2.py`, groups:
  - robot: 288 runs
  - robotm: 384 runs (mass scaling, intermediate solref)
  - rig: 168 runs
  - graded: 13 runs
  - nu49: 96 runs
  - sd: 24 runs
- On the robot itself: `robot_grasp.py` + `robot_runs.py` -> `out/robot_grasp.jsonl`. These are 16 runs (15 configurations) of the full
  2.7 s close-and-lift with both pads, islands disabled and F1-corrected inputs.

The local cases use 5 placements (offsets of up to 0.44 cell); the "±" is the 95 % t-interval over those placements.
The global cases use one centred run; their force chatter (std/mean over the last 100 ms) is <= 0.7 %, except C2: 4.2 % on the
rig choice and 13.6 % on the robot choice.

### (b) Robot, budget ~1-5 ms per 0.5 ms step

**Choice:**
- dof="full", F1-corrected inputs (young 971786.83 Pa, poisson 0.40909091).
- Vertex grid 6 x 8 x 2 = 5 x 7 x 1 cells. That is one element per taxel (grid lines on the taxel boundaries) and one
  through the thickness, aspect 6.8.
- Flex contact solref 0.001 1, solimp 0.99 0.99 0.001.
- `<flag island="disable"/>`, CG 100, plus the robot's other options unchanged.

Why:
- **Cost.** Only this grid meets the budget on the robot: 6.0 ms/step for the whole model with both pads (tag
  6x8x2_B_m1). The alternatives cost 21.1 (11x15x2) and 27.3 ms/step (11x15x3, the current grid).
- **Stability.** No BADQACC and no inverted tets; held 119.7 ± 3.9 / 119.1 ± 3.5 N; 0 N on the rigid blade; box
  lifted 49.9 of 50 mm.
- **Interpolated flexes are excluded** on the robot. Their node forces skip ancestor joints (bench_static finding),
  and they are capped at 50 contacts per geom.

Rejected robot options (robot runs, `out/robot_grasp.jsonl`):

| run (6x8x2 unless noted) | ms/step | held N (L) | compression L (um) | penetration (mm) | inverted tets |
|---|---|---|---|---|---|
| **solref 1e-3 (chosen)** | 6.0 | 119.7 ± 3.9 | 29.1 ± 3.7 | 0.17 | 0 |
| solref 1e-4, iterations 1000, tol 1e-10 ("A") | 21.4 | 120.8 ± 0.3 | 44.5 ± 0.5 | 0.01 | 0 |
| 11x15x2, setting A | 88.9 | 121.1 ± 0.5 | 41.1 ± 0.1 | 0.035 | 0 (run twice by accident: identical results) |
| solref 5e-4 / 3e-4 / 2e-4, CG 100 | 4.8 / 4.3 / 3.9 | 176 ± 103 / 268 ± 220 / 69 ± 98 | 157 / 3002 / 3019 | 0.27 / 0.18 / 0.10 | 1 / 273 / 414 |
| solref 1e-3, flex mass x10 / x100 | 5.5 / 4.0 | 119.9 ± 4.8 / 120.4 ± 3.6 | 56 ± 36 / 94 ± 63 | 0.04 / 0.01 | 0 |
| 11x15x2, solref 1e-3 | 21.1 | 124.3 ± 3.9 | 37.5 ± 5.0 | 0.19 | 0 |
| 11x15x3, solref 1e-3 (current grid) | 27.3 | 121.2 ± 0.2 | 45.3 ± 0.4 | 0.62 | 0 (9.1 N on the rigid blade) |
| chosen, nu = 0.49 | 6.7 | 117.5 ± 2.4 | 16.2 ± 4.6 | 0.16 | 0 |

- **Mass scaling** (it raises the contact stiffness) makes the compression chatter on the robot, and on the rig it
  breaks no-slip convergence: C2 chatter 2-15 at mass x100. It also violates the density rule. Rejected.

Errors of the chosen robot structure on the rig (nu = 0.45, `out/config_errors.csv`), in %:

| case | C1 | C1f | C2 | C3 | C4 | C5 (F at 30 um) | C6 | C6f |
|---|---|---|---|---|---|---|---|---|
| total | -89.0 | -88.3 | -41.1 | -88.4 | -70.7 ± 13.9 | +533 ± 272 | -76.4 ± 5.5 | -74.7 ± 6.7 |
| discretisation / contact | +6.9 / -89.7 | +9.2 / -89.3 | +0.5 / -41.5 | +26.1 / -90.8 | - | - | +371 / -95 | +398 / -95 |
| same mesh, solref 1e-4 + CG 1000 | +12.2 | +16.2 | +38.2 | +32.1 | +509 ± 125 | +26450 | +452 ± 106 | +436 ± 253 |

At nu = 0.49 the chosen structure gives C1 -97.0 %, C2 -40.4 % and C6 -88.9 %.

**Plainly:** within 1-5 ms/step, the robot's skin compliance is the contact penalty, about 10x softer than the PDMS
(penetration share 0.91). Even so:
- the per-taxel *forces* follow equilibrium: the force-map RMS is 0.016-0.020 of the total for C1/C3 and 0.033-0.036
  for C6;
- force stays in the pressed taxel;
- neighbour u_z/delta is +0.015, against exact -0.035;
- the 6x8x2 contact still moves only 0.17 mm under 120 N.

What it gives wrong is deformation and stiffness. If those matter on the robot, use setting A (21 ms/step) or the
rig.

### (a) Fixed-rig sensor study (pad in a static body)

**Choice:**
- dof="full", F1-corrected inputs.
- Vertex grid 41 x 57 x 3 = 40 x 56 x 2 cells (1.275 x 1.268 x 0.75 mm, aspect 1.7).
- Contact solref 0.0001 1, solimp 0.99 0.99 0.001, refsafe disabled, CG iterations 1000 (no-slip needs it).
- For an indenter at a *known* position: a **graded tetrahedral mesh** (flexcomp type="direct", `rig2.py`
  `graded_mesh()`). It clearly wins for C4, C5 and C6. Both versions, in % (nu = 0.45):

| structure | C1 | C1f | C2 | C3 | C4 | C5 F(30 um) | C6 | C6f | ms/step |
|---|---|---|---|---|---|---|---|---|---|
| **40x56x2 uniform (chosen)** | -13.7 | -12.6 | -8.2 | -12.4 | +8.4 ± 7.0 | +313 ± 56 | +9.1 ± 24.8 | +9.8 ± 30.4 | 36-486 |
| its discretisation / contact | +2.6 / -15.9 | +2.3 / -14.6 | +0.5 / -8.7 | +5.1 / -16.7 | -1.8 / +10.4 | - | +36.3 / -20.0 | +35.5 / -19.0 | |
| 20x28x2 uniform | -12.4 | -10.9 | -1.6 | -10.1 | +54.6 ± 20.2 | +1236 ± 287 | +41.4 ± 38.4 | +48.2 ± 46.8 | 7-140 |
| graded, 0.5 mm core, 3 el. (7380 vertices) | | | | | +14.8 ± 7.3 | | +16.6 ± 10.0 | +17.8 ± 7.8 | 119-510 |
| graded, 0.2 mm core, 4 el. (6105 vertices) | | | | | | +20.9 ± 1.8 (a: -5.1 %) | | | 108 |
| graded, 0.15 mm core, 5 el. (16170 vertices) | | | | | | +11.6 (a: -7.6 %) | | | 469 |
| 40x56x2 at nu = 0.49 | -46.9 | -42.0 | -20.6 | -43.5 | +21.8 ± 6.6 | +601 ± 53 | +8.5 ± 16.7 | +27.9 ± 30.8 | 54-495 |

What this table shows:
- **Uniform meshes and localized loads.** The uniform grid's +8-10 % on C4 and C6 is a **cancellation**
  (discretisation +36 %, contact -20 %), and its placement CI is ±25-30 %.
- **Graded meshes.** Their errors are genuine: the contact share is 3-5 %, the CIs are tight, and the cost is comparable.
  The reason, as an inference: flexcomp gives every vertex the same mass, so the small elements carry heavy
  vertices and therefore stiff contacts.
- **Distributed loads.** Expect -12 to -14 % from contact compliance. For a sensor study, read the layer
  deformation from the vertex positions (`flexvert_xpos`) rather than from the indenter travel. Raising the contact
  stiffness (solref 1e-5 / solimp 0.9999, or mass scaling) fixes C1f but breaks no-slip shear: C2 -57 % (10x14x2)
  and -20 % (20x28x3) at CG 1000 (bench_dynamic `contact` group), and C2 -93 % with stiffest plus mass x100 here.
- **nu = 0.49.** The contact error doubles (-45 to -50 %), because the layer is 4x stiffer and the contact is not.
- **Interpolated flexes with contacts were rejected** (group `rig`, mass x100, stiffest, split 5x5 indenter):
  - C2: -93 to -133 % (shear fails).
  - C3: -67 to -73 % (50-contact cap; C3 has no split).
  - C6: no meaningful value.
  - Quadratic 20x28x1 no-slip exceeds 1 s/step (`too_slow`, 9 runs).

## 4. Deliverables and how to rerun

- `recommended_pdms_flex.xml`: the robot flexcomp (6x8x2, F1-corrected, solref 1e-3), with the rig alternative and
  all caveats in its comments. It compiles and steps (`check_xml.py` -> `out/check_xml.out`).
- `summary.json`: GT cross-check, static errors of the key meshes, convergence orders, static-vs-dynamic
  reconciliation, per-configuration and per-case errors with CIs, robot runs, choices.
- `fig/fig_static_convergence`, `fig_thickness`, `fig_pareto`, `fig_solver`, `fig_contact`, `fig_candidates_robot`,
  `fig_candidates_rig`, each as `.pdf` and `.png`. They share one style (`fig_style.py`; reference palette, checked
  with the dataviz validator; identity is also coded by marker).
- `PROGRESS.md`: checkpoint log.

Rerun, with P = /home/dohyun/miniforge3/bin/python, from this directory:

```
$P check_gt.py; $P analyze_static.py
$P sweep2.py sd robot rig robotm graded nu49 --nproc 60      # ~20 min; cached by cfg hash in runs/
$P robot_runs.py 11                                          # robot grasps, ~10 min (setting A runs longest)
$P analyze_dynamic.py; $P make_summary.py; $P figures.py; $P check_xml.py
```

## 5. Verified vs assumed; caveats

- **Verified by runs here:** every number in sections 0-3.
- **Taken from input files:** static (bench_static) and early dynamic (bench_dynamic) numbers, and the GTs.
- **Assumptions and limits:**
  - **Rig vs robot.** The rig's per-step times are for the pad alone in a static body. The robot times are for the
    whole model with both pads. Both were measured with about 20-60 other processes running, so they are
    indicative only.
  - **Global cases have no placement CI.** They have a single centred run each; for them the uncertainty is the
    chatter (<= 0.7 %, except C2: 4-14 %) plus the GT bar (<= 0.08 %).
  - **"unphysical" flag.** Coarse meshes under a punch smaller than a cell often end with the face vertices beyond
    the indenter, i.e. penetration share < 0. Those runs are flagged `unphysical` in `out/config_errors.csv`, for
    example 4 of 5 C6 placements on 5x7x1 with solref 1e-4. They are kept in the means, since that is what the
    model outputs.
  - **Snap-through.** One run snapped through: 10x14x1, C6, solref 1e-4, CG 1000, layer displacement 336 um. It
    is excluded.
  - **Graded meshes.** They were tested only for the frictionless and no-slip punch and for the sphere at the mesh
    centre, at nu = 0.45. They use the same Kuhn diagonal in every hex, so the tet-bias of internals point 5
    applies. The "uniform mass per vertex" explanation for their small contact share is inferred, not measured.
  - **Not run:** C4 at a = 2.5 mm, C5 at R = 25 mm, and no-slip C4.
