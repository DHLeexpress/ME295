# bench_dynamic: MuJoCo 3.17.0 flex with real contacts on the 51 x 71 x 1.5 mm PDMS pad

Contact error, cost and solver convergence vs cell size. Linear elastic, E = 1 MPa, nu = 0.45 (also 0.49),
rho = 965 kg/m^3, no damping. Every number below comes from `results.json` (one record per run) or from the
`tables/` files generated from it, unless a different source is named.

## What was run

* `rig.py` runs one simulation. The pad is a `flexcomp type="grid" dim="3"` in a static body, with the back face
  pinned (`<pin gridrange>` for full flexes, `<pin id>` on the back-face nodes for interpolated flexes). The
  indenter is a rigid body on 3 slide joints driven by position actuators (kp = 1e10, dampratio 1, mass 0.1 kg);
  `driver=mocap` swaps in a kinematic mocap body. Robot options: `integrator="discrete"`, CG, elliptic cone,
  impratio 100, h = 0.5 ms, gravity off. Each phase is a 10 ms linear ramp followed by a 200 ms hold.
  Displacements: C1/C1f 2 um; C2 3 um normal then 1 um along y; C4 (a = 5 mm) 10 um; C5 (R = 10 mm) in 6 levels
  up to 30 um; C6/C6f 5 um. The no-slip cases use condim 3, mu = 2; the frictionless ones use condim 1.
  Stop rule: a run stops if the median of steps 3-12 is above 1 s per step (`too_slow`). The sweep then skips any
  mesh that is componentwise larger in the same branch (`skipped_branch_too_slow`).
* Metrics in each record: the force on the layer from `efc_force` (3 components, mean and std over the last 200
  steps); the layer's own displacement (mean `uz` of the free-face vertices under the indenter, taken from
  `flexvert_xpos`) kept separate from the indenter displacement and the contact penetration (`-con.dist`); per-taxel
  force, contact count and displacement (5 x 7); ncon/nefc; chatter (std/mean); settling time to 1 % and 5 % after
  the ramp; solver iterations (median and the fraction of steps at the cap); us per step (median over the steps in
  contact, >= 200); compile time; arena/RSS memory; MuJoCo warnings (counters and text); divergence.
  `pinned_weight_share` is the fraction of the contact force that MuJoCo's barycentric weights put on pinned
  vertices. `F_to_free_dofs_N` is the contact force from `qfrc_constraint` on the free vertices.
* `sweep.py GROUP...` runs the matrix with 90 processes (one per run, single-threaded). Groups: `main` (full flex
  with the F1-corrected inputs, CG 100 iterations, all 30 meshes x 7 cases), `conv` (the same with 1000 iterations),
  `raw` (uncorrected, 4 meshes), `tri`, `quad` (cellcount = cells, one collision vertex per node), `split`
  (indenter split into 2x2/3x3/5x5 geoms), `contact`, `margin2`, `newton`, `driver`, `iter`, `nu49`, `timing`.
  In total there are 1666 records.
* `analyze.py` turns `runs/*.json` into `results.json` and adds ground-truth columns if
  `../gt_transform/results.json` or `../gt_fem/results.json` exist. Neither file existed at 20:57 on 2026-10-30, so
  there are no error-vs-GT columns. Rerun `analyze.py; report.py` once they exist. The parser in `parse_gt` is a
  best-effort reader and may need adapting to their format.
* `report.py` writes `tables/all_runs.csv` (flat, every run), `tables/group_*.md`, `tables/headline_full_corr.md`
  and `plots/*.png`. `q.py` is a small query helper.

Rerun: `python sweep.py main conv raw tri quad split contact margin2 newton driver iter nu49 --nproc 90`, then
`python sweep.py timing --nproc 4`, then `python analyze.py && python report.py`. Cached runs are skipped by
config hash. The full matrix took about 25 min wall time on 90 processes.

Mesh naming: `cells` = elements (full) or interpolation cells (trilinear/quadratic). The robot's 11x15x3 vertex grid
is cells 10x14x2. In-plane cells 5x7, 10x14, 20x28, 30x42, 40x56 give 10.2, 5.1, 2.55, 1.7 and 1.275 mm.

## References used (and what is verified)

* Confined upper bound for C1, computed here: M A/t = 9.157 N/um at nu = 0.45 and 41.3 N/um at nu = 0.49.
  G A/t = 0.832 N/um is the same kind of bound for C2. A uniform field is kinematically admissible, so these are
  upper bounds for any correct discretisation that contains linear fields.
* C1 no-slip FEM reference 8.59 N/um (quadratic hex, 8.593 / 8.578). This is quoted from the main session's F2.
  **Not re-run here.**
* gt_fem / gt_transform: not available (see above). For C4, C5 and C6 there is no ground truth in this directory.
* The C5 Hertz half-space estimate F = 4/3 E/(1-nu^2) sqrt(R) delta^1.5 = 0.0275 N at 30 um (contact radius
  sqrt(R delta) = 0.55 mm) is only a sanity scale, not a ground truth (a/t = 0.37, so a thin layer stiffens it).
* Source facts: the flex-geom contact margin is the **sum** geom_margin + flex_margin
  (`src/engine/engine_collision_driver.c` lines 1968, 1986). Geom-geom contacts use the max instead. The 50-contact
  cap applies only to rigid and interpolated flexes (`doc/changelog.rst`, 3.17 item 10). For solref, "d*k and d*b
  are the effective stiffness and damping in constraint space", i.e. in acceleration units
  (`doc/modeling.rst`, Solver parameters / solref), so the contact stiffness in N/m scales with the effective mass.

## Results

### 1. Full flex, F1-corrected (`plots/C1_stiffness_vs_mesh.png`, `tables/group_main.md`, `group_conv.md`)

C1, no-slip platen, 2 um, CG 1000 iterations (converged); FEM ref 8.59:

| cells | K_app = F/indenter disp. | K_layer = F/mean layer disp. | penetration share | ncon |
|---|---|---|---|---|
| 10x14x2 (robot 11x15x3) | 7.601 | 9.822 | 0.226 | 350 |
| 20x28x1 / 2 / 3 / 4 / 5 / 6 | 9.155 / 7.508 / 6.778 / 6.259 / 5.847 / 5.497 | 10.10 / 9.64 / 9.20 / 8.97 / 8.81 / 8.69 | 0.09 / 0.22 / 0.26 / 0.30 / 0.34 / 0.37 | 1262-2334 |
| 40x56x3 | 6.721 | 9.119 | 0.263 | 4766 |

* **What the robot feels (K_app) is 11-36 % below the FEM reference, and gets worse with through-thickness
  refinement.** The contact itself absorbs 20-37 % of the indenter displacement. The penetration share depends on
  nz (0.09 for nz = 1 up to 0.37 for nz = 6 at 20x28) and hardly at all on the in-plane size (nz = 3: 0.236 at
  10.2 mm, 0.263 at 1.275 mm). The mechanism: solref sets an acceleration-space stiffness, so the force stiffness of
  a contact is proportional to the effective mass of the contacting vertices. More layers mean lighter surface
  vertices and therefore softer contacts. Checked directly: multiplying the flex mass by 10 cuts the penetration
  share from 0.250 to 0.037 and raises K_app from 6.47 to 8.48 (C1f, 20x28x3, `group_contact.md`).
* The layer itself (K_layer, using the free-dof force) converges from above. At nz = 6 it is 8.67-8.78 (no-slip),
  1-2 % above the FEM 8.59. K_layer is only a proxy: under a penalty contact the free-face displacement is not
  uniform (0.8-2.0 um under the 2 um platen), so F/mean(u) is not strictly bounded by M A/t. That is why nz = 2-3
  meshes show 9.2-9.6.
* nz = 1 is special. Every element touches the pinned back face, so 10-14 % of each contact force is put directly on
  pinned vertices (`pinned_weight_share` 0.10-0.14). The total force then overstates the layer's force (for example
  20x28x1 C1: F = 18.29 N, but only 16.45 N reaches the free vertices).
* C1f (frictionless, CG 100 iterations is enough): K_app 7.29 (10x14x2), 6.47 (20x28x3), 6.39 (40x56x3),
  5.94 (40x56x4). K_layer from the free-dof force: 9.28, 8.62, 8.51, 8.34.
* C2 (no-slip shear, 1 um after a 3 um press). With CG 100 the tangential force does not converge: chatter
  std/mean of 0.1-10, sign flips, K_s from -0.20 to 1.0. With CG 1000: K_s = 0.774 (20x28x3), 0.813 (10x14x3),
  0.71-0.74 (nz = 6), against the bound G A/t = 0.832; nz = 1 gives 0.92-1.06 (pinned share again). The rest of the
  10-15 % shortfall is slip: friction is a velocity-level constraint, and 0.1-0.2 um of the 1 um is lost
  (`slip_end_m` 1.0-1.5e-7 at CG 1000).
* C4 punch (a = 5 mm, 10 um, frictionless): K_app 0.294 / 0.167 / 0.131 and K_layer 0.375 / 0.263 / 0.215 N/um for
  10x14x2 / 20x28x3 / 40x56x3, with 8 / 30 / 116 contacts. This is **not mesh-converged** at 1.275 mm, and chatter is
  2-7 % even when CG converges (`settle_1pct` never reached).
* C5 sphere (R = 10 mm, 30 um): the force at 30 um is 11.4 (5x7x2), 1.66 (10x14x2), 0.43 (20x28x2), 0.22
  (30x42x2) and 0.14 N (40x56x3), from 2-6 contacts. The contact radius (about 0.55 mm) is smaller than the finest
  cell, so the sphere pushes one or a few vertices and the result scales roughly as h^-1.5 to h^-2. **C5 at 30 um
  cannot be resolved with these meshes.** The Hertz scale is about 0.03 N, still 5x below the finest result.
* C6 (square punch on the centre taxel, 5 um), and the user's question about contact per cell. **Contact is
  defined per contact point (one per element-geom pair), not per cell or taxel.** Every contact lies under the
  punch, and 100 % of the force is binned into the centre taxel (for example 40x56x3 C6: 134 contacts, centre
  share 1.012). Neighbouring taxels get displacement but no force. Their mean uz relative to the centre taxel is
  mesh-dependent:

  | mesh | edge neighbours | corner neighbours |
  |---|---|---|
  | 10x14x2 | 0.27-0.35 | 0.01-0.14 |
  | 20x28x3 | 0.03-0.21 | up to 0.04 |
  | 40x56x3 | 0.003-0.07 | up to 0.025 |

  So a reading like "+1, +2, +1" exists only in displacement, and on coarse meshes it is largely a discretisation
  artefact. The penetration share is 0.50-0.65, because the box-flex collider makes few contacts: 8 at 10x14x2.
  C6 no-slip also shows Fy of about -0.09 N at Fz = 0.80 N (40x56x3), an asymmetry from the tet layout of the
  grid plus friction.
* nu = 0.49 (`group_nu49.md`): the penetration share rises to 0.44-0.68, because the layer is stiffer and the
  contact is not. C1 K_app 18.1-18.8 at nz = 2 against a bound of 41.3.

### 2. F1 bias (`plots/F1_bias.png`, `group_raw.md`)
Uncorrected inputs, C1f K_layer: 15.29 (10x14x2), 14.01 (20x28x3), 13.49 (40x56x4). Corrected: 9.28, 8.62, 8.34.
The uncorrected values are about 1.6x higher and violate the 9.157 upper bound by 47-67 %, which is consistent with
F1. Re-verified indirectly here: with the corrected inputs, every nz >= 2 K_layer(C1f) is <= 9.28 and converges
below the bound. The patch test itself was not re-run.

### 3. Interpolated flexes and the 50-contact cap (`plots/interp_cap_split.png`, `group_tri.md`, `group_quad.md`, `group_split.md`)
* `ncon` is exactly 50 for every platen case. The load is then carried by 50 penalty contacts, so K_app collapses as
  the mesh is refined. Trilinear C1f: 1.82 (10x14x2), 0.36 (20x28x3), 0.10 (40x56x3), with penetration share
  0.80-0.98. Quadratic is worse (0.19 at 10x14x2).
* Workaround tested: split the platen into s x s geoms. The cap is per geom, so the contact count rises to
  200 / 450 / 1102. **It helps.** Trilinear 10x14x2 C1f with split 5x5: K_app = 8.245, penetration share 0.10,
  chatter 6e-4. That is a better K_app than the full flex (7.29). Trilinear 20x28x3 with split 5x5: 6.16
  (share 0.26). Quadratic with split 5x5: 3.46 (10x14x2). Trilinear 5x7x1 is unstable: C1 and C1f-split3
  diverged (`mjWARN_BADQACC`), and split 5x5 gave nonsense (K = 311).
* The elasticity is exact (F3), but cost is high. Each quadratic node is a dof. Quadratic is too slow (> 1 s/step)
  from 20x28x2 or 10x14x6 on for platens, and the M+K solve is under-converged in 100 % of quadratic runs with
  nz >= 2.

### 4. Contact settings (`plots/penetration.png`, `group_contact.md`, `group_margin2.md`)
C1f at 20x28x3, CG 1000:

| setting | penetration share | K_app |
|---|---|---|
| stiff (solref 1e-4 1, solimp 0.99 0.99 0.001, refsafe off) | 0.250 | 6.47 |
| stiffest (solref 1e-5, solimp 0.9999) | 0.083 | 8.04 |
| radius 0.1 mm | 0.307 | 5.94 |
| margin 0.1 mm on the flex only | 0.314 | 5.87 |
| MuJoCo default (refsafe on) | 1.000 | 0 (the 2 um are absorbed by penetration, no force) |
| stiff, flex mass x10 | 0.037 | 8.48 |
| stiff, flex mass x100 | 0.036 | 8.49 |

* The margin helps C6, where it captures more contacts (10x14x2: 30 contacts instead of 8, share 0.17 instead of
  0.59, K_app 0.53 instead of 0.24). It hurts C2: K_s 0.42 at CG 1000 on 10x14x2.
* **Rig bug, found and fixed:** the first "margin" preset set margin = 0.1 mm on both the flex and the geom. MuJoCo
  sums them for flex-geom pairs, which produced a 0.1 mm preload (F of about 490 N). Those records are kept as
  contact = "margin" and are invalid. The corrected runs are `margin_flex`.

### 5. Solver, cost, overmeshing (`plots/cost_iter_chatter.png`, `group_iter.md`, `group_newton.md`, `group_timing.md`)
* No-slip cases (C1, C2, C6) with elliptic cone + impratio 100 **always hit the CG 100 cap** (frac_at_cap = 1.00 in
  every main run). Chatter is 3e-2 to 2e-1 for nz >= 2, and the force never settles to 1 %. CG 1000 converges in
  376-1000 iterations, with chatter 1-2e-3 and 1 % settling in 1-33 ms on meshes of 10x14 and finer (about 0.2 s on 5x7). Frictionless cases converge in 30-55
  iterations; budgets of 30 / 300 / 3000 / 10000 give identical forces (`group_iter.md`).
* opt.iterations is also the iteration budget of the effective-metric (M+K) PCG. With 100 iterations, the warning
  "Flex stiffness is too ill-conditioned ... M+K solve ran out of iterations" appears in 14/84 runs with 1.7-1.275
  mm cells at nu = 0.45 and in 40/45 at nu = 0.49. It appears in 0 of more than 400 runs with >= 300 iterations.
* Newton (`group_newton.md`) agrees with CG 1000 (C1 10x14x2: 7.605 vs 7.601) in 2-9 iterations. With the servo
  indenter, whose dofs couple to every contact, it costs 310 ms/step at 10x14x2 and 63 s/step at 20x28x3
  (too_slow). With a mocap indenter it costs 25 ms (10x14x2) and 3.0 s (20x28x3 C1).
* The mocap driver cannot transmit shear. C2 with mocap gives K_s = -0.04, because friction acts on velocity and a
  mocap body has none. Use joints and actuators for tangential loading.
* Cost (low-load re-measurement, `group_timing.md`, 4 processes; the 90-process sweep timings are 1.1-1.7x
  higher and 3.3x higher for quadratic; forces are bit-identical):

  | mesh | C1f | C1 (CG 100) | C1 (CG 1000, sweep timing) |
  |---|---|---|---|
  | 10x14x2 | 3.1 ms | 9.6 ms | 48 ms |
  | 20x28x3 | 20 ms | 46 ms | 218 ms |
  | 30x42x4 | 79 ms | 134 ms | 658 ms |
  | 40x56x3 | 103 ms | 240 ms | 951 ms |

  Compile time is at most 12.5 s (40x56x6: 11.5 s). Arena use is at most 394 MB of the 2 GB, RSS at most 463 MB.
  No arena overflow; the only divergence is trilinear 5x7x1. The compile warning "not rigid and has no ... passive
  forces" appears in 1497 runs (false alarm, F5).
* **Overmeshing, in numbers.**
  * In-plane refinement beyond 2.55 mm buys less than 1 % for the platen: C1 K_app 6.778 to 6.721, K_layer
    9.20 to 9.12. Over the same step, contacts grow x3.8 (1262 to 4766), cost x4.4-5.3, and the M+K warnings start.
  * Through-thickness refinement beyond 2 elements makes K_app *worse*: -10 % per added layer at 20x28 (7.51, 6.78,
    6.26, 5.85, 5.50). CG iterations grow (C1 CG 1000 median 703 to 1000 from nz = 2 to 6; 49-74 % of steps at the
    cap at nz = 5-6), cost grows x1.3-1.4 per layer, and chatter at CG 100 rises from 1e-2 to 1.4e-1. The layer
    stiffness improves only from 9.64 to 8.69.
  * Localized loads (C4, C5, C6) are *not* converged at 1.275 mm. For them refinement still matters, but full
    convergence is out of reach within 1 s/step.
  * Best compromise for this rig: about 2.55 mm in-plane with 2 elements through the thickness (20x28x2, 1262
    contacts, 41 ms/step at CG 100 in the sweep), frictionless or with a CG budget >= 1000 for no-slip. Or trilinear
    10x14x2 with a split indenter for platen-type contacts.

## Not done / limits
* No C3 runs; C4 a = 2.5 mm and C5 R = 25 mm were not run (not in this task's case list).
* There are no ground-truth columns, because gt_fem and gt_transform had not finished.
* Settling times are measured after a 10 ms ramp; "never" means the force stayed outside the 1 % band in the
  200 ms hold.
* The F1 patch test and the F2 FEM numbers were not re-run here.

Files: `rig.py`, `sweep.py`, `analyze.py`, `report.py`, `q.py`, `results.json`, `runs/<id>.json|.npz`, `tables/`,
`plots/`, `logs/`, `PROGRESS.md`, and `rig_v1_backup.py` (rig before the margin_flex addition, otherwise identical).
