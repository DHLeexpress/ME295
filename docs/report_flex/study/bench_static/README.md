# bench_static: MuJoCo 3.17 flex as a finite-element operator (no contact)

What this does: it builds the 51 x 71 x 1.5 mm PDMS pad (back face pinned, E = 1 MPa, nu = 0.45 / 0.49 / 0.499,
rho = 965) as a MuJoCo flex, then takes **MuJoCo's own elastic stiffness** as a finite-difference Jacobian of
`qfrc_spring` with respect to the vertex/node coordinates. It solves the benchmark cases with exact BCs (no contact)
and studies the conditioning of the implicit metric M + h^2 K (h = 0.5 ms) and the PCG iteration counts.

Variants: `full_corr` = dof="full" with F1-corrected inputs; `full_raw` = dof="full" uncorrected (this is what the robot
uses); `trilinear` and `quadratic` = interpolated, with cellcount. Meshes: in-plane cells of 10.2/nt mm,
nt = 1,2,4,6,8,12,16 (10.2 ... 0.64 mm). Through the thickness: full has count z = 2..7 (1-6 elements), trilinear has
cellcount z = 1,2,3,4,6, quadratic has cellcount z = 1,2,3 with nt <= 8. In total **295 models, all status ok**.
The robot's 11 x 15 x 3 grid is `full_raw_nt2_nz3`.

## How to rerun
```
P=/home/dohyun/miniforge3/bin/python
$P driver.py 60            # out/<id>.json, one per model; skips finished ones (~40 min wall at 60 procs)
$P patch_check.py 4        # P0 patch tests -> patch.json
$P timing_driver.py 16     # cost: compile / passive / mj_step time -> timing.jsonl
$P upstream.py upstream.json; $P verify_upstream_force.py   # upstream-joint study (small models)
$P collect.py              # -> results.json (one record per model) + summary.csv; adds gt_fem error columns
$P plots.py                # -> plots/*.png
```
Files: `common.py` (geometry, XML, FD extraction, checks, surface integrals, solvers), `run_model.py` (one model),
`results.json` (full config, all cases, per-taxel maps, conditioning, timing, `patch_P0`, `error_vs_reference`),
`summary.csv` (flat table; the `err_<case>_vs_gt` columns hold relative errors vs gt_fem), `plots/`.

## Method (what was verified)
- **K extraction.** Central differences (step 10 nm) of `qfrc_spring`. Graph colouring (distance-2 on the
  element adjacency) needs only 108-750 `mj_passive` evaluations per model. Checks over all 295 models:
  symmetry error <= 7.2e-10 (max-abs, relative). Rigid-body null space: |K r|/(|K||r|) <= 1.4e-10. FD K against an
  independent assembly was <= 1.5e-9 (relative Frobenius): for full flexes, from MuJoCo's packed per-tet
  `flex_stiffness`; for interpolated flexes, from my own (o+1)^3 Gauss Lagrange-hex stiffness. FD K against MuJoCo's own
  assembled `efm_K` CSR (divided by h^2) was <= 1.6e-9. M from `mj_mulM` is diagonal (lumped vertex/node masses) to
  machine precision. Equilibrium (platen reaction against back-face reaction) <= 2.7e-12.
- **BCs.** Back face clamped. Platen/punch cases prescribe u at the free-face nodes inside the loaded region:
  G1 (u_z on the face, lateral faces on rollers), C1 (u = (0,0,d)), C1f (u_z only), C2 (u = (0,d,0)), C3 (y >= +5.07 mm),
  C6 (centre taxel; ns = no-slip, f = frictionless). The C4 punch is done two ways. **C4n** constrains u_z only at nodes
  with r <= a. **C4p** is the "through the interpolation" version: a quadratic penalty kappa * sum_q w_q (N_q u_z - d)^2
  at sub-sample points (spacing a/50) inside the disc, which enforces the punch on the interpolated surface (for
  trilinear/quadratic flexes this means the surface polynomial). kappa is 1e7 M/dz; the remaining gap is <= 0.22 %
  of d. The K change from 1e6 to 1e7 has a median of 1.3e-4, but reaches 9 % on 63 coarse meshes, so C4p values are
  accurate only to that level. The interpolated flexes' own nodes coincide with the grid, so nodal BCs are exact BCs
  on their DOFs.
- **Per-taxel maps.** Force per taxel = nodal reactions split by the free-face shape-function integrals over each
  taxel. Mean u_z per taxel = exact integral of the interpolated surface displacement.
- **Conditioning.** Extreme eigenvalues come from Lanczos and shift-invert (SuperLU). The PCG is an exact emulation
  of mj_effSolve (x0 = 0, stop at |r| < 1e-8 |b|, 3x3 vertex blocks). MuJoCo's own count was checked by running
  `mj_forward` with opt.iterations = N-1, N and reading mjWARN_INERTIA: the emulation equals MuJoCo's count, or MuJoCo
  needed exactly one iteration fewer, in 291/295 models. In 4 interpolated models MuJoCo needed more
  (quadratic_nt1_cz2/cz3_nu0.49, quadratic_nt2_cz1_nu0.499, trilinear_nt2_cz1_nu0.499).
- **Ground truth.** `../gt_fem/results.json` (finite-pad quadratic-hex FEM by the gt_fem agent; file as of 20:42, still
  being refined to level 3, so re-run `collect.py` when it finishes). C4 GT = their axisymmetric runs (disc
  R = 25.5 mm; their finite-size check is < 2e-7). Their stated K uncertainty is <= 0.02 % (C1, nu = 0.45), 0.05 % (C1,
  nu = 0.49), 0.25 % (C6). `../gt_transform/results.json` did not exist when I finished. G1 is compared with exact
  M A/t.

## Results

### P0 patch tests (patch.json, 154 models; F1 re-verified)
Energy 1/2 u'Ku for exact homogeneous strains, divided by the exact W V:

| variant | uniaxial strain | simple shear | hydrostatic |
|---|---|---|---|
| full_raw, nu=0.45 | 1.818182 | 1.000000 | 1.931034 |
| full_raw, nu=0.49 | 1.960784 | 1.000000 | 1.986577 |
| full_corr / trilinear / quadratic (0.45, 0.49, 0.499) | 1.000000 | 1.000000 | 1.000000 (max dev 2e-7) |

These raw ratios are exactly (2 lambda + 2 mu)/(lambda + 2 mu) and (2 lambda + 2 mu/3)/(lambda + 2 mu/3), i.e.
lambda_eff = 2 lambda. **F1 is confirmed** for every mesh tried, and the F1 input correction is exact, including at
nu = 0.499. G1 (confined compression) is exact to 9e-10 for every corrected/interpolated model. For every raw model it
is 1.8182 (nu = 0.45) and 1.9608 (nu = 0.49).

### Static stiffness vs ground truth, nu = 0.45 (relative error %, full table: summary.csv / results.json)
GT: C1 8.567, C1f 8.011, C2 0.8279, C3 3.594, C6ns 0.2170, C6f 0.2001, C4(a=5) 0.1476, C4(a=2.5) 0.03276 N/um.

| model | C1 | C1f | C2 | C3 | C6ns | C6f | C4p a5 | step ms* | PCG its |
|---|---|---|---|---|---|---|---|---|---|
| robot: full_raw 11x15x3 (nt2 nz3) | +85.2 | +77.3 | +0.5 | +100.3 | +343.7 | +349.6 | +509.7 | 1.5 | 36 |
| full_corr nt2 nz3 (same grid) | +4.4 | +4.8 | +0.5 | +14.2 | +157.3 | +170.0 | +266 | 1.4 | 30 |
| full_corr nt4 nz3 (2.55 mm) | +3.2 | +3.0 | +0.5 | +8.0 | +72.2 | +75.7 | +118 | 6.3 | 40 |
| full_corr nt8 nz3 (1.275 mm) | +2.6 | +2.3 | +0.5 | +5.0 | +36.0 | +35.2 | +65 | 30 | 66 |
| full_corr nt16 nz7 (0.64 x 0.25 mm) | +0.7 | +0.8 | +0.4 | +1.8 | +15.1 | +15.0 | +27.6 | 718 | 154 |
| trilinear nt4 cz2 | +2.4 | +1.9 | +0.5 | +5.7 | +41.5 | +42.0 | +92.6 | 7.9 | 53 |
| trilinear nt16 cz6 | +0.5 | +0.6 | +0.3 | +1.3 | +8.4 | +8.2 | +23.7 | 1812 | 136 |
| quadratic nt2 cz1 (5.1 mm, 1 cell thick) | +1.0 | +0.9 | +0.5 | +3.3 | +28.9 | +30.0 | +76 | 11.9 | 49 |
| quadratic nt4 cz1 | +0.8 | +0.8 | +0.4 | +1.9 | +13.6 | +13.0 | +53 | 62 | 62 |
| quadratic nt8 cz3 | +0.3 | +0.3 | +0.2 | +0.7 | +4.9 | +4.9 | +32 | 2735 | 157 |

\* `mj_step` median for the pinned pad alone (timing.jsonl). Timings were taken while 16 sweep processes and other
agents ran, so treat them as indicative. "PCG its" = 3x3-block PCG iterations on the gravity right-hand side.

nu = 0.49 (GT C1 33.77, C3 13.15, C6ns 0.4843): the robot grid raw is C1 +100.7 %, C3 +120.6 %, C6ns +655 %.
Corrected on the same grid: +9.5 / +24.1 / +336 %. full_corr nt16 nz7: +1.5 / +3.7 / +41.6 %. trilinear nt16 cz6:
+1.1 / +3.1 / +25.7 %. quadratic nt8 cz3: +0.5 / +1.4 / +11.7 %.
nu = 0.499 (subset nt = 2/4/8): all low-order variants lock. C1 is +35-39 % for full_corr with 2 elements
through the thickness (nz3) and +108 % for trilinear cz1. At the finest thickness tried it is still +5.5 % (full_corr
nz7) and +12.7 % (trilinear cz3). Quadratic cz3 gives +0.9 % (C1) but +30 % on C6ns at nt8.

What the numbers say:
1. **On the robot grid, F1 is the dominant error for distributed loads** (+85 % C1 at nu = 0.45, +101 % at 0.49).
   Shear (C2) is unaffected and is within 0.55 % of GT on every mesh.
2. With F1 corrected, **through-thickness refinement saturates at 2-3 elements** for whole-face/half-face loads
   (right-hand panels of plots/convergence_nu*.png). One element is enough only for C1/G1-like (uniaxial-strain)
   fields; at nu = 0.49 one element gives +22 % on C1.
3. **Localized loads converge slowly with in-plane size** (C6, C4: rigid-punch edge singularity). Nodal BCs also spread
   the imposed displacement one element beyond the loaded region, so the effective punch is larger by about one cell
   on each side. At 5.1 mm cells, a 5 mm disc and a 10.2 mm square are the same problem: C4p_a5mm = C6f to within 4e-6
   (the penalty residual) for every nt2 model. Even 0.64 mm linear tets are 15 % stiff on C6. Quadratic hexes do best per DOF (5 % at
   1.275 mm). C4n (nodes inside r <= a) comes within a few percent on the finest meshes at a = 5 mm
   (-2.6 % quadratic nt8 cz3, +1.1 % full_corr nt16 nz7, -7.2 % trilinear nt16 cz6). C4p (penalty on the
   interpolated surface) stays 24-32 % stiff there. Neither is reliable on coarse meshes.
4. **Cross-talk (C6; the user's "+1,+2,+1" question).** Ground truth for a rigid square punch on the centre taxel: the
   4 edge-neighbour taxels move **-0.034** of the punch depth at nu = 0.45, and **-0.127** at nu = 0.49 (no-slip). This
   is a small outward bulge: the continuum does **not** make neighbours follow the pressed taxel. On the robot grid
   (raw or corrected), the neighbours instead move **+0.22 to +0.25 d**. That is pure interpolation spill-over: the
   pressed taxel's edge nodes are shared with its neighbours. The two neighbours are also unequal (0.252 vs 0.223),
   because the flexcomp tet split is anisotropic. The correct small bulge is approached only on fine meshes:
   full_corr nt16 nz7 gives +0.002 to -0.008, trilinear nt16 cz6 -0.017, quadratic nt8 cz3 -0.024 (GT -0.034). So on the robot grid "per-cell" deformation is a mesh
   artefact, not physics (plots/taxel_maps_nu0.45.png, row 4).
5. **The F2 contact-rig numbers are inconsistent with the discretisation.** The exact-BC C1 of the uncorrected full
   flex is 16.65 (11x15x2), 15.87 (11x15x3), 15.42 (21x29x4) and 15.19 (41x57x5) N/um. F2 reported 18.9 / 16.9 /
   15.6 / 15.1. For 11x15x2, 18.9 exceeds the doubled-lambda bound of 16.65, which no displacement BC on that mesh can
   exceed. So F2's coarse-grid values contain a contact or measurement artefact (its K was force / measured
   penetration in a force-controlled rig, rig2.py). I did not investigate the cause further.

### Conditioning and solver convergence (plots/conditioning.png, columns in summary.csv)
- **h * omega_max = 22-916 >> 2 (nu = 0.45/0.49; up to 2072 at 0.499).** No explicit scheme can run this pad at
  0.5 ms. omega_max is set by the node spacing
  through the thickness: it follows 2 c_p / dz, with c_p = sqrt(M/rho) = 63 m/s (nu = 0.45) and 133 m/s
  (nu = 0.49). It barely depends on the in-plane size. h * omega_min is 2.2-9.5, so even the lowest mode is stiff
  relative to h.
- **cond(M + h^2 K)** grows like (h omega_max)^2: from 34 to 6.1e4 at nu = 0.45/0.49, and up to 3.4e5 at nu = 0.499.
  It is dominated by dz. At fixed dz it even falls slightly with in-plane refinement, because h omega_min rises from
  ~4.8 toward ~8.3.
- **3x3-block-Jacobi PCG** (what mj_effSolve uses when the flex is assembled, i.e. on a fixed body) needs 12-419
  iterations at nu = 0.45/0.49 and 81-1154 at nu = 0.499. It grows with both refinements. It exceeds MuJoCo's
  default opt.iterations = 100 (so mjWARN_INERTIA "too ill-conditioned" appears in every step) for:
  - nu = 0.45: full_corr at nt4/6 nz7, nt8 nz6-7, nt12 nz4-7, and nt16 for every nz; trilinear at cz6 for every
    nt, plus cz4 at nt2/4/16; quadratic at cz3 for every nt, nt6 cz2, and nt8 for every cz.
  - nu = 0.49: many more (e.g. full_corr nt2 nz7, trilinear nt8 cz3).

  The robot grid needs 36 (raw) / 30 (corrected) iterations at nu = 0.45 and 60 / 48 at nu = 0.49. On the pinned pad
  alone the robot grid converges.
- **Element aspect ratio and the preconditioner.** The 3x3 vertex blocks capture the stiff through-thickness coupling
  well only when the elements are flat (dx >> dz). For full_corr nu = 0.45 with nz = 7:
  - at fixed dz, cond(P^-1 A) rises monotonically as the cells are refined in-plane: 78 (nt1), 96, 125, 172 (nt8),
    277 (nt16). With nz = 2 it goes from 2.3 to 209;
  - at fixed in-plane size, it rises with nz.

  So PCG iterations grow with **both** refinements. Large aspect ratios (up to 41 here) caused no accuracy problem for
  these loads (patch tests exact; C1 at nt1 nz7 is +5 %, set by in-plane resolution). "Overmeshing" in-plane below
  ~2.5 mm raises PCG iterations and step cost but changes distributed-load stiffness (C1, C3) by only a few percent;
  it pays off only for localized loads (C6, C4).
- **Upstream joint (upstream.json, verify_upstream_force.json; relevant to F4).** I put one slide or hinge joint on the
  pad's parent body:
  - `d.nefmK = 0`, `d.nefmdof = 0`: the flex stiffness is no longer assembled into the effective-metric CSR. The
    metric still contains h^2 K: MuJoCo's iteration count matches the emulated PCG on M_gen + h^2 K_gen within 1.
  - The preconditioner falls back to M^-1 only. cond(M^-1 A) jumps from 368 to 8262 (robot-like grid, corrected), and
    from 669 to 15017 (raw).
  - With a hinge, MuJoCo needs 104 (corrected) / 132 (raw, robot grid) / 202 (nu = 0.49) / 217 (nt2 nz5) iterations.
    That is more than 100, so the warning appears in the first step.
  - This explains the F4 warning. It does **not** by itself explain the divergence (F4 saw divergence even with 1000
    iterations), so that remains open.
  - The mass-only rows of plots/conditioning.png show the same preconditioner on the pinned pad: 44-1189 iterations
    over all nu = 0.45/0.49 models.
  - The slide-joint "1 iteration" entries are trivial: the gravity rhs is a rigid translation.
- **New bug-level finding (interpolated flexes with an upstream joint).** In `src/engine/engine_passive.c` (3.17.0,
  lines ~163-172, and ~405-414 for shells), node forces of interpolated flexes take a "fast path" that writes only to
  the node body's own 3 DOFs (`mji_addTo3(d->qfrc_spring + m->body_dofadr[bid], ...)`). They are not propagated to
  ancestor joints (`mj_applyFT` is used only for pinned/offset nodes).
  - Measured (verify_upstream_force.json, random 1 um deformation, slide-z joint on the parent): generalized force on
    the parent joint Q_base = +2.534 N for trilinear and +0.602 N for quadratic, i.e. exactly -sum(free-node forces).
    The correct value is 0 by momentum balance. The full flex gives 5e-16.
  - Consequence: the FD generalized K is non-symmetric (relative asymmetry 0.06-1.0), and M^-1(M + h^2 K) has negative
    eigenvalues (-7.7 to -11). The emulated PCG breaks down (pAp <= 0) at iteration 2.
  - So interpolated flexes are not momentum-consistent on the robot, independently of F3.

### "Best cell structure" for this layer (static operator only; contact is studied elsewhere)
- **dof="full" (needed for per-vertex contacts):** apply the F1 input correction. Use 2-3 elements through the
  thickness (nz = 3-4). In-plane size is set by the load size you need to resolve:
  - 2.55 mm (21x29) gives C1 / C3 within 3 % / 8 %, but punch-type loads 70 %+ too stiff.
  - 1.275 mm (41x57) gives 2.6 % / 5 % / 36 %, at 30 ms per step and 66 PCG iterations.

  Below that, step cost and PCG iterations rise faster than accuracy for anything but localized loads.
- **Per DOF, quadratic is the most accurate** (nt2 cz1 beats full_corr nt8 nz3 on every case except C4p, at 0.4x
  the time per step). But interpolated flexes carry F3 (contacts reduced to 50 per geom) and the upstream-joint force bug
  above.
- **nu -> 0.5:** only quadratic avoids locking. Linear tets need >= 4-6 elements through the thickness even at 0.49,
  and PCG iterations exceed 100.

## Not done / caveats
- No contact here by design. C5 (sphere) is not a no-contact case.
- Timings are indicative only (shared machine). Timing runs cover nu = 0.45 and the robot raw grid only.
- quadratic nt12/16 were not run (memory/time). trilinear and full go to nt16.
- GT values were taken from gt_fem's interim results.json (20:42). Re-run `collect.py` + `plots.py` after gt_fem's
  level-3 run finishes.
- The C4p penalty values on coarse meshes are sensitive to kappa (up to 9 %, see above).
