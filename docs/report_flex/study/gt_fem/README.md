# gt_fem: ground-truth FEM for the finite bonded PDMS pad (no MuJoCo involved)

This directory is the reference solution for the finite 51 x 71 x 1.5 mm pad, in the cases where the pad edges matter
(C1, C1f, C2, C3). It also gives C6 and C4 as a cross-check of the exact infinite-layer solution in ../gt_transform.
The model is linear elastic and small strain, with E = 1 MPa and nu = 0.45, 0.49 and 0.499. The back face (z = 1.5 mm)
is clamped. The four lateral faces are traction-free: the pocket walls are not modelled, which matches the MuJoCo flex,
whose only pins are on the back face. Loads are imposed displacements of a rigid platen or punch, and every force is
given per unit imposed displacement.
Frame: origin at the centre of the free face. x runs across the pad (|x| <= 25.5 mm), y along it (|y| <= 35.5 mm),
and +z points into the layer (positive force = pressing into the skin, positive u_z = indentation, negative u_z = bulging out).
Taxel arrays are stored as [j][i][component], with j = row along y (0 at y = -35.5) and i = column along x.

## Files
| file | role |
|---|---|
| `fem_layer.py` | Checked copy of `<scratch>/fem_ref/fem_layer.py` (the original is kept as `fem_layer_orig.py`). It uses 27-node Q2 hexes in two forms: a full displacement form, and a Q2/P1disc mixed (B-bar) form with the pressure condensed. Tensor-product graded box meshes, Kronecker assembly, nested-dissection Cholesky. The header lists the review and the changes: a `graded_breaks` bug fix, a faster solver, threads, and MMS helpers. |
| `cases.py` | Case definitions (G1, C1, C1f, C2, C3, C6, C6f), meshes, BCs, quarter/half symmetry, per-taxel and profile post-processing. |
| `run_cases.py` | Sweep driver: `runs/CASE_FORM_nuNU_mM.json`, `logs/`. |
| `axi_fem.py`, `run_c4.py` | Axisymmetric Q2 code (full and mixed forms) for C4: `runs_c4/`, `c4_verify.json`. |
| `verify_fem.py` | Code verification: `verify_fem.json` (V1-V7, below). |
| `postprocess.py` | Richardson extrapolation and error bars, pressure method, comparison with gt_transform: `results.json`, `summary_table.md`. |
| `plots.py` | `fig/*.png`. |
| `refs_download/` | The opened source for the pressure method (Kelly & Van Engelen, PEER 2015/03; PDF plus text dump). |
| `PROGRESS.md` | Session log. |

Rerun (about 2.5 h on 56 threads; level 3 needs up to 154 GB RAM for C3):
```
PY=/home/dohyun/miniforge3/bin/python
$PY verify_fem.py v1 v2 v3 v4 v4b v5 v6 v7        # code verification -> verify_fem.json
$PY run_c4.py verify; $PY run_c4.py runs          # C4 axisymmetric
$PY run_cases.py sweep --jobs 2 --threads 28 --levels 1 2
$PY run_cases.py sweep --jobs 1 --threads 48 --levels 3 --reduced   # level 3: full forms, mixed 0.499, all of C1
$PY postprocess.py; $PY plots.py
```

## Meshes, accuracy and error bars
- Every element is a 27-node Q2 brick. Meshes are graded geometrically (growth 1.5) toward every singular line: the
  pad edges (where the free face meets a clamped or free lateral face), the platen edge of C3 (y = +5.0714 mm) and the
  punch edges of C6. The base element size at those lines is 0.1 mm, the largest in-plane element is 3 mm, and the
  largest through the thickness is 0.5 mm.
- At level m = 1 there are 7 Q2 elements through the 1.5 mm in C1, C1f, C2 and C3 (graded toward both faces) and 5 in
  C6. Level m splits every element into m^3, so the levels are nested and geometrically similar. m = 1, 2, 3 give 21, 42
  and 63 elements through the thickness.
- Model sizes: C1, C1f and C2 use a quarter or half model (anti-symmetric in y for C2) with 54k, 409k and 1.35M DOF. C3
  is a half model with 3.39M DOF at m = 3. C6 and C6f are quarter models with 1.63M DOF at m = 3.
- Solver: the sparse direct solver is a nested-dissection Cholesky in fem_layer.py, checked against scipy SuperLU (V5).
  The largest solve is C3 at m = 3: 1094 s and 154 GB peak RSS (logs/sweep3_C3.log, runs/*.json `maxrss_GB`).
- K is computed two ways, as the platen reaction and from the strain energy (2U/delta^2). The two agree to the printed
  digits at every level (results.json `K_energy_N_per_um_levels`).
- Error bars: Richardson extrapolation over m = 1, 2, 3, solving for the observed order p. The error bar is the
  GCI-type 1.25 |K* - K_m3|, and it is also maxed with |K*_full - K_mixed| wherever both forms were run. Two
  independent discretisations with the same limit therefore have to agree within the bar.
- Observed orders: p = 1.2-1.5 for all 3D cases. This is the expected rate: the corner where the free face meets the
  bonded or lateral face, and the punch edge, carry stress singularities, so the asymptotic rate under uniform
  refinement of a fixed graded mesh is below the smooth-solution rate. The axisymmetric C4 gives p = 1.0.
- Exception: C1 at nu = 0.499 (mixed form) has level differences 1.57e-3 and 1.26e-3 N/um, too close together to fit
  any p > 0 with m = 1, 2, 3. It is flagged "no order found", and the reported value is the finest level with a bar of
  2|K_m3 - K_m2| = 2.5e-3 N/um (1.3e-5 relative).
- Taxel maps and profiles come from the finest level. The per-entry error is |Q_m3 - Q_m2| x max(1, 1.25/(r^p - 1)).
  The profile error is the maximum pointwise change on the coarse nodes. It is largest (0.015-0.44 delta) at the nodes
  next to the C3 and C6 contact edges, where u_z has a singular gradient; away from the edges it is much smaller.

## Code verification (verify_fem.json, c4_verify.json; all rerun or checked this session)
| check | result |
|---|---|
| V1: Kronecker assembly vs independent element-by-element isoparametric 3x3x3 Gauss (both forms, nu 0.3-0.499) | max rel. diff 1.0e-15 |
| P0 / V2: patch tests (uniaxial strain, simple shear, hydrostatic; full and mixed; nu 0.3, 0.45, 0.499; 36 tests) | interior nodal error <= 6.3e-13, energy ratio 1 +- 8e-12 |
| G1 / V3: laterally confined layer vs exact M A/t | rel. err <= 1.3e-14 (9.15655 N/um at nu 0.45) |
| V4: MMS (smooth sine field, uniform and graded meshes) | L2 rate -> 3.17, H1 rate -> 2.06 (theory 3 and 2) |
| V4b: solenoidal MMS (div u = 0, f = -mu lap u, lambda-free; uniform and graded meshes, m = 1, 2, 4, 8) | Mixed form: finest rates 3.00 (L2) and 2.00 (H1), and the error is identical at nu = 0.45, 0.499 and 0.49999 (graded L2 2.88e-5), so it is locking-free. Full form: L2 error 2.90e-5 at nu = 0.45, 1.42e-4 at nu = 0.499 (5x) and 6.04e-4 at nu = 0.49999 (21x), where its rates drop to 2.11 (L2) and 1.18 (H1). That is volumetric locking, which is why the mixed form is used at nu = 0.499. At nu = 0.49 the full and mixed values agree within the bars (results table). |
| V5: nested-dissection Cholesky vs SuperLU | rel. diff in u <= 5.8e-14 |
| V6: symmetry-reduced vs full model | full form <= 6e-11. Mixed form at nu = 0.499: 2-3e-7 (round-off from conditioning, lambda/mu = 998), far below the discretisation error |
| V7: graded-mesh generator (the original sampled the density too coarsely for hmin < 1e-3 span) | first element within 20-29% of the requested hmin (0.2, 0.02 and 0.002 mm), neighbour ratio <= 1.64 |
| Axisymmetric C4 code: patch tests and confined disc | nodal error <= 8.7e-13 |

The F1 claim from the main session was re-derived for this code: the energy is lam (div u)^2/2 + mu eps:eps, and V2
recovers the exact stresses, so this FEM has no factor-2 lambda error. The closed forms in ../gt_transform give the same
patch ratios that F1 reports for MuJoCo (1.8182 and 1.9310 at nu 0.45).

## Results (results.json; K in N/um per unit imposed displacement; value +- error bar)
| case | nu = 0.45 | nu = 0.49 | nu = 0.499 (mixed) | Winkler / bound line (comparison only) |
|---|---|---|---|---|
| G1 confined (exact) | 9.156552 (FEM 1e-14) | 41.313423 | 403.40694 | = M A/t |
| C1 no-slip platen, normal | **8.5662 +- 0.0007** (p 1.41) | **33.757 +- 0.007** | **194.329 +- 0.003** | M A/t = 9.157 / 41.31 / 403.4 (rigorous upper bound; +6.9% / +22% / +108%) |
| C1f frictionless platen, normal | **8.0102 +- 0.0005** | **27.252 +- 0.005** | **91.727 +- 0.002** | same bound |
| C2 no-slip platen, shear along y | **0.82791 +- 0.00002** | **0.80602 +- 0.00005** | **0.801255 +- 0.000003** | G A/t = 0.8324 / 0.8101 / 0.8052 (FEM 0.5% lower: free edges) |
| C3 platen over y >= 5.07 mm, no-slip | **3.5927 +- 0.0007**, F_y = -0.0248 | **13.139 +- 0.007**, F_y = -0.0546 | **55.376 +- 0.010**, F_y = -0.137 | Winkler 3.924 / 17.71 / 172.9 (+9% / +35% / +212%) |
| C6 centre-taxel punch, no-slip | 0.21663 +- 0.00031 | 0.4822 +- 0.0015 | 0.73244 +- 0.00075 | Winkler 0.2616 / 1.180 / 11.53 |
| C6f centre-taxel punch, frictionless | 0.19976 +- 0.00028 | 0.34352 +- 0.00068 | 0.421085 +- 0.000049 | as above |
| C4 a = 5 mm frictionless (axisymmetric) | 0.14757 +- 4e-5 | 0.24139 +- 1.2e-4 | 0.28798 +- 1e-5 | |
| C4 a = 5 mm no-slip | 0.16120 +- 5e-5 | 0.33947 +- 2.6e-4 | 0.48832 +- 1e-4 | |
| C4 a = 2.5 mm frictionless | 0.032764 +- 2e-5 | 0.042580 +- 4e-5 | 0.045993 +- 3e-6 | |
| C4 a = 2.5 mm no-slip | 0.034881 +- 2e-5 | 0.050831 +- 7e-5 | 0.057828 +- 2e-5 | |

All other force components are zero by symmetry, to round-off. The one exception is C3, which has an in-plane F_y on
the layer: F_y < 0, i.e. toward -y, the uncovered part of the pad.
- The full and mixed forms agree within the bars wherever both exist. For example, C1 at nu 0.45 gives Richardson
  values of 8.566157 (full) and 8.566102 (mixed).
- C3/C1 = 0.419, 0.389 and 0.285 against an area fraction of 3/7 = 0.429. The platen edge inside the pad costs
  stiffness, and more so as nu -> 0.5. Outside the platen the free face bulges out: the minimum u_z at x = 0 just
  outside the platen edge is about -0.25, -1.0 and -3.5 delta for nu = 0.45, 0.49 and 0.499 (fig_profiles.png).
- Taxel force maps (5 x 7, three components, at the contact face and at the bonded face) are in results.json
  (`taxel_force_N_per_um`, `taxel_force_on_blade_N_per_um`, `taxel_mean_disp_per_delta`, each with an `_err` array).
  Profiles u(y) at x = 0 and u(x) at y = 0 are in `u_free_at_x0_per_delta`, `u_free_at_y0_per_delta` and the
  `t_free_*` tractions.
- C1 at nu = 0.45: the centre taxel carries 0.2616 N/um and a corner taxel 0.2153. At nu = 0.499 those become 9.48 and
  2.07, because the pressure profile turns parabolic (fig_C1_traction.png).
- C6 cross-talk (nu = 0.45, no-slip). The contact force exists only on the centre taxel, because that is where the
  punch is. The neighbouring taxels move outward: their mean u_z is -0.035 delta (-0.20 delta at nu = 0.499).
  At the bonded face, the centre taxel takes 88.4% of the force (no-slip) or 84.2% (frictionless), and the rest spreads
  to its neighbours: 5.85-5.89 mN/um F_z on each edge neighbour, against 191.75 at the centre (finest level). The
  bonded face also carries in-plane shear under a pure normal punch: +-8.46 mN/um F_y on the y-neighbours
  (fig_taxels_C6.png). On the user's question ("cells 1,2,3 deform +1,+2,+1, is contact per cell?"): in the continuum, a
  single-cell indentation does not deform the neighbours inward. It bulges them outward without any contact force on
  them. Any contact force a readout puts on the neighbours is therefore an artefact of the readout or of the
  discretisation, not of the elasticity.

## Classical thin-layer estimate for C1 (pressure method)
Source opened: J.M. Kelly and N.C. Van Engelen, "Single series solution for the rectangular fiber-reinforced
elastomeric isolator compression modulus", PEER Report 2015/03, sec. 2.1, eqs. (2.1)-(2.21)
(https://peer.berkeley.edu/sites/default/files/webpeer-2015-03_james_m._kelly_and_niel_c._van_engelen_.pdf;
local copy in refs_download/). It assumes parabolic bulging, sigma_xx = sigma_yy = sigma_zz = -p, and
compressibility tr eps = -p/K. With rigid reinforcement (E_f -> infinity in eq. 2.21) this gives
lap p - 12G/(K t^2) p = -12 G (delta/t)/t^2, with p = 0 on the edges. I solved it by a single cosine series and checked
the series against a finite-difference solve, which agrees to 4e-6 relative.

| nu | FEM C1 | pressure method | error | M A/t (Winkler, upper bound) |
|---|---|---|---|---|
| 0.45 | 8.5662 | 7.3372 | -14.3% | 9.1566 (+6.9%) |
| 0.49 | 33.757 | 32.487 | -3.8% | 41.313 (+22%) |
| 0.499 | 194.33 | 193.56 | -0.39% | 403.41 (+108%) |

The pressure method assumes that the in-plane stresses equal -p, which is accurate only for near-incompressible
layers. At nu = 0.45 it misses the uniform confined part of the response (fig_C1_traction.png: the FEM traction
plateaus at M/t in the interior, while the pressure method plateaus 12% lower).

## Comparison with gt_transform (exact infinite layer; ../gt_transform/results.json, read on 2026-10-31)
- C4 (axisymmetric FEM vs Fredholm / ring BEM), all 12 combinations: rel. diff 2e-6 to 2.7e-5. Every difference is
  below 0.06 of the FEM bar, so the GCI bars are conservative.
  Finite-size effect, from disc radius 25.5 mm vs 51 mm: |dK/K| <= 2e-7.
- C6 and C6f (3D finite-pad FEM vs rectangular BEM): rel. diff 7e-5 to 1.2e-3. Each difference is 0.14-0.38 of the
  combined error bar.
  - Base-force fraction maps agree to <= 6.5e-4 (max abs) and taxel-mean u_z/delta to <= 1.3e-3.
  - Example, nu = 0.45 no-slip: centre base-force fraction 0.8841 (FEM) vs 0.8845 (transform); x-neighbour mean u_z
    -0.0346 vs -0.0349 delta.
  - Every FEM K is slightly above the transform value. That is consistent with discretisation error from above and
    with a finite-pad softening that is too small to see.
  - The finite-pad effect on C6 is therefore below the FEM resolution, < ~2e-4 relative at nu 0.45. The punch edge is
    >= 20 mm (13 t) from the pad edge.
- Conclusion: for loads more than a few t from the pad edge, the infinite-layer transform solution is the more
  accurate reference. The finite-pad FEM is needed for C1, C1f, C2 and C3, where the pad or platen edge carries the
  answer. In C1 at nu = 0.45, the edge zone lowers K by 6.4% below the infinite-layer value M A/t.

## Relation to the main-session findings
- **F2**: the bound M A/t = 9.157 N/um holds: the converged C1 is 8.5662 +- 0.0007, and C1f 8.0102, both below it.
  The earlier fem_ref values of 8.593 and 8.578 lie 0.3% and 0.14% above the converged value. That is consistent with
  this code converging from above, but their meshes are not recorded here. The uncorrected MuJoCo K_n of 15.1-18.9 N/um
  is 1.76-2.21x the truth. The MuJoCo K_s of 0.862-1.02 N/um compares with a true C2 of 0.82791.
- **F1**: not testable here, since this code is independent of MuJoCo, but the reference values it needs are given above.

## Not done / caveats
- V4b first failed with `NameError: name 'j' is not defined`, a scoping bug in a nested generator in verify_fem.py. I
  fixed it and reran it this session (logs/verify_v4b.log).
- At level 3 the mixed form was run only for nu = 0.499, and for C1 at all three nu. For the other cases, mixed nu 0.45
  and 0.49 stop at m = 2 and serve only as the cross-form check.
- C4 was modelled on a disc of radius 25.5 mm, the inscribed circle of the pad. A 51 mm disc changes K by < 2e-7, so
  the rectangular outline is immaterial.
- The tractions on the free face are singular at the pad, platen and punch edges. The plotted traction profiles are
  clipped there, and the pointwise values next to an edge are not converged. The integrated forces (K, taxel forces)
  are converged.
