# gt-transform: exact ground truth for local loads on the bonded PDMS layer

This directory gives exact results for an infinite linear-elastic layer (thickness t = 1.5 mm, E = 1 MPa,
nu = 0.45 / 0.49 / 0.499) perfectly bonded to a rigid base. The free surface carries normal and tangential
tractions. MuJoCo is not used anywhere. Everything is numpy/scipy (MuJoCo-independent; no venv needed).
Conventions: the origin is at the centre of the free face, z points into the layer, u_z > 0 means indentation,
SI units in the files, stiffness reported in N/um.

## Files
| file | content |
|---|---|
| `gt_transform.py` | exact surface/base transfer functions (Papkovich-Neuber), Hankel tables of the real-space Green's functions, half-space closed forms (Boussinesq, Cerruti, Love rectangle, Sneddon, Hertz), rectangle and ring influence elements, closed forms P0/G1 |
| `derive_sympy.py` -> `derive_sympy.out` | symbolic check: Navier residual 0, boundary conditions 0, both limits |
| `validate.py` -> `validate.json/.out`, `fig_kernels.png`, `fig_green_realspace.png` | validations V1-V7 (below) |
| `gt_contact.py` | contact solvers: Fredholm 2nd-kind (axisymmetric), ring BEM (frictionless + bonded), rectangular BEM (square punch, frictionless + no-slip), exact Fourier post-processing (taxel means, base reactions), spectral check of the rectangle correction |
| `run_all.py` | runs every case (pool of 6 x 8 threads) -> `runs/*.pkl`; `python run_all.py --c5` recomputes only C5 |
| `assemble.py` | `runs/*.pkl` -> `results.json`, `summary.txt`, `fig_c4_punch.png`, `fig_c5_sphere.png`, `fig_c6_profiles.png`, `fig_c6_taxels.png`, `fig_c6_convergence.png`, `fig_literature.png` |
| `lit_asymptotic_check.py` -> `.out` | small-alpha coefficients compared with the published asymptotics |
| `compare_gt_fem.py` -> `compare_gt_fem.json/.out` | cross-check against the finite-pad FEM of agent gt-fem (its results.json, levels 1-2) |
| `compare_gt_fem_levels.py` -> `compare_gt_fem_levels.json/.out` | same, against every gt-fem run file incl. level 3 (`../gt_fem/runs/C6*_m*.json`, finished 21:45-21:52, after gt-fem's results.json was assembled) |
| `lit/` | `argatov_1203.0918.pdf` (arXiv, opened), `a.txt` (pdftotext), `argatov_table1.json` (the extracted Table 1) |
| `PROGRESS.md` | checkpoint log |

To rerun: `cd` here, then `OMP_NUM_THREADS=8 python run_all.py --workers 6 && python assemble.py && python compare_gt_fem.py && python compare_gt_fem_levels.py`
(about 33 min wall time on the shared machine; the no-slip n=96 C6 solves take up to 32 min each).
Then `python validate.py` (about 1 min) and `python derive_sympy.py`.

## 1. Closed forms (P0, G1, infinite layer shear, half-space)
A = 51 x 71 mm^2. M = lambda + 2 mu.
| nu | G1 confined K = M A/t | same with lambda doubled (MuJoCo F1) | simple shear G A/t | uniaxial-strain ratio (2lam+2mu)/M | hydrostatic ratio (6lam+2mu)/(3lam+2mu) | F1 workaround (young, poisson) |
|---|---|---|---|---|---|---|
| 0.45 | 9.1566 | 16.648 | 0.8324 | 1.8182 | 1.9310 | 971787 Pa, 0.409091 |
| 0.49 | 41.313 | 81.007 | 0.8101 | 1.9608 | 1.9866 | 993552 Pa, 0.480392 |
| 0.499 | 403.41 | 805.20 | 0.8052 | 1.9960 | 1.9987 | 999336 Pa, 0.498004 |

These reproduce the main session's F1 patch-test ratios (1.8182 and 1.9310 at nu = 0.45) and its workaround inputs.
They also reproduce F2's bounds (9.157 and 16.65 N/um). P0 patch tests are exact: sigma_zz = M eps (uniaxial strain),
tau = mu gamma (simple shear), mean stress = 3K eps (hydrostatic). Half-space limits used below: Boussinesq
u_z = (1-nu)/(2 pi mu r); Cerruti u_x = [(1-nu) + nu x^2/r^2]/(2 pi mu r); Sneddon K = 2 E a/(1-nu^2);
Hertz F = (4/3) E/(1-nu^2) R^1/2 delta^3/2 with a = sqrt(R delta).
*Fix made this session:* `closed_forms()` had the hydrostatic ratio as (9lam+6mu)/(4.5lam+3mu) (= 2.000). That was wrong
and is now corrected to (6lam+2mu)/(3lam+2mu).

## 2. Exact layer Green's function (derivation sketch)
Take the 2-D Fourier transform in (x, y). For each wavenumber k, the Navier equations become ODEs in z. Solve them
with Papkovich-Neuber potentials (P-SV part) plus a decoupled SH part. Impose u = 0 at z = t and the prescribed
tractions at z = 0. With u = kt and kap = 3 - 4 nu:

    D' = 2 kap cosh 2u + 1 + kap^2 + 4u^2
    C_zz = (1-nu)/(mu k) (2 kap sinh 2u - 4u)/D'       (u_z <- p)
    C_L  = (1-nu)/(mu k) (2 kap sinh 2u + 4u)/D'       (longitudinal in-plane)
    C_T  = tanh(u)/(mu k)                              (transverse in-plane, SH)
    C_c  = [kap (kap-1)(cosh 2u - 1) - 4u^2]/(2 mu k D')   (normal-tangential coupling)
    u^_a = [C_L kh_a kh_b + C_T (d_ab - kh_a kh_b)] q^_b + i C_c kh_a p^ ,   u^_z = -i C_c kh.q^ + C_zz p^

The base-traction transfer functions B_zz, B_zx, B_xz, B_LL, B_TT are in the `gt_transform.py` header.
mu k C_zz/(1-nu) is exactly the published kernel L(lambda) quoted from Vorovich/Argatov
(arXiv:1203.0918, sec. 1.2). Limits:
* kt -> oo: C_zz = C_L = (1-nu)/(mu k), C_T = 1/(mu k), C_c = (1-2nu)/(2 mu k). These are the Boussinesq/Cerruti
  transforms.
* kt -> 0: C_zz -> t/M, C_L, C_T -> t/mu.
Real space = half-space closed form + correction. The correction is the Hankel transform of the exponentially
decaying difference, tabulated on a 15 um radial grid.

**Validations (validate.out, derive_sympy.out):**
* V1: closed forms vs an independent Chebyshev-collocation solve of the raw Navier ODEs (no potentials), kt = 0.01..40.
  Worst relative difference 5.4e-8, 2.7e-6, 1.2e-7 (nu = 0.45, 0.49, 0.499).
* V2: both limits hold to <= 1.7e-8.
* V3: real-space tables integrate to t/M and t/mu exactly, and base tractions sum to the load.
* V4: half-space rectangle closed forms vs dblquad: 8e-14.
* V5: ring closed forms <= 2.6e-5. The entry `disk_w r=0.999` is NaN in validate.json. That NaN is a failure of the
  *reference* quadrature (scipy `ellipk(1)` hit exactly at s = r), not of the closed form. Re-checked with a split
  quadrature using `ellipkm1`: 5.2e-15.
* V6: Struve-form annulus transform: 4e-15.
* V7: square patch on the layer, split method vs a direct 2-D Fourier quadrature: <= 1.1e-7.
* New this session: all 8 rectangle-correction components (zz, zx, zy, xz, xx, xy, yz, yy) vs an independent polar
  Fourier quadrature of the kernel differences agree to <= 1.4e-9 of the element response (`gt_contact.rect_correction_spectral`).
  This checks the signs of the coupling terms used by the no-slip solver.

**scipy bug found:** `scipy.special.struve(0, x)` (scipy 1.17) returns NaN near its zero x = 25.76536. It is
guarded in `gt_transform._struve` by the mean of the two neighbours at x(1 +- 1e-6).

## 3. Contact solvers and convergence
* **C4/C5 (axisymmetric, frictionless): Fredholm equation of the 2nd kind.** Represent H0[p](k) = int_0^a g cos(k tau).
  Then g(tau) + (2/pi) int_0^a K(tau,s) g(s) ds = (2/pi) mu/(1-nu) h*(tau), with K = [H(tau-s)+H(tau+s)]/2 and
  H(x) = int_0^oo (mu k C_zz/(1-nu) - 1) cos kx dk. h* = delta for the flat punch and delta - tau^2/R for the sphere.
  The force is F = 2 pi int g, and the pressure is p(r) = g(a)/sqrt(a^2-r^2) - int_r^a g'/sqrt(tau^2-r^2).
  For the sphere, the contact radius comes from g(a) = 0 (bounded pressure, found by root finding at each delta).
  Gauss-Legendre Nystrom converges exponentially: N = 24/48/96 agree to 1e-13 (C4), and N = 12 already matches N = 96
  to 1e-15 (C5).
  Checks:
  - thick layer (t = 10 m): Sneddon to 5e-4 (the remaining difference is the a/h = 5e-4 thickness effect) and Hertz to 5e-5;
  - the surface displacement recomputed from g reproduces h(r) under the indenter (<= 3e-14 relative for C4, <= 2e-5 um for C5);
  - p >= 0 everywhere for the sphere (minimum 1.7-2.6 kPa at r = 0.999a, at delta = 30 um);
  - independent ring BEM: flat punch Richardson-extrapolated value within <= 4e-9 of Fredholm; sphere with an active
    set, F within <= 6.6e-7.
* **C4 bonded/no-slip:** ring BEM with p and q_r piecewise constant on sine-graded annuli, n = 100..800. Richardson order
  2.0, estimated error <= 1.3e-6. Validation: on a half-space it reproduces the Mossakovskii/Spence ratio
  K_bonded/K_frictionless = (1-nu) ln(3-4nu)/(1-2nu) for nu = 0, 0.3, 0.45, 0.49 to <= 4e-6.
  I used this formula from memory and did not open a source for it; the 1e-6 agreement is the evidence for it.
* **C6 (square punch over taxel (2,3), 10.2 x 10.14 mm):** rectangular BEM with quarter symmetry,
  sine-graded n x n quarter mesh, n = 12/24/48/96 (96 = 9216 elements per quarter; no-slip = 27648 unknowns).
  Observed order 1.96-2.02. Richardson error estimate 5.5e-5 to 8.7e-5. Taxel quantities come from the exact Fourier
  series of the piecewise-constant tractions on a 102.4 mm periodic cell (Ng = 2048 vs 1024), at the n = 96 solution.
  Accuracy:
  - base reactions: change n=48 -> 96 <= 6.4e-5 of F, Fourier resolution 1e-29;
  - taxel-mean displacements: about 1e-3 delta, limited by the edge singularity falling exactly on the taxel border
    (the centre-taxel mean u_z = 1 - 2...4e-4, where the exact value is 1);
  - the surface contact force map is exact by construction.

## 4. Results (results.json; summary.txt)
**C4 flat circular punch** (K in N/um; Fredholm, error <= 1e-13; bonded: ring BEM, error <= 1.3e-6)
| nu | a=5 frictionless | a=5 bonded | a=2.5 frictionless | a=2.5 bonded | Sneddon a=5 / 2.5 | Winkler k=M/t (comparison only) a=5 / 2.5 |
|---|---|---|---|---|---|---|
| 0.45 | 0.147570 | 0.161202 | 0.032764 | 0.034880 | 0.01254 / 0.00627 | 0.1986 / 0.0497 |
| 0.49 | 0.241385 | 0.339461 | 0.042579 | 0.050830 | 0.01316 / 0.00658 | 0.8961 / 0.2240 |
| 0.499 | 0.287981 | 0.488318 | 0.045992 | 0.057828 | 0.01332 / 0.00666 | 8.750 / 2.188 |

**C5 sphere, frictionless.** F(delta) and a(delta) are in results.json for delta = 1..30 um; error <= 5e-13.
Values below are at delta = 30 um.
| nu | R=10: F [N] | a [mm] | F/F_Hertz | R=25: F [N] | a [mm] | F/F_Hertz |
|---|---|---|---|---|---|---|
| 0.45 | 0.04397 | 0.6316 | 1.601 | 0.09623 | 1.0724 | 2.215 |
| 0.49 | 0.04822 | 0.6396 | 1.673 | 0.11015 | 1.0949 | 2.416 |
| 0.499 | 0.04938 | 0.6418 | 1.693 | 0.11422 | 1.1013 | 2.476 |

Hertz gives a = 0.5477 / 0.8660 mm. On the thin layer the contact radius is LARGER than the Hertz value at the same
depth: material pushed out sideways piles up around the contact.

**C6 square taxel punch** (K in N/um, Richardson-extrapolated; error <= 9e-5 relative)
| nu | frictionless | no-slip | half-space square (n=48) | Winkler M A_tax/t (comparison only) |
|---|---|---|---|---|
| 0.45 | 0.199710 | 0.216580 | 0.01470 | 0.2616 |
| 0.49 | 0.343268 | 0.481576 | 0.01542 | 1.1804 |
| 0.499 | 0.421055 | 0.732327 | 0.01561 | 11.526 |

Cross-talk ("if cells 1,2,3 deform +1,+2,+1, is contact defined per cell?"). Numbers are for the frictionless punch, then no-slip:
* Contact force is 100% in the centre taxel. The neighbours carry zero contact force, exactly, because the punch does
  not touch them. Contact is defined by where the surfaces touch, not per cell.
* The neighbouring surfaces do not go down. They rise: the layer is nearly incompressible and bonded at the back, so
  displaced material bulges up around the punch.
  - Mean u_z / delta of the edge neighbours: -0.047 / -0.035 (nu 0.45), -0.147 / -0.127 (0.49), -0.200 / -0.202 (0.499).
  - Diagonal neighbours: -0.011, -0.028, -0.038 (frictionless).
  - Peak rise just outside the punch edge (x = 5.5-6 mm): 0.29 delta (0.45), 0.74 (0.49), 0.99 (0.499); 1.23 delta for
    no-slip at 0.499 (fig_c6_profiles.png).
  A "+1, +2, +1" pattern is therefore not what an exact elastic layer does under a taxel-sized flat punch.
* The force transmitted to the rigid blade, binned in the same 5 x 7 cells at the bond plane, does spread:
  - centre 84.2 / 88.5 % (0.45), 84.9 / 91.8 % (0.49), 85.1 / 93.0 % (0.499);
  - each edge neighbour 3.4-3.6 % (frictionless), 1.6-2.7 % (no-slip); diagonals 0.1-0.3 %;
  - x-neighbour in-plane base reaction F_x/F = +-0.051 at nu = 0.45.
  The full 5 x 7 x 3 maps are in results.json, under `C6.*.taxel_maps`.

**Effect of nu.** It is large for confined contacts: the no-slip C6 stiffness goes 0.217 -> 0.482 -> 0.732 N/um for
nu = 0.45 -> 0.49 -> 0.499. It is small for the small sphere: R = 10 mm at 30 um changes by only +12%. The 0.45
placeholder therefore underestimates real PDMS (nu ~0.49-0.4999) pad stiffness under flat contacts by a factor of 2-3.4.
**Winkler (k = M/t, comparison line only):** +35% (C4 a=5, nu 0.45) up to 30x (nu 0.499) too stiff, and for the
sphere it is not even the right shape. It is not a usable reference.

## 5. Literature cross-check (what was actually opened)
* **Opened:** Argatov, Daniels, Mishuris, Ronken, Wirz, arXiv:1203.0918v1 (2012), https://arxiv.org/pdf/1203.0918.
  - Sec. 1.2 gives the bonded-layer kernel L(lambda) = (2kap sh 2lambda - 4lambda)/(2kap ch 2lambda + 1 + kap^2 + 4lambda^2).
    It is identical to my mu k C_zz/(1-nu). I quadrature-checked a0 = int(1-L) separately: a0 = 1.37685, 1.62547,
    1.77022 for nu = 0.3, 0.45, 0.5.
  - The asymptotic flat-punch model, their eq. (7), matches my exact kappa_c to 2e-5 at a/h = 0.05 and 0.1. It
    departs above a/h ~ 0.6, as they state (fig_literature.png).
  - Their **Table 1** (kappa_s = F/F_Hertz vs a/h for the sphere, "based on the results obtained by Hayes et al. (1972)";
    21 a/h values x nu = 0.30..0.50) vs my exact values: mine are systematically higher by +0.20 % to +0.56 %
    (max abs 0.063 at a/h = 3, nu = 0.5). My small-a/h behaviour satisfies the exact asymptotics: the first-order
    coefficient 2a0/pi, and the second-order coefficient converges to 14a0^2/(3pi^2) in varpi = sqrt(R delta)/h, which
    is their eq. (47) (`lit_asymptotic_check.out`). Hayes' tabulated values at a/h = 0.04 are below the exact
    first-order asymptote, so I attribute the 0.2-0.6 % gap to the accuracy of the 1972 numerics. This is an inference,
    not a check against the original.
  - **Their eq. (48)** (kappa_s in alpha) gives the alpha^2 coefficient as 2a0^2/pi^2. My numerics, and a re-expansion
    of their own eq. (47), give 10a0^2/(3pi^2) (0.8924 vs 0.5354 at nu = 0.45). This looks like a misprint in eq. (48).
* **Not opened.** None of the numbers below come from these sources:
  - Hayes, Keer, Herrmann & Mockros, J. Biomech. 5 (1972) 541-551: paywalled. Its kappa values were used only through
    Argatov's Table 1.
  - Dimitriadis et al., Biophys. J. 82 (2002) 2798-2810: the PMC page (PMC1302067) shows only the abstract, and the
    PDF is behind a JavaScript proof-of-work page that curl cannot pass. Not verified. The bonded-sample fit
    alpha0 = -(1.2876 - 1.4678 nu + 1.3442 nu^2)/(1-nu) is from memory only. It gives -alpha0 = 1.6351 at nu = 0.45
    vs my exact a0 = 1.6255 (0.6 %), consistent with a polynomial fit, but this is unverified.
  - Shull, Mater. Sci. Eng. R 36 (2002): not opened (paywalled); not used.

## 6. Cross-check with gt-fem (finite 51 x 71 mm pad, quadratic FEM; ../gt_fem/results.json, summary_table.md)
| case | nu | K FEM (+- its 2-level err) | K this work | diff | base-force map max diff (frac. of F) | taxel mean u max diff (/delta) |
|---|---|---|---|---|---|---|
| C6f | 0.45 | 0.200119 +- 0.00046 | 0.199710 | +0.20 % | 7.7e-4 | 4.5e-4 |
| C6f | 0.49 | 0.3445 +- 0.0017 | 0.343268 | +0.36 % | 9.6e-4 | 1.8e-3 |
| C6f | 0.499 | 0.421166 +- 0.0002 | 0.421055 | +0.03 % | 7.6e-5 | 3.6e-4 |
| C6 | 0.45 | 0.21704 +- 0.00054 | 0.216580 | +0.21 % | 6.0e-4 | 4.1e-4 |
| C6 | 0.49 | 0.484348 +- 0.0038 | 0.481576 | +0.58 % | 5.1e-4 | 2.0e-3 |
| C6 | 0.499 | 0.733412 +- 0.0012 | 0.732327 | +0.15 % | 5.4e-5 | 6.3e-4 |

All six FEM stiffnesses lie within their own stated error of the exact values. They sit above, as expected for a
displacement FEM. u_z(x, 0) outside the punch agrees to <= 0.005 delta. C4: gt-fem's axisymmetric FEM gives
0.14757 / 0.241388 / 0.287981 (frictionless, a = 5) and 0.161202 / 0.339465 / 0.488319 (bonded). These match the
table above to <= 2.7e-5 relative for all 12 entries (a = 5 and 2.5 mm), which is within gt-fem's stated errors.

Convergence of gt-fem towards these values, per run level (`compare_gt_fem_levels.out`; level 3 has 1.63 M dof;
dBase = max |base-force fraction difference| over the 5 x 7 x 3 map, dProf = max |u_z(x,0)/delta difference| for 6 < x < 25 mm):

| case | nu | form | L1 dK/K | L2 dK/K | L3 dK/K | L3 (or L2) dBase | L3 (or L2) dProf |
|---|---|---|---|---|---|---|---|
| C6f | 0.45 | full | +4.4e-3 | +2.1e-3 | +1.35e-3 | 5.2e-4 | 8.7e-4 |
| C6f | 0.45 | mixed | +7.2e-4 | +3.0e-4 | - | 1.3e-4 | 1.4e-3 |
| C6f | 0.49 | full | +8.5e-3 | +3.6e-3 | +2.3e-3 | 6.5e-4 | 2.3e-3 |
| C6f | 0.49 | mixed | +7.0e-4 | +2.6e-4 | - | 8.5e-5 | 2.8e-3 |
| C6f | 0.499 | mixed | +7.3e-4 | +2.6e-4 | +1.65e-4 | 5.7e-5 | 1.8e-3 |
| C6 | 0.45 | full | +4.6e-3 | +2.1e-3 | +1.39e-3 | 4.1e-4 | 6.8e-4 |
| C6 | 0.45 | mixed | +1.2e-3 | +5.4e-4 | - | 5.6e-5 | 1.2e-3 |
| C6 | 0.49 | full | +1.37e-2 | +5.8e-3 | +3.7e-3 | 3.5e-4 | 2.5e-3 |
| C6 | 0.49 | mixed | +2.3e-3 | +1.1e-3 | - | 4.9e-5 | 2.6e-3 |
| C6 | 0.499 | mixed | +3.2e-3 | +1.5e-3 | +9.8e-4 | 3.9e-5 | 2.3e-3 |

Every FEM sequence approaches the transform value monotonically from above. A displacement FEM should do this. The
free pad edges would push the FEM below the infinite-layer value, but none of the FEM values falls below it. The
smallest remaining gaps are +1.65e-4 (C6f, 0.499, L3) and +2.6e-4 (C6f, 0.49, mixed L2), so the finite-pad effect on K
is below about 2e-4 relative. That bound is inferred and was not computed directly.

## 7. Assumptions and limits
* The layer is infinite in x and y. The pad edges are >= 10 t from every load here, and the kernels decay
  exponentially (the total point response at 10 t is 2-5e-4 of the half-space value, V3). The FEM comparison (section 6) bounds
  the finite-pad effect on the C6 stiffness at about 2e-4 relative.
* Small strain, linear elasticity, rigid indenters, perfect bonding. "No-slip" means the punch face moves only in z.
* Not done here: C1-C3 for the finite pad (that is gt-fem's job; for the infinite layer C1 -> G1 = M A/t, C2 -> G A/t),
  and the no-slip sphere.
