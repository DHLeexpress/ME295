# MuJoCo 3.17.0 flex internals for a thin, stiff, bonded PDMS layer in contact

Source: MuJoCo 3.17.0 (git tag 3.17.0, commit 800a109), checked out read-only at
`<scratch>/src/mujoco` (`$S` below). Every `file:line` is in `$S`.
Checks: `scripts/*.py` in this directory, outputs in `out/` (`$D` = this directory). The tags mean:
**[run]** = number from a run I did (script -> output), **[code]** = read in the source, **[derived]** = algebra on
[run]/[code] facts, not run on its own, **[other agent]** = found by another flexstudy agent and checked here as stated.

Layer: 51 x 71 x 1.5 mm, E = 1 MPa, nu = 0.45 (also 0.49), back face pinned. Our mesh is flexcomp grid 11x15x3 vertices,
so the hexes are 5.10 x 5.07 x 0.75 mm (aspect ratio 6.8) and every hex is split into 6 tets.

## 0. Verdicts, ranked by how much they matter for us

1. **Bug (F1 confirmed): dof="full" StVK uses Lame lambda TWICE too large.** This holds for every nu > 0, every mesh,
   rotated or moving parents, and also at 1 % finite strain. The fit of lambda_eff/lambda gives 2.00000 with residual
   < 1e-11 (p0 [run]). Item 2 has the code derivation. Workaround (inputs giving lambda/2): nu 0.45 -> young 971787,
   poisson 0.409091; nu 0.49 -> young 993552, poisson 0.480392. With these the ratio is 1.00000 (p0 [run]). The SNH
   path (`elastic3d=1`), trilinear and quadratic are all correct.
2. **Robot divergence: the island metric leaves out the flex stiffness of a non-"simple" flex.** A flex is non-simple
   when it is pinned to a body that has dofs. The per-island operator `flexMatrixFree` (engine_metric.c:2485-2491) gates
   on `mjd_flexStiff_active`, and that returns 0 for every flex that is not `mj_flexSimple` (engine_derivative.c:1905-1907).
   The monolithic operator `mj_effMulAdd` (engine_metric.c:236-246) does include it. As a result, the constraint solve
   inside an island sees M without K. [other agent: flexstudy/robot found this; here source-verified, plus a minimal
   repro]. Repro, v1 [run]: one hinge carrying a pinned flex plus one joint-equality row gives BADQACC at step 3 with
   islands on (3x3x2 and 11x15x3). The same model is stable with `mjDSBL_ISLAND`, and stable with no constraint row.
   Workaround: `<flag island="disable"/>`.
3. **Contact is a soft penalty spring in series with the layer, and its stiffness scales with vertex mass.** Per vertex
   contact: k_c = K imp^2 m_v / (1-imp), K = 1/(d_width^2 tc^2 dr^2). This measures 105887 N/m at our settings. Over the
   whole free face that is 17.5 N/um, only about 2x the true layer stiffness (FEM 8.59 N/um). Under the discrete
   integrator k_c cannot go above about mjMAXIMP/(1-mjMAXIMP) m_v/h^2 = 4.235e5 N/m per vertex (69.6 N/um for the face)
   at any solref (v1 [run]). The C1 platen therefore reads K_n = 7.9-8.8 N/um with our settings. That is
   (doubled-lambda layer 15.9) in series with (contact about 17.5). It looks "right" only by coincidence of two errors
   (c2 [run]).
4. **Any non-plane geom collides with whole tets (one GJK contact per layer-0 tet), not with face vertices.**
   - The contact point is a GJK support point. For flat-on-flat contact it jumps between tet vertices from step to step:
     the median jump is 7.2 mm, i.e. a full element (c3/c3b [run]).
   - Its force is spread by inverse-distance weights over all 4 tet vertices, including the bonded back-layer vertex.
     The mean weight on non-surface vertices is 21 % for a platen (c1 [run]).
   - Result: a static box platen never settles (force CV 4.5 %, qvel about 7e-3 m/s), while a plane, which uses vertex
     contacts, settles (CV 0.4 %).
   - Taxel binning of these contacts (tactile.py) will flicker between neighbouring taxels.
5. **The tetrahedralisation is biased.** All hexes use the same body diagonal, corner (1,0,0) to (0,1,1) (6-tet Kuhn
   pattern, user_mesh.cc:2703-2738). On the centre taxel, a normal load gives a spurious tangential displacement of
   15.7 % of u_z at 11x15x3 and 6.4 % at 21x29x5. A mirrored ("J1") pattern gives 0 (t1 [run]).
6. **Mass and childclass.**
   - Flex mass is `mass/npnt` per vertex, and pinned vertices get none: our flex carries 2/3 of the specified mass
     (m1, v1 [run]). Contact stiffness (point 3) is proportional to this mass.
   - Vertex slide joints inherit the parent's childclass: armature, stiffness and frictionloss are copied; joint
     damping is a compile error (v1 [run], user_flexcomp.cc:2715). An inherited joint stiffness would tie every vertex
     to its rest position (an artificial elastic foundation). The robot model has all of these at 0 (m1 [run]).
7. **The effective-metric PCG needs more than opt.iterations = 100 on the robot (116 at the home keyframe)**, so the
   "ill-conditioned" warning fires every step (i1 [run]). It is not the cause of the divergence (see point 2).
8. **dim=2 membranes (elastic2d="stretch") come out 4x softer than a plane-stress membrane of the documented
   `thickness`, and use 2 lambda_3D rather than the plane-stress lambda** (p0 [run]). MuJoCo's own test
   `ElasticEnergyMembrane` encodes the 1/4. This does not affect our dim=3 layer.
9. **A strain equality on an interpolated flex replaces the elasticity.** It becomes soft constraint rows (eigenmodes of
   a K built with E = 10 and nu = 0.3), so `young` and `poisson` are ignored. The stiffness then comes from solref
   (p0 [run]: nu_eff = 0.300 for any input nu; E_eff = 8.3 Pa with solref 0.02, 3.3 kPa with 0.001).
10. **F5 confirmed**: the "not rigid and has no equality constraints or passive forces" warning ignores `young`
    (user_model.cc:3694-3702).

---------------------------------------------------------------------------------------------------------------------

## 1. Data model and compilation

**flexcomp types** (user_flexcomp.cc:169-198): grid -> `MakeGrid` (819); box -> `MakeBox` (861); mesh -> `MakeMesh`
(914); gmsh -> `MakeGMSH` = `MakeMesh` (1079-1082); direct takes points and elements as given. `pingrid` and
`pingridrange` are allowed only for grid or interpolated flexes (219-224); a direct, mesh or gmsh flex must pin by
`pin id` or `range`.

**dim / dof**: `dim` is 1 (rod), 2 (triangles) or 3 (tets). `dof` (user_flexcomp.cc:485-560):
- `full`: a new body per unpinned vertex with 3 slide joints along the parent axes (529-540).
- `radial`: 1 slide along the vertex direction from the centre; the centre is always pinned (378-379, 520-527).
- `2d`: x and y slides only (543-551).
- `trilinear` / `quadratic`: all collision vertices are attached to the parent body, and dofs live on separate node
  bodies placed on a (cellcount*order+1)^3 grid (580-706). The collision vertices follow the nodes through
  interpolation weights.

**Grid tetrahedralisation** (user_mesh.cc:2703-2738):
- `cube2tets = {0,3,1,7},{0,1,4,7},{1,3,2,7},{1,2,6,7},{1,5,4,7},{1,6,5,7}`.
- That is 6 tets per hex, all sharing the body diagonal from local corner 1 = (1,0,0) to corner 7 = (0,1,1), and the
  same pattern is used in every hex. m1 [run]: 280 hexes -> 1680 tets, one shared diagonal, all 280 free-face quads
  split along (1,-1,0).
- **Does it bias the response? Yes** (t1 [run], `scripts/t1_tet_bias.py`, `out/t1_tet_bias.txt`). I took MuJoCo's own
  assembled K (efm_K/h^2, lambda remapped so the material is correct) and loaded the centre taxel:

  | mesh | pattern | patch u_z per N (um/N) | normal->shear (u_x/u_z, u_y/u_z) | mirror asym. of u_z | tangential compliance (1,1) / (1,-1) |
  |---|---|---|---|---|---|
  | 11x15x3 | MuJoCo (Kuhn) | 2.368 | -0.157, +0.158 | 2.3 % | 0.972 |
  | 11x15x3 | mirrored J1 | 2.776 | 0 (1e-16) | 0 | 1.000 |
  | 21x29x5 | Kuhn | 3.480 | -0.064, +0.064 | 5.6 % | 0.988 |
  | 21x29x5 | J1 | 3.912 | 0 | 0 | 1.000 |

  - The Kuhn mesh is 15 % (11x15x3) and 11 % (21x29x5) stiffer under a local normal load, and it produces shear
    cross-talk that a symmetric layer cannot have.
  - The patch compliance is still far from converged at 11x15x3 (2.37 -> 3.48 um/N).
  - Workaround: build the mesh yourself with `type="direct"` and a mirrored pattern (`t1_tet_bias.py:mesh()` does this),
    pinning the back face by `pin id`.

**cellcount (interpolated)**: `cellcount` x `order` defines the node grid. The collision surface is the mesh vertices,
which do not carry dofs. Contact forces reach the nodes through `mj_vertBodyWeight` (engine_core_util.c:660, called from
`mj_contactBodyWeight` 822-824). Their contacts can be reduced to 50 per geom (item 3).

**Pins** (user_flexcomp.cc:286-379, 490-502): a pinned vertex creates no body; its `vertbody` is the flexcomp's parent
body, so it is rigidly welded to that body at its local position (m1 [run]: pinned local positions are in the parent
frame, e.g. (-25.5, -35.5, 0.75) mm). If the parent has dofs, `mj_flexSimple` (engine_core_util.c:1277-1283) returns 0.
That switches the flex to the matrix-free stiffness path and exposes it to the island bug (item 4).

**Mass** (user_flexcomp.cc:476-477, 514): `bodymass = mass / npnt`, with npnt counting pinned points. Every free vertex
gets the same mass whether it is a corner, face or interior vertex, and pinned vertices give their share to nobody.
- m1/v1 [run]: mass = 0.0052414 kg (= rho V) and 495 points give 1.0589e-5 kg per vertex body. The total is 0.0034943 kg
  = 2/3 of the attribute.
- Compared with consistent lumping, per-vertex masses are off by factors of 0.57 to 6.8 (m1).
- Interpolated flexes use Simpson-type weights, normalised to `mass` (626-706).
- Consequences: dynamics have 2/3 of the physical mass; contact stiffness is proportional to the vertex mass (item 3).
  To get the physical total, set `mass = rho V npnt/nfree`.

**Inheritance** (user_objects.cc:2455-2460 `AddBody`: child classname = parent's; 2515-2517 `AddJoint`:
`def_map[classname]`):
- The vertex bodies and slide joints take the parent's childclass. v1 [run]: armature 0.01, stiffness 5 and
  frictionloss 0.001 were copied to all 990 vertex dofs.
- Joint damping is rejected: "flex vertex/node body 'f_0' cannot have joint damping" (user_flexcomp.cc:2715).
- In the robot (m1 [run]) all of them are 0. Keep it that way: an inherited joint `stiffness` would be a world-anchored
  spring on every vertex.

## 2. Elasticity

### 2a. dof="full", dim=3: StVK, and the derivation of F1
- **Compile time.** `ComputeStiffness` (user_flexcomp.cc:1313-1342) sets mu_c = mu V/4 * thickness and
  la_c = lambda V/4 * thickness, with the default `thickness = 4` for 3D, so mu_c = mu V and la_c = lambda V.
- `ComputeBasis<Stencil3D>` (1278-1310) gives, for each edge, B_e = (n_L n_R^T + n_R n_L^T)/(72 V^2) from the area
  normals of the two faces that do not contain the edge. Since n = -6V grad N, this is
  B_e = (g_a g_b^T + g_b g_a^T)/2: the dual basis, which gives E = (1/2) sum_e s_e B_e for s_e = l_e^2 - L_e^2.
- `MetricTensor` (1179-1220) forms **k_ef = mu_c tr(B_e B_f) + la_c tr(B_e) tr(B_f)**.
- **Run time.** `mj_flexPassiveStretch` (engine_passive.c:587-640) builds s with `mj_stretchElongation`
  (engine_core_util.h:302-309), tension t = k s (`mj_stretchTension`, 312-320), and forces -t_e * (edge vector)
  (`mj_stretchForce`, 333-345).
- Header comment (engine_core_util.h:323): "quadratic energy s' K s / 4".
- Put together: W = s'ks/4 = V [mu tr((sum s_e B_e/2)^2) + lambda (tr sum s_e B_e/2)^2] = **V [mu tr E^2 + lambda (tr E)^2]**.
  StVK is V [mu tr E^2 + (lambda/2)(tr E)^2], so the code's lambda is 2x too large.
- The SNH code in the same file (`ComputeSNH`, 1345-1372) uses `MetricTensor(mu V, -mu V)` plus
  k[22] = V (lambda + 2 mu)/2 times (J-1)^2. Its small-strain limit is V [mu tr E^2 + (lambda/2)(tr E)^2], i.e.
  correct, which strengthens the case that the StVK factor is a slip.
- Verified (p0 [run], `scripts/p0_patch_ext.py`, `out/p0_patch_ext.txt`): exact homogeneous fields (random, uniaxial,
  shear, general) on a 5^3 cube and on the 11x15x3 layer. qfrc_spring was fitted against exact SVK nodal forces of the
  same tets, f = mu f_mu + lambda f_lam:

  | case | lambda_eff/lambda | mu_eff/mu | uniaxial M_eff/M |
  |---|---|---|---|
  | nu = 0 | 0 | 1.00000 | 1.0000 |
  | nu = 0.3 / 0.45 / 0.49 | 2.00000 | 1.00000 | 1.4286 / 1.8182 / 1.9608 |
  | layer 11x15x3, nu 0.45 | 2.00000 | 1.00000 | 1.8182 |
  | rotated parent; rotated + freejoint parent | 2.00000 | 1.00000 | 1.8182 |
  | 1 % finite strain (uniaxial, shear, general) | 2.00000 (residual <= 1e-13 vs exact nonlinear SVK) | 1.00000 | |
  | remapped inputs, nu 0.45 / 0.49 | 1.00000 | 1.00000 | 1.0000 |

  So the defect is exactly lambda -> 2 lambda in the full nonlinear StVK, not a small-strain artefact or a frame issue.
- `bug_repro.py` (17 lines, one free tet, uniaxial strain) gives ratio 1.8182 (x) and 2.0000 (y), i.e.
  (2 lambda + 2 mu)/(lambda + 2 mu) (`out/bug_repro.txt` [run]).
- **Is it a bug? Yes.** The documented inputs are Young's modulus and Poisson's ratio. The material the code actually
  simulates is (lambda' = 2 lambda, mu), i.e. nu_eff = lambda'/(2(lambda' + mu)) and
  E_eff = mu(3 lambda' + 2 mu)/(lambda' + mu) [derived]. Values:
  - nu 0.45: nu_eff = 0.4737, E_eff/E = 1.016, bulk modulus x1.9310 (this equals the main session's hydrostatic patch
    ratio), uniaxial-strain modulus x1.8182.
  - nu 0.49: nu_eff = 0.4949, E_eff/E = 1.003, bulk x1.987, M x1.9608.

  So E is almost unchanged, while every compressibility-controlled quantity (confined layer, indentation of a bonded
  thin layer) is up to 2x too stiff. MuJoCo's `ElasticEnergySolid` (test/engine/engine_passive_test.cc:578) uses
  poisson = 0, where the bug is invisible.
- The minimal fix is `la /= 2` in `ComputeStiffness`, **not** in `MetricTensor` (SNH reuses it with la = -mu). This is
  numerically identical to the verified input remap, because the remap feeds exactly lambda/2 with the same mu
  [derived]. I did not build a patched MuJoCo. Draft report: `bug_lambda.md`.
- Rayleigh damping, if used, reuses the same k with kD = damping/timestep (engine_passive.c:599). It is timestep
  dependent, and its lambda part is doubled too. We use damping 0.

### 2b. Stable Neo-Hookean (mjsFlex.elastic3d = 1; spec only, not in MJCF: test/user/user_flex_test.cc:156 Elastic3DNotInMJCF)
- `ComputeSNH` (user_flexcomp.cc:1345-1372) and `mj_snhCubic`/`mj_snhVolume` (engine_passive.c:634-636) use
  P = mu(F - J F^-T) + (lambda + mu)(J - 1) J F^-T.
- p0 [run], nu 0.3 and 0.45: lambda_eff/lambda = 1.00000 and mu_eff/mu = 1.00000 at small strain. At 1 % general strain
  the residual against that P is 3e-14. Pure shear cannot identify lambda (tr E = 0), hence the "0" in that row.
- **Its small-strain limit is correct linear elasticity.** It requires the discrete integrator
  (user_flex_test.cc:227) and is "experimental" (changelog 3.16, item 5). Damping is skipped in the SNH branch
  (engine_passive.c:647-649).

### 2c. Interpolated linear elasticity (trilinear/quadratic)
- `ComputeLinearStiffness` (user_flexcomp.cc:1683-1768) computes
  K = -int [lambda div u div v + 2 mu sym grad u : sym grad v] with full Gauss-Legendre quadrature, (order+1)^3
  points (1692).
- p0 [run]: 1.00000 / 1.00000 for 1 and 2 cells, trilinear and quadratic, and for a rotated parent. It is
  co-rotational (flexelem_krot).
- With 2x2x2 full integration, a trilinear hex across a 1.5 mm layer locks in bending, and also volumetrically as nu
  approaches 0.5 (standard FE behaviour; not run here).
- **Strain equality override** (user_flexcomp.cc:3047-3049): if any `mjEQ_FLEXSTRAIN` references the flex, K is built
  with E = 1e1 (the comment says "young=1") and nu = 0.3. It is then eigendecomposed into sqrt(lambda_i) v_i rows
  (1974-2000, 3168-3190), which become soft constraint rows (engine_core_constraint.c:706-790). Passive elasticity is
  off (p0 [run]: qfrc_spring = 0).
  - p0 [run], 2x2x2 trilinear: E_eff = 8.31 Pa with default solref (0.02 1), 33 Pa with 0.01, 3324 Pa with 0.001.
  - nu_eff = 0.3000 whatever poisson we set.
  - Unusable as a material model for us.

### 2d. dim=2 (membrane, elastic2d="stretch"/"both", thickness > 0)
- `ComputeStiffness<Stencil2D>` gets `thickness` (user_flexcomp.cc:2848-2855), so mu_c = mu A t/4 and
  la_c = lambda_3D A t/4. The basis normalisation equals 3D's (B_e = (g g^T + g g^T)/2; algebra in this note).
- p0 [run], fitted against exact plane SVK membranes of thickness t: mu_eff/mu = 0.25000 for nu 0, 0.3, 0.45 and for
  t = 1 and 2 mm. lambda_eff = 0.5 lambda_3D, which is 0.875 and 2.75 times the plane-stress lambda for nu 0.3 and 0.45.
- So a MuJoCo membrane is 4x softer in shear than the documented "shell thickness" implies, and its in-plane Poisson
  coupling uses 2 lambda_3D instead of 2 mu lambda/(lambda + 2 mu).
- `ElasticEnergyMembrane` (engine_passive_test.cc:528-574) asserts `4*energy/volume == 2*scale^2`, i.e. it encodes the
  1/4, while `ElasticEnergySolid` asserts `energy/volume == 3*scale^2` with no such factor. Not relevant to our dim=3
  layer; worth a second upstream issue.

### 2e. Edge stiffness/damping and edge equalities
- `edgestiffness`/`edgedamping` are per-edge springs (engine_passive.c:800-830): a mass-spring network, not a continuum.
- `<edge equality>` (mjEQ_FLEX) and `young` cannot be combined (compile error "flex constraints and elasticity (young)
  cannot both be present", user_flex_test.cc:1906).
- Neither gives linear elasticity with prescribed (E, nu).

## 3. Contact

**Which primitives collide** (engine_collision_driver.c:825-860):
- For each geom/flex pair passing contype/conaffinity: if the geom is a **plane** -> `pushGeomFlex` ->
  `mjc_PlaneFlex` (engine_collision_flex.c:216-249). Every flex vertex is a sphere of `flex radius`, with
  dist = n.(v - p) - radius and the contact at the midpoint.
- **Any other geom** -> one `mjc_GeomElem` pair per *active* element (`mj_isElemActive`: `flex_elemlayer <
  activelayers`, driver.c:414-420).
- For dim=3 that is `mjc_ConvexElem` (collision_flex.c:388): GJK/EPA between the geom and the tet, inflated by radius.
  It yields at most 1 contact per tet (`mj_maxContact`-type limit, driver.c:1903-1914). Elements sharing the geom's
  body are skipped (318-325).
- No `vertcollide` attribute exists in 3.17 (grep: absent from XMLreference, mjmodel.h and the engine).
- c1 [run] (11x15x3, platen pressed in 10 um): box -> 350 contacts on 700 layer-0 tets; sphere R10 -> 6; capsule R5 ->
  22; cylinder a = 5 mm -> 8 (9 surface vertices touched); plane -> 165 vertex contacts. activelayers = 3 -> 840.
- [derived] For C5: at 30 um depth, a = sqrt(2 R delta) = 0.77 mm (R = 10 mm), 1.2 mm (R = 25 mm). Both are well below
  the 5 mm element, so a sphere sees 1-2 elements at 11x15x3.

**Contact position and frame**:
- Element contacts take the GJK witness point. For flat-on-flat this is a vertex of the tet and jumps between equally
  deep vertices: c3b [run] shows one tet's contact moving between corners 5 mm apart on consecutive steps, weights going
  (0.04, 0.69, 0.03, 0.24) -> (0.95, ...) -> (0, 1.0, 0, 0).
- c3 [run], static platen at 5 um:

  | tool | force CV | median contact-point jump | median qvel at the end |
  |---|---|---|---|
  | box | 4.5 % | 7196 um | 7.4e-3 m/s (never at rest) |
  | plane | 0.43 % | 0.0018 um | 3.6e-6 m/s |
  | sphere R25 | 0.09 % | 0.0002 um | |

**Force distribution** (engine_core_util.c:785-830 `mj_contactBodyWeight`; 618-657 `mj_elemVertWeight`):
- A vertex contact goes to one vertex body.
- An element contact is spread over the tet's 4 vertices with weights proportional to 1/|x_contact - x_v|, normalised.
- c1 [run]: the replica matches efc_J to 3e-16. The mean weight on non-free-face vertices is 0.21 (platen) and 0.23
  (cylinder punch); up to 0.45 lands on bonded back-layer vertices.
- Interpolated flex: the vertex weights are mapped again onto nodes (`mj_vertBodyWeight`, 660).
- diagA uses the same weights (engine_core_constraint.c:1803-1820).

**mjMAXCONPAIR = 50 reduction**:
- Applies only if `flexReducible` = rigid or interpolated (driver.c:572-575). This is changelog **3.16.0** item 10 (commit
  871e3725b), not 3.17 as F3 says.
- Pairs of one geom form a group (driver.c:835). `filterPreContacts` (2110-2175) keeps the deepest contact, then by
  greedy farthest-point sampling up to 50.
- c1 [run], trilinear 1-cell: box platen -> exactly 50 contacts.
- Full flexes keep every contact.

**solref/solimp and R**:
- Our flex has priority 2, so its solref/solimp/friction are used without mixing.
- v1 [run]: the same test *without* priority mixed the geom defaults and gave 1.06e3 N/m instead of 1.06e5 N/m per
  contact.
- `mj_makeImpedance` (engine_core_constraint.c:2036-2122):
  - R = (1-imp)/imp * diagA, with diagA = body_invweight0 of the vertex body = 1/m_v = 94440 kg^-1 (m1 [run]).
  - K = 1/(d_width^2 tc^2 dr^2), B = 2/(d_width tc).
  - Discrete integrator: R <- max(R/f, Rmin) with f = 1 + hB + h^2 K imp and Rmin = (1-mjMAXIMP)/mjMAXIMP diagA
    (2119-2121).
- solimp "0.99 0.99 0.001": solimp[0] == solimp[1] means the flat branch, imp = 0.99 at all depths (1988-1992; the
  width does not matter).
- Measured static spring per vertex contact (v1 [run], plane, 11x15x3):

  | tc (s) | k_c measured (N/m) | K imp^2 m_v/(1-imp) | face total (N/um) |
  |---|---|---|---|
  | 1e-3 | 1058.87 | 1058.87 | 0.17 |
  | **1e-4 (ours)** | **105887** | 105887 | **17.47** |
  | 3e-5 | 376930 | 1.18e6 | 62.2 |
  | 1e-5 | 407061 | 1.06e7 | 67.2 |
  | 1e-6 | 421816 | 1.06e9 | 69.6 |

  - The ceiling is mjMAXIMP/(1-mjMAXIMP) m_v/h^2 = 4.235e5 N/m.
  - This is why **contacts on tiny vertex masses are soft**: the regulariser scales with 1/m_v, and the stiffness is
    capped at about 1e4 m_v/h^2 whatever solref is.
  - [derived] Uniform per-vertex mass means the face carries mass/nz. Refining in z (or in-plane at fixed mass) does not
    raise the total contact stiffness, and refining in z lowers it.
  - (A direct check of K imp/(f R) from efc_R differed by a constant factor 4 from the measurement. I do not rely on it;
    the formula in the table matches the measurement to 1e-6.)
- C1 rig consequence (c2 [run], box platen pressed 10 um, our settings):

  | case | K_n (N/um) | notes |
  |---|---|---|
  | box, no-slip (mu 1) | 8.757 (8.609 with 1000 iterations) | not converged at 100 iterations |
  | plane, no-slip | 7.933 | |
  | box, frictionless | 8.026 | |
  | plane, frictionless | 7.435 | |
  | flex alone, vertices prescribed (doubled lambda) | 15.869 (no-slip), 14.204 (frictionless) | |
  | bounds M A/t | 9.157 (lambda), 16.648 (2 lambda) | |

  - Free-face mean u_z/delta is 0.46-0.57, so the rest of delta is penetration.
  - Springs in series 1/(1/15.87 + 1/17.47) = 8.3 [derived], close to the measured 7.9-8.8.
  - refsafe on -> 1.32 (box) / 0.65 (plane). solref 0.002 -> 0.092. solimp 0.9/0.95 -> 2.69.

**refsafe under the discrete integrator**:
- The classical clamp tc >= 2h is skipped when `mj_isMetric` (= integrator discrete, engine_core_util.c:1615-1617;
  engine_core_constraint.c:1903-1908).
- Instead, if refsafe is enabled, contact and limit rows with h^2 K imp > 1 get K = 1/(h^2 imp) and B reduced
  (2101-2117; doc XMLreference.rst:640-645).
- c2 [run]: refsafe on gives K = 4.04e6 and an 8.5x softer contact. Our "refsafe disabled" is therefore right and
  needed.

**Passive contacts** (`<contact passive="true">`, doc XMLreference.rst:4902-4923; metric rank-1 class
`effContactBuild` engine_metric.c:2894):
- A frictionless penalty with stiffness "chosen as a natural frequency scaled by vertex mass, not user-specified".
- It covers only flex-flex, self and static-geometry contact; contact with a movable body stays in the solver.
- Not useful for a grasp, which needs friction against a moving object.

**ipc flag**: discrete-integrator incremental-potential contact, **dim-2 flexes only**, frictionless (changelog 3.14.0
item 1; XMLreference.rst:781-790). Not applicable to a dim-3 layer.

## 4. Time integration (discrete integrator)

- `mj_discrete` -> `mj_advance` (engine_forward.c:1584-1588). The step solves for velocity in the effective metric
  M_hat = M + h D + h^2 K (+ contact rank-1 terms).
- The flex stiffness enters as s1 = h^2 and the damping as s2 = h (engine_metric.c:3083-3085). The stiffness of a
  "standard" (simple) flex is assembled into CSR (`mjd_flexStiff_assemble`, 3098-3112). Only tensile edge tension enters
  the geometric part (engine_core_util.h:356-363).
- The constant explicit shift is in `mj_effShift` (2247-2275).
- **Solve.** `mj_effSolve` (engine_metric.c:1638-1714) is PCG on M_hat.
  - Preconditioner `effBlockApply`: 3x3 vertex blocks of M+K for CSR-assembled flexes (`effBlocks`, 1474), with the qH
    backbone elsewhere.
  - Budget `opt.iterations`; tolerance `opt.tolerance` on the relative residual (1658-1665).
  - **Curvature breakdown** pAp <= 0 exits silently (1679-1681), without the warning.
  - When the budget is exhausted the "ill-conditioned ... Reported as a singular inertia" warning is raised as
    mjWARN_INERTIA (1693-1711).
- **Articulated attachments** (commit eed008905, changelog **3.15.0** item 7): bending/stretching include the motion and
  reaction of articulated vertex attachments. Such flexes are applied matrix-free (`mj_effMulAdd`, engine_metric.c:236-246)
  with no 3x3 blocks, and Newton refuses them ("flex with general attachments requires solver='CG'",
  engine_forward.c:1753). PGS and noslip are refused too (1769).
- **Dense fold**: `mj_effPrecFold` (2001) folds the rank-1 classes and the quadratic-zone efc rows into a copy of the
  3x3 blocks (and dense blocks for uncovered components, `effDenseBlock` 1914). Used by the constraint solver.
- **Islands (the robot bug)**: per-island operators `mj_effMulAddIsland`/`mj_effPrecIsland` (2593, 2788) take the
  flexes from `effIslandBuild` (2497), which calls `flexMatrixFree` (2485-2491). That returns
  `mjd_flexStiff_active(...) && (...)`, and `mjd_flexStiff_active` is 0 for non-simple flexes (engine_derivative.c:1905).
  So a pinned flex on an articulated body has **no stiffness in the island metric** whenever islands exist, i.e. when
  there are constraint rows. v1 [run]:

  | flex | constraint | islands | result |
  |---|---|---|---|
  | 3x3x2 (nv 28) | joint equality | on | BADQACC at step 3, qvel 35 m/s |
  | 3x3x2 | joint equality | off (mjDSBL_ISLAND) | stable, qvel 1e-5 |
  | 11x15x3 (nv 991) | joint equality | on | BADQACC at step 3 |
  | 11x15x3 | joint equality | off | stable |
  | both | none | on | stable |

  (My "joint limit" variant never activated, nefc = 0, so it tests nothing.) [other agent] flexstudy/robot has an
  instrumented build with a one-line fix and robot runs.
- **PCG cost** (i1 [run], `out/i1_metric_pcg.txt`): iterations to tol 1e-8, and ms per mj_forward once converged:

  | mesh | world-fixed parent (CSR + 3x3 blocks) | hinge parent (matrix-free, no blocks) |
  |---|---|---|
  | 6x8x2 | 14 it, 0.23 ms | 49 it, 0.37 ms |
  | 11x15x2 | 23 it, 0.95 ms | 69 it, 1.7 ms |
  | 11x15x3 | 36 it, 2.3 ms | 133 it, 8.1 ms |
  | 21x29x3 | 49 it, 6.8 ms | 147 it, 22 ms |
  | 11x15x5 | 67 it, 4.4 ms | 262 it, 19 ms |
  | 21x29x5 | 83 it, 18 ms | 267 it, 74 ms |
  | 41x57x5 | 107 it, 91 ms | 269 it, 365 ms |
  | robot 11x15x3, home keyframe | | 116 it |

  - Iterations grow with the number of through-thickness layers (aspect ratio), not just with nv.
  - The robot's iterations = 100 is below the 116 needed: the warning comes from that, the divergence does not.
- **Arena**: robot narena = 64 MB (m1 [run]). Overflow happens only after divergence (20k+ contacts, main-session logs).
  The default arena for flex models is now sized from topology (changelog 3.17, commit 39dd10f2d).
- **What controls stability and cost**:
  - (i) The island bug: fatal.
  - (ii) PCG iterations vs layers in z: cost.
  - (iii) Contact rows: ~2 per surface quad for element contacts, 3 efc rows each with condim 3.
  - (iv) h: the contact ceiling scales with 1/h^2; cost scales with 1/h.
  - The elastic part is implicit and stable at any h for a linear material.

## 5. Blind-spot checklist for `asm/11x15x3/pdms_left.xml` and the robot options

| setting | what the code does | right? |
|---|---|---|
| young 1e6, poisson 0.45 | lambda doubled (item 2a) | **No**: use 971787 / 0.409091 (or 993552 / 0.480392 for nu 0.49) |
| grid 11x15x3 | Kuhn 6-tet pattern, 5.1x5.07x0.75 mm tets | biased (t1); patch compliance not converged |
| pin gridrange back face | 165 vertices welded to `left_pdms`; flex non-simple on the robot | correct bond, but triggers the island bug and the matrix-free path |
| mass 0.0052414 | 1.0589e-5 per free vertex; total 2/3 of attribute | statics fine; contact stiffness and dynamics use 2/3 mass |
| radius 0 | element contacts against the tet itself; plane contacts at the vertex | geometrically right; contact = penetration |
| margin/gap 0 | no anticipation | fine for quasi-static |
| solref 0.0001 1 + refsafe off | K = 1.02e8, f = 36.4; static k_c = 1.06e5 N/m per vertex; face 17.5 N/um | contact compliance ~ layer compliance; cannot exceed ~70 N/um at h = 0.5 ms |
| solimp 0.99 0.99 0.001 | flat imp 0.99 (width ignored) | ok; mjMAXIMP 0.9999 is the ceiling |
| priority 2 | flex parameters win | right (without it: 100x softer) |
| condim 3, friction 1, elliptic, impratio 100 | friction rows R/100 | ok, but friction acts at the jumping GJK point for element contacts |
| selfcollide none, activelayers 1 | layer 0 = 700 of 1680 tets (every surface tet, incl. back and sides) | ok |
| contype/conaffinity 2 | geom-flex filtering at flex level (driver.c:827) | ok |
| integrator discrete, solver CG | required for articulated attachments | ok; Newton/PGS/noslip refused |
| iterations 100, tolerance 1e-8 | also the mj_effSolve budget; robot needs 116 | raise to >= 300 or tolerance 1e-6 |
| timestep 0.5 ms | contact ceiling 1e4 m_v/h^2 | smaller h stiffens contact |
| memory 64 MB | enough for 2 x 11x15x3 when stable | ok |
| island flag (default on) | **flex stiffness missing from the island metric** | **disable** (`<flag island="disable"/>`) until fixed |
| childclass of `left_pdms` | inherited armature/stiffness/frictionloss go to the vertex joints | currently all 0: ok, fragile |

Missed items, ranked by impact:
1. The island-metric omission (fatal).
2. lambda x2.
3. Contact compliance in series, which depends on vertex mass and on h.
4. Element contacts on platens/punches chatter and put ~20 % of the load on bonded vertices. Use vertex-type contact
   (plane geoms, or a thin dim-2 skin flex) when testing.
5. Tet-pattern shear cross-talk (16 % at 11x15x3).
6. Flex mass is 2/3 of physical.
7. PCG budget warning.
8. The F5 false warning hides real ones.

## 6. MuJoCo tests and example models that exercise these features vs our regime

Script `scripts/t2_models_tests.py` -> `out/t2_models_tests.{txt,json}` [run]; parameters below were read from the
compiled models.

**Example models** (`model/flex/`):
- dim=3 full flexes with elasticity:
  - floppy.xml: E 5e4, nu 0.2, edge 141 mm, Euler, h 1 ms.
  - jelly.xml: E 5e4, edge 99 mm, Euler.
  - press.xml: E 1e4/5e4, nu 0/0.4, edge 200 mm, discrete, Newton.
- gripper.xml and sphere_full.xml: no elasticity (edge 17 mm and 64 mm).
- Interpolated:
  - trilinear.xml, quadratic.xml: E 1e4, nu 0.1.
  - gripper_trilinear.xml: E 1e4, nu 0.1, discrete.
  - sphere_trilinear.xml: E 1e3/1e4, nu 0.4/0.49, direct solref -5000 -10.
  - mechanisms/cantilever.xml: quadratic, E 8e6/1.6e7, nu 0.25, discrete.
  - strain.xml: strain equality.
- Membranes: trampoline.xml (stretch, E 3e5, nu 0); bag.xml and drape.xml (both, nu 0.2).
- Contact settings in the examples: solref 0.003-0.02, solimp (0.95-0.99, 0.99-0.999, 1e-4..1e-3).

**Tests** (`test/`):
- `ElasticityTest.ElasticEnergySolid` (engine_passive_test.cc:578, poisson 0).
- `ElasticityTest.ElasticEnergyMembrane` (528, poisson 0; encodes the 1/4).
- `ElasticityTest.ElasticEnergyShell` (421).
- `UserFlexTest.StiffnessMatrix` (user_flex_test.cc:690, young 1, interpolated, rest only).
- SNH tests: `UserFlexTest.SNHSpecOnly` / `SNHRequiresDiscreteIntegrator`, `DerivativeTest.SNHStiffnessThroughInversion`
  / `SNHProjectionSpectralReference` (engine_derivative_test.cc:2934, 3053).
- Stretch derivative and assembly: `DerivativeTest.FlexStretchDerivatives(Tensile)` / `FlexStiffAssemble`
  (3179-3372).
- Strain constraints: `CoreConstraintTest.StrainConstraint*` (engine_core_constraint_test.cc:863-1076).
- Activelayers: `MjCollisionTest.Flex3DActiveLayersMidphaseDisabled` (young 300, nu 0.2, discrete).
- Islands: `IslandTest.FlexStiffnessUnionsTrees` (engine_island_test.cc:907).
- 298 tests mention flex.

**Gap.** No test or example checks a nonzero-Poisson full StVK against an exact solution. None uses mm-size elements
with E ~ 1 MPa, nu >= 0.45, contact with solref ~1e-4, or a pinned flex on an articulated body together with a
constraint row (which is exactly the island bug). Our element edges (0.75-5.1 mm) are 3-200x smaller than any example,
and our contact time constant is 30-200x shorter.

## 7. Corrections to the "established" findings

- F1: confirmed and extended (finite strain, moving and rotated parents, layer mesh). The membrane and SNH paths
  behave differently (2b, 2d).
- F2: K_n values not re-run. The 9.157 bound is correct, and the 16.648 bound for doubled lambda is reproduced (c2).
  F2's platen numbers come from a rigid-weld rig. With *contacts* (our settings) the same platen gives 7.4-8.8 N/um,
  because the contact is in series (item 3).
- F3: the 50-contact reduction is changelog **3.16.0** item 10 (commit 871e3725b), not 3.17. Selection: deepest first,
  then farthest point (driver.c:2110-2175). "Very soft with default refsafe" is explained by the discrete-integrator
  refsafe rule (2101-2117).
- F4: PCG and the pAp <= 0 silent exit are confirmed (engine_metric.c:1679-1681). The divergence cause is the
  island-metric omission (verdict 2), not PCG convergence. In i1 and the robot agent's trace, PCG reaches 1e-8 and never
  breaks down.
- F5: confirmed (user_model.cc:3694-3702 checks rigid, edgeequality, edgestiffness, edgedamping, damping and bending,
  but not young).
- eed008905 is in 3.15.0 (changelog line 513), not 3.17.
