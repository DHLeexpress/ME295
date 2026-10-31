# flexstudy/internals: how MuJoCo 3.17 flex works, for a thin bonded PDMS layer

What: a code-referenced reading of MuJoCo 3.17.0 flex (compiler, elasticity, contact, discrete integrator), with every
claim checked by a small numerical experiment. Main document: `NOTES.md`. Draft upstream bug report: `bug_lambda.md`
(not posted). Progress log: `PROGRESS.md`.

## Rerun
Python: /home/dohyun/miniforge3/bin/python (mujoco 3.17.0, numpy, scipy). No extra packages. From this directory:
```
P=/home/dohyun/miniforge3/bin/python
$P scripts/p0_patch_ext.py        # item 2: patch tests (full StVK, remap, SNH via mjSpec, trilinear/quadratic, strain eq, membranes)
$P scripts/m1_compile_inspect.py  # item 1: compiled layer (tets/hex, diagonals, pins, mass, inheritance, robot options)
$P scripts/t1_tet_bias.py         # item 1: Kuhn vs mirrored tetrahedralisation on the centre taxel
$P scripts/c1_contact_census.py   # item 3: which primitives collide, contact weights vs efc_J
$P scripts/c2_platen_static.py    # item 3: C1 platen K_n with contacts vs flex-only; solref/refsafe/solimp sweeps
$P scripts/c2b_box_diag.py        # item 3: per-contact static relation, box vs plane
$P scripts/c3_chatter.py; $P scripts/c3b_jump_detail.py   # item 3: element-contact chatter
$P scripts/i1_metric_pcg.py       # item 4: PCG iterations of mj_effSolve, fixed vs hinge parent, mesh sweep, robot
$P scripts/v1_misc.py             # items 1/4/5: childclass inheritance, pinned mass, island-bug repro, contact ceiling
$P scripts/t2_models_tests.py     # item 6: example models / tests inventory
$P bug_repro.py                   # 17-line repro for bug_lambda.md
```
Outputs go to `out/<script>.{txt,json}`. Each script takes under a few minutes on one core. i1 uses a pool of 12 processes;
the robot case reads <scratch>/asm/11x15x3/scene.xml (read-only).

## Results (details and file:line references in NOTES.md)
- dof="full" StVK uses 2 lambda exactly: lambda_eff/lambda = 2.00000, also at 1 % strain and with rotated or moving
  parents. Remap young/poisson = 971787/0.409091 (nu 0.45) or 993552/0.480392 (nu 0.49) gives 1.00000. SNH, trilinear
  and quadratic are correct. Membranes are 1/4 as stiff, with 2 lambda_3D.
- Robot divergence = island metric leaves out the stiffness of a pinned flex on an articulated body
  (engine_metric.c:2485 + engine_derivative.c:1905). Minimal repro diverges at step 3 with islands, stable without.
- Contact: per-vertex penalty k = K imp^2 m_v/(1-imp) = 1.06e5 N/m at our settings (17.5 N/um for the face, about 2x
  the layer), capped at about 1e4 m_v/h^2 (70 N/um). The platen K_n of about 8 N/um is (2-lambda layer 15.9) in series
  with the contact.
- Non-plane geoms collide with whole tets: GJK support points jump (7 mm), the force goes ~21 % to bonded vertices,
  and a box platen never comes to rest. A plane (vertex contacts) is clean.
- The grid's single-diagonal Kuhn pattern gives 16 % / 6 % normal->shear cross-talk (11x15x3 / 21x29x5).
- The flex carries 2/3 of `mass` (pinned share dropped). Vertex joints inherit childclass
  armature/stiffness/frictionloss.
- PCG needs 116 iterations on the robot (opt.iterations = 100).
