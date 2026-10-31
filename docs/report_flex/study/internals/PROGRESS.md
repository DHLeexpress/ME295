# PROGRESS (flex-internals)

## Session 1 (cut off; no notes written) - outputs reconstructed in session 2
Done (scripts/ -> out/):
- p0_patch_ext.py -> out/p0_patch_ext.{txt,json}: full StVK lam_eff = 2 lam exactly (nu 0.3/0.45/0.49, cube, layer 11x15x3,
  rotated parent, rotated+freejoint, 1% finite strain: fits vs exact SVK resid <1e-11); remap gives 1.0000; SNH (elastic3d=1)
  small-strain exact (lam, mu ratio 1); trilinear/quadratic exact (incl. rotated parent); trilinear+strain eq: E_eff tiny,
  nu_eff 0.3 (needs explanation); membrane dim=2: mu_eff/mu = 0.25, lam_eff = 0.5 lam3D (needs check of reference/code).
- m1_compile_inspect.py -> out/m1_compile_inspect.*: 6 tets/hex, single diagonal, 165 pinned, masses, robot opt.
- t1_tet_bias.py -> out/t1_tet_bias.*: Kuhn (MuJoCo) pattern vs mirrored J1: normal->shear crosstalk 16% (11x15x3), 6% (21x29x5);
  J1 0. Normal patch compliance Kuhn 2.37 vs J1 2.78 um/N (11x15x3).
- c1_contact_census.py: box/sphere/capsule/cylinder -> element contacts (layer 0 tets) w/ inverse-distance weights incl. interior
  vertices (~21%); plane -> vertex contacts (165). trilinear -> 50 contacts.
- c2_platen_static.py, c2b: C1 rig K_n with our contact settings = 8.76 (box), 7.93 (plane) N/um vs flex-only 15.87 (no-slip
  vertex-prescribed, doubled lambda): contact compliance in series ~ equal to the layer. refsafe on -> 1.32/0.65.
- c3_chatter.py, c3b: box (element contacts) never settles: contact point jumps between tet vertices each step (F cv 4.5%),
  plane settles (cv 0.4%).
- i1_metric_pcg.py: PCG iterations to tol 1e-8: fixed parent 36 (11x15x3) .. 107 (41x57x5); hinge parent (matrix-free, no blocks)
  133 .. 269; robot 116.

## Session 2
- Code read + line refs collected (stiffness user_flexcomp.cc:1179-1372,2820-2862,3039-3049; passive engine_passive.c:587-660;
  core_util.h:289-370; contact collision_driver.c:572,825-860,1897-1915,2110-2170; collision_flex.c:216-392;
  core_util.c:618-657,785-830; constraint.c:1800-1830,1885-1925,2030-2122; metric.c:225-246,1638-1714,2485-2491,3040-3134;
  derivative.c:1901-1913; user_model.cc:3694-3702; flexcomp mass/bodies user_flexcomp.cc:476-560; damping ban 2715).
- t2_models_tests.py -> out/t2_models_tests.{txt,json}: example models/tests regime inventory.
- v1_misc.py -> out/v1_misc.{txt,json}: childclass armature/stiffness/frictionloss inherited (damping -> compile error);
  flex mass = 2/3 of attribute (pinned dropped); hinge+flex+joint equality diverges at step 3 with islands, stable with
  mjDSBL_ISLAND (3x3x2 and 11x15x3); plane vertex-contact stiffness = K imp^2 m_v/(1-imp) (105887 N/m at tc 1e-4 ->
  17.47 N/um total face), saturating at 9999 m_v/h^2 = 4.235e5 N/m (69.6 N/um at tc 1e-6).
- Membrane: test ElasticEnergyMembrane (engine_passive_test.cc:528) codifies the 1/4 factor (4*energy/volume).
Next: bug_repro.py + bug_lambda.md, NOTES.md, README.md.
- bug_repro.py -> out/bug_repro.txt (ratios 1.8182 / 2.0000). bug_lambda.md drafted (not posted).
- NOTES.md and README.md written (session 2). Task complete unless further checks are requested.
