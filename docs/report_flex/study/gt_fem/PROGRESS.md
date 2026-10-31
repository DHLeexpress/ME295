# gt_fem PROGRESS (successor: read this first)
## Session 2 (resumed 2026-10-30 ~20:15)
Found from session 1 (no PROGRESS.md existed):
- fem_layer.py = checked copy of fem_ref/fem_layer.py (orig kept as fem_layer_orig.py); changes listed in its header
  (graded_breaks fix, faster NDCholesky, threads, MMS helpers, face-traction maps).
- cases.py: case definitions G1,C1,C1f,C2,C3,C6,C6f (no C4 yet), meshes, BCs, run().
- verify_fem.py: V1 (kron vs elementwise, 1e-15), V2 (patch, 1e-13), V3 (G1 exact 1e-14), V7 done -> verify_fem.json.
  V4 (MMS), V5 (ND vs SuperLU), V6 (symmetry) NOT yet run.
- _prof_m2.log: C1 nu=.45 full m=2 (408639 dof): K = 8567.119 N/mm (old factor code, 490 s).
- Session 2 added: run_cases.py (sweep: runs/CASE_FORM_nuNU_mM.json, logs/), axi_fem.py + run_c4.py (axisymmetric Q2
  code for C4; verified: c4_verify.json patch 1e-12, confined disc 1e-14; runs_c4/), V4b solenoidal MMS in verify_fem.py.
- V5 (ND vs SuperLU) done: rel diff <= 6e-14.  V4 (MMS, non-solenoidal): rates -> 3 (L2), 2 (H1); mixed-form errors scale
  with lambda there because p = lam div u = O(lam) (expected for a mixed method; V4b tests div u = 0).
- Pressure-method source opened: Kelly & Van Engelen, PEER 2015/03, sec. 2.1 eqs 2.1-2.11, 2.32-2.34
  (refs_download/kelly.txt). Classical compressible pressure eq: lap p - 12G/(K t^2) p = -12 G eps_c / t^2, p=0 on edges.
- Timing: C1 m=2 (409k dof) 35 s / 11 GB at 32 threads (new NDCholesky).
- 20:35 levels 1-2 of all 3D cases x 5 (form,nu) combos done/finishing (runs/). chain_m3.sh launched: level 3
  (--reduced: full forms + mixed 0.499, + mixed 0.45/0.49 for C1), C3 first (56 thr), then rest (2x32 thr), plus V4b.
  When logs/chain_m3.done exists: run postprocess.py, plots.py (fig/), write README.md.
- axi_fem.solve now uses SPD SuperLU options (no pivoting); C4 runs restarted (runs_c4/, logs/c4_runs.log).
- First numbers (m=1/2): C1 nu.45 K=8.5671 N/um (m2), pressure method 7.337 (-14%), M A/t 9.157 (+6.9%).
- NOTE: verify_v456 process will overwrite verify_fem.json V6 with the OLD (buggy-mesh) C3 entry: rerun 'verify_fem.py v6' at the end (cases.singular_lines fixed for names like C3_fullmodel; after fix C3 half vs full 8.5e-15).
- 20:47 chain_m3.sh killed (was blocked on verify); chain_m3b.sh runs level 3 (C3 48 thr, then rest 2x28 + V4b 8 thr, then V6 rerun) -> logs/chain_m3.done. C4 runs complete (p~1, full/mixed agree 3e-6 rel, R=25.5 vs 51 < 2e-7). postprocess.py now writes summary_table.md.
## Session 3 (resumed 2026-10-31)
- chain_m3b finished (all level-3 runs rc=0; C3 m3 3.39M dof 1094 s, 154 GB; others 1.35-1.63M dof ~200-480 s, 58 GB).
- V4b had a NameError (generator scoping of j); fixed in verify_fem.py (old copy logs/verify_fem_pre_v4bfix.py.bak), rerun.
- postprocess.py: added compare_transform() (C4/C6 vs ../gt_transform/results.json) + code_verification block.
- plots.py: C6 profile overlay of the exact infinite-layer profile.
- Level-3 numbers (postprocess, logs/post_m3.log): C1 .45 8.56616+-6.8e-4; C1f .45 8.01016; C2 .45 0.827911; C3 .45 3.59267;
  C6 .45 0.216632 (transform 0.21658); C6f .45 0.199756 (transform 0.19971).
- Next: rerun postprocess+plots after V4b, README.md.
- 2026-10-31: postprocess (with compare_transform) + plots rerun; README.md written (summary table, verification,
  pressure method, gt_transform comparison: C4 rel diff <= 2.7e-5, C6 <= 1.2e-3, all within bars).
- Remaining: when V4b (graded family) finishes -> rerun postprocess.py (puts V4b rates into results.json code_verification),
  check README V4b row against logs/verify_v4b.log.
- V4b finished (4968 s): mixed locking-free (graded L2 2.88e-5 at all nu), full locks (x5 at .499, x21 at .49999).
  postprocess + plots rerun; README updated. gt_fem task COMPLETE.
