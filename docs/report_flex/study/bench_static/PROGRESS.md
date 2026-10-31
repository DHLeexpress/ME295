# bench_static PROGRESS (resume notes)

## Reconstructed at 20:15 (session 2) from session-1 files (no PROGRESS.md existed)
- Code: common.py (geometry, XML, FD stiffness w/ colouring, checks, surface integrals, solvers), run_model.py (one model:
  FD K, cross-checks, cases G1 C1 C1f C2 C3 C6ns C6f C4n(nodal) C4p(penalty) UP, conditioning + PCG emulation +
  MuJoCo's own PCG iteration check), driver.py (sweep, skips status ok), collect.py (-> results.json, summary.csv),
  plots.py, timing.py/timing_driver.py (cost; NOT yet run), upstream.py (+scratch/upstream_test.json: upstream joint ->
  nefmK=0, preconditioner M^-1), verify_upstream_force.py/.json (momentum check with an upstream slide joint).
- Session 1 sweep: 185/295 configs ok in out/; 33 nz=2 (full) errors = bug in solve_dirichlet when no free dofs (C1 with
  nz=2), already fixed in common.py (verified 20:15, scratch/t2); nt12/nt16 full, nt16 trilinear?, quadratic nt8 never run.
- scratch/out: cost tests of the largest models (25-30 min, 4-8 GB each) - valid but not in out/.
- results.json (16:27) is STALE; re-run collect.py at the end.

## Session 2 plan
1. driver.py 60  -> remaining ~110 models  (log logs/driver_b.log)
2. patch_check.py (P0 exact homogeneous strains from the FD K, all variants) -> patch.json
3. timing_driver.py 16 -> timing.jsonl
4. collect.py, plots.py, README.md

## Session 2 milestones
- 20:25 patch_check.py DONE -> patch.json (154 models, nt<=4 + some nt8): full_raw energy ratios uniaxial 1.818182 / 1.960784,
  shear 1.000000, hydrostatic 1.931034 / 1.986577 (nu .45/.49) = exactly lambda_eff = 2 lambda; full_corr, trilinear,
  quadratic all 1.000000 (|dev| <= 2e-7) incl. nu=0.499. F1 CONFIRMED.
- collect.py extended (patch_P0, err_*_vs_gt columns). upstream.py running -> upstream.json.
- Source check: engine_passive.c 3.17.0 lines ~163-172 (and ~405-414): interpolated-flex node forces use a "fast path"
  that writes only to the node body's own 3 dofs -> no propagation to ancestor joints (explains verify_upstream_force.json:
  trilinear/quadratic Q_base = -sum(free forces) != 0, full flex Q_base ~ 5e-16).
- 20:55 sweep DONE: 295/295 ok (logs/driver_b.log). timing.jsonl DONE (93 runs, 16 procs, shared machine).
  upstream.json DONE. collect.py now reads gt_fem/results.json (interim, 20:42) -> err columns; plots regenerated
  (conditioning.png rewritten). README.md written (all numbers checked against results.json).
- NEXT (optional): when gt_fem level 3 finishes (gt_fem/logs/chain_m3.done), re-run collect.py + plots.py and
  re-check README's GT-error tables (values may shift by ~GT err, <=0.25 %). gt_transform/results.json absent at 21:05.
