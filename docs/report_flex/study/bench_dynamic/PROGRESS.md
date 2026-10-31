# bench_dynamic PROGRESS (successor: read this first)

## Done (reconstructed 2026-10-30 ~20:15 from files of the cut-off attempt)
- rig.py: one simulation per call (cfg JSON) -> runs/<id>.json (+ .npz time series). Pad = flexcomp grid in static body,
  back face pinned; indenter = 3 slide joints + position servo kp 1e10 (or mocap); robot options.
- sweep.py: groups main conv raw tri quad split contact newton nu49 iter driver smoke; 90 procs.
- FULL SWEEP COMPLETED (logs/sweep_main_tri_quad_raw_contact_newton_split_iter_driver_conv_nu49.log, 1628 configs,
  1391 ok, 77 too_slow, 158 skipped_branch_too_slow, 2 diverged (trilinear 5x7x1 C1 and C1f split3)).
  NOTE: timings measured with ~90 concurrent processes on the machine (contention possible).
- analyze.py: derive() + results.json writer (no tables/plots yet).
- gt_fem/results.json, gt_transform/results.json did NOT exist at 20:15.

## Next
- run analyze.py -> results.json; write report.py (tables + plots); timing re-check of a few runs with low load;
  README.md.

## 20:30 update
- analyze.py run -> results.json (1628 records). Key facts checked (q.py = small query helper):
  * CONTACT "margin" preset had a rig bug: flex+geom margins are SUMMED for flex-geom pairs
    (engine_collision_driver.c:1968/1986) -> 0.1 mm preload; old runs kept, labelled; new preset "margin_flex",
    group margin2 (running).
  * "default" contact -> force ~0 (pf = 1). stiffest / mass_scale runs show penetration ~ 1/vertex-mass.
  * nz = 1 meshes: 10-14 % of the contact force goes straight into pinned vertices (pinned_weight_share).
  * mocap driver cannot transmit shear (friction is velocity-level, mocap has no velocity): C2 mocap K ~ 0.
- timing group (rep=1, 4 procs) queued after margin2.
- Next: report.py (tables/plots), README.md.

## 20:55 update
- margin2 (28 runs) and timing (10 runs, 4 procs) done; analyze.py + report.py rerun (tables/, plots/).
- Timing: sweep (90 procs) us/step inflated x1.1-1.7 vs low load (quadratic x3.3); results bit-identical.
- Next: README.md (only remaining deliverable).

## 21:00 DONE: README.md written. All deliverables present (results.json, tables/, plots/, README.md).
If gt_fem/gt_transform results.json appear: python analyze.py && python report.py (check parse_gt format).
