# PROGRESS (robot-mount task) -- resumable log

## Reconstructed from the earlier (cut-off) attempt, 2026-10-30 ~16:30
- common.py (load/edit specs, F1 inputs, BADQACC counter), run.py (stability run -> JSON line), sweep.py (parallel runner,
  out/<name>.jsonl), repro.py (per-step diagnostics), grasp.py (proof run), grasp_diag.py (contact partition on the box).
- build/: instrumented MuJoCo 3.17.0 copy (build/mujoco_src, patches build/me295_instrument_*.patch), lib in build/bld/lib,
  python package copy using it in pymj/mujoco (PYTHONPATH=pymj). Env switches: MJ_FIX_ISLANDFLEX=1 (fix), MJ_EFFDBG=1, MJ_SOLDBG=1.
- Hypothesis H1 (source-verified 16:3x): island metric (engine_metric.c flexMatrixFree, used by mj_effMulAddIsland) gates on
  mjd_flexStiff_active, which returns 0 for any flex that is not mj_flexSimple; monolithic mj_effMulAdd (line ~243) handles
  !mj_flexSimple flexes matrix-free. Our pinned flex is non-simple (pinned verts weld to finger body with hinge dofs) ->
  its stiffness is missing from the per-island constraint-solver metric.
- out/g1.jsonl (grasp.py, 11x15x3 unless noted, box half 25x40x30 mm, 2.7 s sim):
  wheel islands ON: fatal stack overflow at step 3 (43836 CONTACTFULL) = divergence.
  wheel islands OFF: no BADQACC, 25 ms/step (nu .45) ; +F1: 24 ms/step ; 6x8x2 F1: 2.8 ms/step ; 21x29x3 F1: 92 ms/step
  pymj MJ_FIX_ISLANDFLEX=1 (islands ON): no BADQACC, 28 ms/step ; +F1 26.7 ms/step
  CONCERN: left/right normal forces unequal (e.g. 45 vs 28 N), 'other' contacts on box = 8, skin max_compression 1.8 mm
  (> 1.5 mm thickness) -> contact forces not yet shown sensible. grasp_diag.py written to partition contacts, not yet run.


## Session 2 (20:1x) done
- out/s1.jsonl (my 41 tags appended after older tags of the earlier attempt; summ.py keeps last per tag). Islands ON, wheel:
  every material (nu .3/.45/.49, F1, E 1e4/1e5), solver (it 1000, tol 1e-12, h 1e-4..1e-3, impratio 1, pyramidal, mem 2 GB,
  nowarm), bisection (noeq, noact, one_flex, pdms_mass 1e-2, armature 0, lock_arm, lock_arm_but_w3, grids 6x8x2..21x29x3)
  diverges at step 3-5. Stable: islands OFF (any material), pymj fix, lock_grip / lock_all (no constraint rows -> no island),
  nograv (trivial: nothing moves). Newton + noslip refused by the engine ("flex with general attachments requires solver='CG'",
  "PGS and noslip not yet supported with flex"). damp0 errored (my bug: spec joint damping is an array) - not rerun.
- minimal.py + out/m1.jsonl: ONE hinge + pinned flex + ONE constraint row (joint equality, or active limit, or floor
  contact) diverges at step 3-6, even for a 2x2x2 flex (nv 13) and 3x3x2 (nv 28); no constraint -> stable; island disabled
  -> stable; MJ_FIX_ISLANDFLEX=1 -> stable. repro XMLs: repro/min_m_*.xml
- MJ_EFFDBG trace (11x15x3 min, eq): mj_effSolve never takes pAp<=0 (negcurv=0), capped at 100 it with relres 1e-8..3e-6;
  island CG converges in 2 it without fix (metric = M only), 25-50 it with fix.
- metric.py (out/metric_min3x3x2_fix{0,1}.json): monolithic engine operator == h^2 K_fd on flex block (relerr 9e-9);
  island operator flex block = 0 exactly without fix; max gen. eig (M+S_isl)^-1(M+S_mono) = 3598 (fix0) vs 1.000 (fix1);
  all metrics PD (no negative eigenvalues). flex-hinge coupling block 3.8e-6 (vs flex block 1.59): negligible.
- metric.py robot (out/metric_robot_fix{0,1}.json): same picture on the robot (arm joints free): island flex block 0,
  max gen eig 1.5e4; fix -> 1.000; A_ref, M+S_mono PD. Engine general-attachment operator matches FD K (4e-7 relerr).
- effdbg_robot_fix{0,1}.txt: no negcurv exit; flex max|qacc| step1 7.95 (bug) vs 0.455 (fix), then x1800, x5.7e5.
- repro/minimal_repro.xml + repro/run_repro.py (out/repro_wheel.txt, out/repro_pymj_fix.txt). integ.py: Euler diverges
  either way; implicit/implicitfast refused for flex in 3.17; discrete+Newton/PGS refused; RK4 no BADQACC in 200 steps (not
  studied further).
- SECOND PROBLEM (contact quality, after the divergence workaround): grasp_diag.py / grasp_diag2.py / contact_trace.py
  (out/diag_*.txt, out/diag2_noisl_f1.txt, out/trace_noisl_f1.txt): the rigid blade rims carry most of the box load; the
  PDMS front layer snaps through and inverts (90 tets inverted, free face pushed ~1.5 mm) within 20 ms of first contact.
  Cause: box yawed 0.48 deg vs PDMS normal (sim_gripper aligns to the site-to-site vector) -> 0.43 mm wedge over 51 mm ->
  edge strain > StVK peak. mirror_test.py (out/mirror_test.json): free-face force peaks at w=0.3 mm (2637 N F1, whole
  face, front sublayer), zero at 0.75 (flat), reverses sign beyond, ZERO at 1.5 mm (mirror state): no inversion barrier.
- grasp.py got options align (box yaw onto PDMS normal) + end_normal_N (pdms vs rigid blade) + inverted_tets_end.
  sweep g2 (out/g2_opts.json) launched: align/noalign, islands off / fix, grids, nu .49.
- g2 (out/g2.jsonl): align alone does NOT fix the collapse (inverted ~850 tets/side; note g2's inverted_tets_end used a
  wrong sign-vs-median test, fixed later with rest volumes). trace_noisl_f1_align.txt: face-on contact, force rises to
  ~1000 N per side in 25 ms (>> the 2F-140's ~120 N), compression 0.5 mm, then snap-through.
- c1 (out/c1.jsonl, 6x8x2, c1summ.py): ROOT CAUSE of bad contact = flex contact solref 1e-4 (time constant h/5, refsafe off)
  with the CG 100-iteration cap -> spikes (max 65 kN), collapse. solref 0.001-0.002 (2h-4h): held nL=nR=119-121 N +-0.3,
  rigid blade 0-3 N, compression 22-27 um, no inversion. it 1000 with solref 1e-4 also OK (23 ms vs 4.3 ms/step).
  solref >= 0.005: too soft -> blades carry load. solref 5e-4: oscillates/collapses.
- p1 proof sweep launched (out/p1_opts.json): 11x15x3 etc with solref 1e-3, fast standard closing.
- p1/q1 (out/p1.jsonl, out/q1.jsonl): proof done. A: islands off + solref 1e-4 + it 1000 tol 1e-10 -> 120.2/120.1 N, rigid 0,
  comp ~60 um, no inversion, pen 55 um, 145 ms/step. B: islands off + solref 1e-3 -> 121/122 N, rigid 9 N, comp 45 um,
  pen 0.62 mm, 27.7 ms/step; patch same. solimp 0.9999 collapses again. 21x29x3 B ok (105 ms/step).
- NOTES.md, README.md, fix/island_flex_fix.patch written. TASK ESSENTIALLY COMPLETE (20:5x).
