# flexstudy/robot: the dof="full" PDMS flex on the UR10e + 2F-140 (MuJoCo 3.17.0)

Findings, evidence tables and recommendations are in **NOTES.md**. Resumable log: PROGRESS.md.

## Short version
1. Divergence (F4) is an engine bug in 3.17.0. The per-island constraint-solver metric (`flexMatrixFree` in
   engine_metric.c) drops the stiffness of any flex with articulated attachments; the monolithic path keeps it. Any
   constraint row creates an island, so the robot diverges at step 3. Minimal repro: `repro/minimal_repro.xml`
   (one hinge, 3x3x2 flex, one equality). Fix: `fix/island_flex_fix.patch`; workaround: `<flag island="disable"/>`.
2. Then the contact settings: flex contact solref 1e-4 with the 100-iteration CG produces force spikes of 1-65 kN, and
   the StVK layer snaps through and stays inverted (the edge-length energy has no inversion barrier). Working settings:
   A = solref 1e-4 + iterations 1000 / tol 1e-10 (120 N per side, no rigid-blade load, 145 ms/step), or
   B = solref 1e-3 (120 N per side, 28 ms/step, but up to 0.6 mm contact penetration).

## Files
- common.py: spec loading and edits (F1 inputs, options), BADQACC counter. run.py: one stability run at home pose (JSON
  line). sweep.py: parallel runner -> out/<name>.jsonl (+ .log). summ.py, c1summ.py, g2summ.py, metsumm.py: tables.
- repro.py: per-step diagnostics on the robot. minimal.py: 1-hinge models (writes repro/min_*.xml). repro/run_repro.py:
  stand-alone repro. integ.py: integrator/solver matrix on the repro.
- metric.py: explicit M, finite-difference K/D (all dofs), engine operators (monolithic and island) via the
  instrumented build, eigen checks -> out/metric_*.json/.npz.
- grasp.py: 2.7 s close-and-lift proof with per-step force, compression and inversion logs. grasp_diag.py,
  grasp_diag2.py: contact partition and geometry per phase (saves states). contact_trace.py: replays from a saved
  state with a per-step trace. mirror_test.py: flex force vs through-thickness push, out to the mirrored state.
- asm/: copies of the scratch assemblies (asm/model -> repo model/). build/: instrumented 3.17.0 build (sources
  build/mujoco_src, patches build/me295_instrument_*.patch, library build/bld/lib). pymj/: the python package with that
  library. fix/: the clean patch.

## Rerun (python = /home/dohyun/miniforge3/bin/python, run from this directory)
- Repro: `python repro/run_repro.py` (stock wheel). With the fix: `PYTHONPATH=pymj MJ_FIX_ISLANDFLEX=1 python repro/run_repro.py`.
- Robot stability sweep: `python sweep.py s1 run.py 28 @out/s1_opts.json; python summ.py out/s1.jsonl`.
- Minimal variants: `python sweep.py m1 minimal.py 21 @out/m1_opts.json`.
- Metric check: `PYTHONPATH=pymj MJ_FIX_ISLANDFLEX=0 python metric.py '{"tag":"robot_fix0","model":"robot"}'` (about 3 min).
- Effective-metric solve trace: `PYTHONPATH=pymj MJ_EFFDBG=1 MJ_SOLDBG=1 python repro.py '{"steps":5}'`.
- Contact sweep (6x8x2): `python sweep.py c1 grasp.py 13 @out/c1_opts.json; python c1summ.py out/c1.jsonl`.
- Proof: `python sweep.py p1 grasp.py 10 @out/p1_opts.json; python c1summ.py out/p1.jsonl` (the setting-A run takes about 16 min).
- Rebuild the instrumented library: cmake Release in build/bld from build/mujoco_src (see build/cmake_config.log, build/build.log).

Nothing under /home/dohyun/projects/ME295 was modified. No packages were installed.
