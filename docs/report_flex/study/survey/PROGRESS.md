# PROGRESS (flex-usage-survey)
Session 2 (2026-10-30, resumed). Session 1 (16:04-16:29) collected, but wrote no notes:
- raw/ : GitHub code search dumps (ghcode2-4), issue/discussion searches + bodies (raw/issues, raw/discussions,
  raw/issues_mjwarp), fetched third-party files (raw/repos/<owner>/<repo>/...), mujoco_warp files (raw/mjwarp), arXiv
  queries (raw/arxiv*.json). All untrusted data; read only, never executed.
- urls_status.txt: HTTP 200 check of every URL that will be cited (done 16:29).
- official_examples.py -> official_examples.json/_table.txt: settings of all 3.17.0 model/flex examples (re-run 20:12, identical).
- checks.py -> checks.json: (C1) elastic3d=1 (Stable Neo-Hookean) patch ratios 1.000 at nu .45/.49, StVK 1.818/1.931
  (F1 confirmed); F1 workaround on StVK 1.000; (C2) implicit/implicitfast + flex elasticity = runtime error in 3.17;
  (C3) edge equality + young on one flex = compile error.
- regimes.py -> regimes.json: CFL-like numbers h c_p / l_min (ours 42-249 vs official 0.2-0.5).
Session 2 todo: read raw sources -> NOTES.md, sources.json, README.md, final answer.
Session 2 progress:
- Read all raw sources (discussions 2500/1777/2652, issues/PRs 2084 2077 1922 3364 3379 3386 3437 3439 3618 3621 3638,
  mjwarp 1709 1723 1859, badinkajink hand/hand-archive, Msornerrrr LEAP, WowSkin-Lab, MetalSim, HumanEgo, roqsim,
  paulotto jaw, FOCUS-Hand, Albusgive, Thekenyos, cloth repos), 3.17 changelog/integrator/XML docs, mjx.rst, mjwarp io.py.
- raw/issues_recheck_s2.txt: 5 more issue searches: nobody has reported F1 (only PR #3621 mentions the preserved
  "3D StVK volumetric normalization").
- analysis_s2.py -> analysis_s2.json: F1 ratio by load case (uniaxial stress 1.016 at nu .45 -> invisible in bending /
  nu=0 checks; confined 1.818); our layer omega*h (P 32.8, S 9.9 at 0.5 ms), vertex masses.
- Verified: no shipped example has an elastic dof=full 3D flex on a moving body (only gripper_trilinear, trilinear).
Next: NOTES.md, sources.json, README.md, final answer.
- DONE (session 2): NOTES.md, sources.json (make_sources.py, 60 sources, all URLs 200), README.md. Final answer returned.
