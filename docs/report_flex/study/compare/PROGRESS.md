# compare: PROGRESS (resumable log)
## 02:45 start (no earlier partial work existed in compare/)
- Read inputs: gt_transform, gt_fem, bench_static, bench_dynamic READMEs; robot/NOTES.md sec 2-3; internals NOTES (contact k formula).
- Plan: gt.py (GT loader) -> check_gt.py; check_static_dynamic; analyze.py (error tables static/dynamic);
  rig2.py (copy of bench_dynamic/rig.py + C3, offsets, presets) -> confirmation sweep; plots; XML; README.
## 03:05 milestone 1
- gt.py, check_gt.py -> out/check_gt.json: C4 (12) and C6/C6f (6) agree within combined bars (max rel 2.7e-5 / 1.2e-3).
- analyze_static.py -> out/static_errors.csv, static_orders.json (two-param fit e = a dx^p + b dz^q: p ~ 0.8-1.2 all types).
- rig2.py (copy of bench_dynamic/rig.py + C3, offsets, plane, presets robotB/tc2e-4), plan.py, sweep2.py.
- sweeps done: sd (24), robot (288), robotm (384), rig (168; 9 too_slow = quadratic 20x28x1 no-slip + 40x56x2 C2 stiffest).
  sd result: plane indenter (vertex contacts) + mass x100 + stiffest reproduces static K to -0.4 % (static/dynamic consistent);
  box (tet contacts) is +3.6..+7.7 % stiffer than exact BC at mass x100, and -13..-21 % at robot settings.
- analyze_dynamic.py -> out/dynamic_errors.csv, config_errors.csv (placement 95% CI; flags unphysical / snap).
- robot_grasp.py (copy of robot/grasp.py + flex_mass_scale) + robot_runs.py -> out/robot_grasp.jsonl (11 runs, running).
- Next: figures (fig_style.py, figures.py), choose structures, nu49 confirm, XML, summary.json, README.
## 03:10 milestone 2
- robot_grasp: 6x8x2 B (solref 1e-3) stable 6.0 ms/step; mass x10/x100 chatter -> rejected; solref 2e-4..5e-4 -> overload/inversion.
- graded mesh option in rig2.py (flexcomp type=direct); smoke test C5 0.15 mm core: F(30um) 0.049 N vs exact 0.0440 (+11 %).
- figures.py (fig_static_convergence, fig_thickness, fig_pareto, fig_solver, fig_contact, fig_candidates_*), out/chosen.json,
  recommended_pdms_flex.xml (checked by check_xml.py -> out/check_xml.out), make_summary.py -> summary.json.
- Running: graded group (13), nu49 group (96). Next: final analyze/summary/figures, README.md.
## 03:20 DONE
- All groups finished (graded 13, nu49 96, robot grasps 16 lines incl. one identical duplicate of 11x15x2_A_m1).
- Final: out/*.csv|json, summary.json, fig/ (7 figures, pdf+png), recommended_pdms_flex.xml (check_xml ok), README.md.
