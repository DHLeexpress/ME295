# PROGRESS (gt-transform)
## Done (session 1, before cut-off; re-inspected in session 2)
- gt_transform.py: exact layer kernels (Papkovich-Neuber), Hankel tables, half-space closed forms,
  rectangle + ring influence elements.  derive_sympy.py/.out: symbolic check (Navier residual 0, limits).
- validate.py -> validate.json/.out, fig_kernels.png, fig_green_realspace.png:
  V1 kernels vs Chebyshev ODE <= 2.7e-6; V2 limits exact; V3 table moments = t/M, t/G exactly;
  V4 rect closed forms 8e-14; V5 ring forms <= 2.6e-5 (one nan entry at r=0.999b, reference quad);
  V6 1e-15; V7 square-patch forward check <= 1e-7.
## Session 2
- found bug in closed_forms(): hydrostatic ratio_if_lambda_doubled used (9lam+6mu)/(4.5lam+3mu); correct
  (6lam+2mu)/(3lam+2mu) = 1.9310 at nu=0.45 (matches F1).  Fixed.
## Next
- gt_contact.py: C4 (Fredholm 2nd kind + ring BEM, frictionless; ring BEM bonded), C5 (Fredholm, g(a)=0),
  C6 (rect BEM frictionless + no-slip, quarter symmetry; spectral post-processing for profiles/taxel maps).
- run_all.py -> results.json, plots; literature check; README.
- gt_contact.py written & tested: Fredholm (C4 a=5mm nu=.45 K=0.147570 N/um, converged 1e-14; ring BEM -> same to 4e-6),
  thick-layer limits Sneddon/Hertz to 5e-4/5e-5; bonded ring BEM matches Mossakovskii half-space ratio to 1e-6;
  rect correction (all 9 comps) vs independent polar Fourier quadrature 1e-9.  C6 n=16 nu=.45: K=0.1993 (fric) 0.2162 (no-slip);
  neighbour taxels BULGE (mean u_z ~ -0.048 delta), base Fz centre 84%.
- scipy 1.17 bug: special.struve(0,x) = NaN near its zero x~25.76537; guarded in gt_transform._struve.
- Literature: Argatov et al. arXiv:1203.0918 Table 1 (kappa_s, from Hayes 1972) extracted to lit/argatov_table1.json;
  Dimitriadis PMC PDF not retrievable (JS proof-of-work page); Hayes 1972 / Shull 2002 paywalled, not opened.
- run_all.py launched 20:28 (6 workers x 8 threads) -> runs/*.pkl, run_all.log.  Next: assemble.py -> results.json + figs, README.
## Session 2 (cont., 20:31-21:11, reconstructed in session 3)
- run_all.py finished (run_all.log, 1940 s); assemble.py -> results.json, summary.txt, fig_c4/c5/c6*/literature PNGs;
  lit_asymptotic_check.py/.out; compare_gt_fem.py/.json/.out (vs gt_fem/results.json, levels 1-2); README.md written.
## Session 3 (2026-10-31)
- Re-inspected; all deliverables present.  gt_fem level-3 C6 runs (gt_fem/runs/*_m3.json, 21:45-21:52) postdate
  gt_fem/results.json -> new compare_gt_fem_levels.py -> compare_gt_fem_levels.json/.out: every FEM sequence converges
  monotonically from above; best gaps +1.65e-4 (C6f nu.499 L3), +2.6e-4 (C6f nu.49 mixed L2).  README section 6/7 updated.
- C4 vs gt_fem summary_table.md re-checked: agreement <= 2.7e-5 relative (all 12 entries).
- Dimitriadis 2002 full text retried (EuropePMC REST 500, cell.com HTML wall): still not accessible.
## Status: DONE.  Nothing pending.
