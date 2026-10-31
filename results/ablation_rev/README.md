# Ablation re-run (2026-10-31)

Same grid as results/ablation (40 runs), with the revised cell model: PDMS 1.5 mm, no damping, Newton solver
(assembly/pdms_option.xml). Outcomes match results/ablation exactly; the skin squeeze per book is 15 / 28 / 57 / 83 N per
blade. Reproduce: python scripts_sim/ablation.py --out results/ablation_rev --figure results/ablation_rev/sim_ablation.png
