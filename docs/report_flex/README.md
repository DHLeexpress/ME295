# docs/report_flex: MuJoCo flex study of the blade PDMS skin

- `flex_study.tex` / `flex_study.pdf`: the compact report (5 pages). Build with `pdflatex flex_study.tex` (twice).
- `SUMMARY.md`: one page, plain language.
- `make_figs.py` -> `fig/*.pdf`: the report's figures, re-plotted from `study/`.
- `../media/flex_contact_closeup.mp4`: close-up video of the layer under eight objects (1920x1080, 41.7 s).
- `study/`: a curated copy of the study's task directories (scripts, notes, compact results; raw runs, logs, frames
  and downloaded papers are left out, so heavy sweeps must be re-run to regenerate them):

| directory | what it holds |
|---|---|
| `internals/` | code reading of MuJoCo 3.17 flex (`NOTES.md`), patch tests, contact census, the doubled-lambda bug draft (`bug_lambda.md`, `bug_repro.py`) |
| `survey/` | external code and MuJoCo issues that use flex (`NOTES.md`, `sources.json`) |
| `gt_transform/` | exact bonded-layer solution (integral transforms + boundary-element contact) for local loads: `results.json` |
| `gt_fem/` | independent quadratic-hex FEM for the finite pad (`fem_layer.py`, `results.json`) |
| `bench_static/` | MuJoCo's elastic operator vs cell size, conditioning (`summary.csv`, `patch.json`) |
| `bench_dynamic/` | MuJoCo with contacts vs cell size and contact settings |
| `compare/` | errors vs ground truth, cost, the cell-structure choice (`README.md`, `summary.json`, `recommended_pdms_flex.xml`, `fig/`) |
| `robot/` | the robot divergence: island-metric bug, minimal repro (`repro/`), engine patch (`fix/`), contact settings |
| `video/` | the close-up video's simulation and rendering scripts |
| `torsion/` | torsional grip of a grasped box: cells vs the 6x8x2 flex (`torsion.py`, `out_*.json`) |

The scripts were written for the scratch layout in which they ran (`flexstudy/<task>/`, mirrored here as
`study/<task>/`); a few still refer to `<scratch>` paths outside the study and need adjusting before a re-run.
MuJoCo 3.17.0; timings were measured on a shared machine and are indicative only.
