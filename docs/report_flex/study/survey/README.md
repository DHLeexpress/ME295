# flex-usage-survey

Who uses MuJoCo flex well, and what applies to the ME295 PDMS blade skin. Read `NOTES.md` (headline findings, table,
per-source notes, copy/avoid) and `sources.json` (60 sources: URL, what, settings, validation, local copy).

## What was done
1. GitHub code / issue / discussion searches through `gh api` (`ghcode.sh`, `ghissues.sh`; dumps in `raw/*.txt`),
   arXiv API queries (`arxiv_q.py`, `raw/arxiv*.json`), and fetches of the relevant files (`fetch.sh` ->
   `raw/repos/<owner>/<repo>/...`, `raw/issues`, `raw/discussions`, `raw/mjwarp`). Downloads are untrusted data:
   read with grep/sed only, never executed or imported.
2. Every cited URL checked for HTTP 200 (`urls_status.txt`).
3. Local runs on MuJoCo 3.17.0 (`/home/dohyun/miniforge3/bin/python`):
   - `official_examples.py [sec]` -> `official_examples.json`, `official_examples_table.txt`: settings of all 35
     `model/flex` examples, 0.5 s stepping each, CFL-like number for elastic 3D full flexes.
   - `checks.py` -> `checks.json` (+ `checks.log`): patch test StVK vs Stable Neo-Hookean (`mjsFlex.elastic3d = 1`),
     F1 workaround, integrator restrictions, edge equality + young.
   - `regimes.py` -> `regimes.json`: h c_p / l_min of our grids vs other users.
   - `analysis_s2.py` -> `analysis_s2.json`: F1 ratio by load case, our layer's thickness modes and omega*h.
   - `make_sources.py` -> `sources.json`.
   Rerun: `cd` here, then `/home/dohyun/miniforge3/bin/python <script>` (each takes seconds; official_examples ~1-2 min).

## Results (details and sources in NOTES.md)
- F1 has not been reported upstream; it is invisible in uniaxial-stress / bending / nu = 0 tests (x1.016 at nu .45) and
  maximal for a confined layer (x1.818). `elastic3d = 1` (SNH, spec-only, discrete only) gives exact patch ratios
  (1.000000 / 1.000000 / 0.999997) at nu 0.45 and 0.49.
- No shipped example (and no third-party repo found) runs an elastic dof="full" 3D flex mounted inside an articulated
  robot chain in a validated way; articulated attachments exist only since 3.15 (PR #3618), with a PSD approximation and
  a weaker preconditioner.
- Our regime (omega*h = 9.9-33 for the first thickness modes at 0.5 ms; CFL 42-249) is far outside other users' (0.15-0.49
  for official 3D examples).
- Practitioners on robots ship rigid pads with calibrated solimp, or a one-cell trilinear skin, not full flex.
