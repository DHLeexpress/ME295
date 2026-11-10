# 60-second explainer reel of the results

`results_60s/` is a [paper-reel](https://github.com/haiderfarooq3/paper-reel) piece: a 9:16, 59.8 s cut-paper reel
that walks through [../RESULTS.md](../RESULTS.md). paper-reel is a Claude Code skill (MIT, commit `9c5bc57`).
- **The headline table** is built on screen as a paper table: rows = test suites, columns = policies, transposed to fit
  the vertical frame. Each row appears when its suite is named and each cell when its number is said. Every cell
  shows the count, the percentage and a bar in the policy's colour (the report's figure colours).
- **A second table** shows where the fine-tuned policy's 39 misses happen (§3.2), with the 33 contact failures
  bracketed.
- **The last shot** is the same table with an empty "+ TOUCH" column.
- **Between the tables**, paper "monitors" play real simulation footage.

**Output:** [../results/media/reel_results_60s.mp4](../results/media/reel_results_60s.mp4) (720 × 1280, 5.4 MB,
−14.7 LUFS). The 1080 × 1920 render stays in `results_60s/renders/final.mp4` (git-ignored).

| Line (voice) | On screen |
|---|---|
| Could the best open robot model pick one named book out of a packed bookcase? | nominal shelf, instruction typed |
| We tested Pi 0.5 on our robot arm in simulation, on four test suites. | the cell (wide view), "4 test suites" |
| Its training scene, shuffled layouts, narrower gaps, and heavy, slippery books. | table: the four rows appear |
| Out of the box, it scored 0 for 20. It crashed into the bookcase. | zero-shot clip, 0/20, protective stop |
| Fine tuned on 251 demos, it went 50 for 50 in the scene it trained on. | fine-tuned clip, 50/50 |
| On shuffled shelves it drops to 72 percent. Narrower gaps: 73. Heavy, slippery books: just 43. | table: fine-tuned column fills |
| A scripted expert that knows every pose sets the ceiling: perfect, except heavy books, at 57. | table: expert column fills |
| So where do the 39 misses happen? 33 of them, at contact. | failure-mode table |
| On thick books, the blade tips push the book deeper into the shelf. The pads never touch it. | push-in clip (56 mm) |
| In narrow gaps, it opens as wide as in training and shoves the neighbors aside. | shove clip (56 mm) |
| And heavy books slip out of the fixed grip, for the policy and the expert alike. | slip clips: fine-tuned, then expert |
| The catch: this is simulation, with scripted demos and one training run. | caveats |
| Cameras can't feel slip. Our tactile skin can. That's the next test. | camera ✗, skin taxels; table + touch column |

Every number comes from the evaluation in `results/eval/`. The clips come from re-runs of the same test cases
(`data/media_runs/`; the list is in `scripts/make_explainers.py`). Because the policy samples its actions, each re-run
is a new episode, so the clips illustrate the failure modes but are not counted in the table.

## Rebuild

```bash
REEL=~/.claude/skills/paper-reel/bin/reel            # install: see the skill's README; `$REEL doctor` checks it
                                                     # (Linux: whisper.cpp built from source, Kokoro in ~/.venvs/kokoro)
python baseline/scripts/make_explainers.py reel-assets      # sprite sheets + row thumbnails -> assets/, src/clips.js
$REEL voice baseline/reel/results_60s --voice am_michael --speed 1.25     # Kokoro takes + whisper.cpp word times
python baseline/scripts/make_explainers.py reel-timing      # clamp word times to their line (captions never lead a cut)
$REEL check baseline/reel/results_60s                       # text: PASS, deterministic: true
$REEL export baseline/reel/results_60s                      # -> results_60s/renders/final.mp4 (1080 x 1920)
python baseline/scripts/make_explainers.py reel-publish     # -> results/media/reel_results_60s.mp4 + _strip.jpg
```

The media re-runs need the fine-tuned and DROID policy servers (see `scripts/evaluate.py --media`). On Linux the kit
replaces the macOS fonts with Liberation Sans (headline) and DejaVu Serif (captions).
