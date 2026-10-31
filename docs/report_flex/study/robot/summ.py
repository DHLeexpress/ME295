"""Print a compact table of run.py results from out/<name>.jsonl (last record per tag)."""
import sys
import json
recs = {}
for ln in open(sys.argv[1]):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    recs[r["o"].get("tag", "?")] = r
for tag, r in recs.items():
    if "error" in r:
        print(f"{tag:24s} ERROR {r['error'][:150]}")
        continue
    lib = "pymj " if "pymj" in r.get("mujoco_file", "") else "wheel"
    print(f"{tag:24s} {lib} nv {r['nv']:5d} steps {r['steps']:5d} first_bad {str(r['first_bad']):5s} niter {r['mean_niter']:6.1f}/{r['max_niter']:4d} "
          f"ms/step {r['ms_per_step']:6.1f} max|v_flex| {r['max_flex_qvel']:.2e} fatal {str(r['fatal'])[:40]} warn {r['warnings']}")
