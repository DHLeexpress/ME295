"""Run a list of option dicts through a script in parallel subprocesses; append JSON lines to out/<name>.jsonl.

python sweep.py <name> <script.py> <P> '<json list of option dicts>'   (or @file.json)
Each subprocess is single-threaded (OMP/MKL threads = 1). Option keys used here: "env" (dict of env vars),
"pymj" (true: run against the instrumented build in ./pymj, see build/).
"""
import sys
import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
name, script, P, spec = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
opts = json.load(open(spec[1:])) if spec.startswith("@") else json.loads(spec)
out = HERE / "out" / f"{name}.jsonl"
log = HERE / "out" / f"{name}.log"
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl")


def one(o):
    e = dict(env)
    e.update({k: str(v) for k, v in o.get("env", {}).items()})
    if o.get("pymj"):  # patched/instrumented libmujoco (copy of the mujoco package in ./pymj)
        e["PYTHONPATH"] = str(HERE / "pymj")
    r = subprocess.run([sys.executable, "-W", "ignore", str(HERE / script), json.dumps(o)], cwd=HERE,
                       capture_output=True, text=True, errors="replace", env=e, timeout=o.get("timeout", 3600))
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    with open(log, "a") as fh:
        fh.write(f"### {json.dumps(o)}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}\n")
    if not lines:
        lines = [json.dumps({"o": o, "error": (r.stderr.strip().splitlines() or ["?"])[-1][:300], "rc": r.returncode})]
    with open(out, "a") as fh:
        for ln in lines:
            fh.write(ln + "\n")
    return lines


with ThreadPoolExecutor(P) as ex:
    for lines in ex.map(one, opts):
        for ln in lines:
            print(ln[:400], flush=True)
