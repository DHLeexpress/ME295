"""Robot confirmation runs (UR10e + 2F-140, both PDMS flexes, 2.7 s close-and-lift of a 300 g box) for the candidate
cell structures / contact settings. Uses robot_grasp.py (copy of ../robot/grasp.py + flex_mass_scale) with
island disabled and F1-corrected inputs. Output: out/robot_grasp.jsonl (one JSON line per run), logs/robot_*.log.
python robot_runs.py [nproc]"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = []
for grid in ("6x8x2", "11x15x2", "11x15x3"):
    for ms in (1, 10, 100):
        RUNS.append(dict(grid=grid, flex_solref=[1e-3, 1], flex_mass_scale=ms, tag=f"{grid}_B_m{ms}"))
for grid in ("6x8x2", "11x15x2"):
    RUNS.append(dict(grid=grid, flex_solref=[1e-4, 1], it=1000, tol=1e-10, tag=f"{grid}_A_m1"))
for tc in (2e-4, 3e-4, 5e-4):                      # intermediate flex-contact time constants on the cheap grid
    RUNS.append(dict(grid="6x8x2", flex_solref=[tc, 1], flex_mass_scale=1, tag=f"6x8x2_tc{tc:g}_m1"))
RUNS.append(dict(grid="6x8x2", flex_solref=[1e-3, 1], nu=0.49, tag="6x8x2_B_m1_nu0.49"))
for r in RUNS:
    r.update(island=False, f1=True, save=os.path.join(HERE, "runs_robot", r["tag"] + ".npz"))


def one(o):
    out = os.path.join(HERE, "out", "robot_grasp.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(ln)["o"]["tag"] for ln in open(out) if ln.strip()}
    if o["tag"] in done:
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", MUJOCO_GL="egl")
    p = subprocess.run([sys.executable, os.path.join(HERE, "robot_grasp.py"), json.dumps(o)], capture_output=True,
                       text=True, cwd=HERE, env=env, timeout=4 * 3600)
    with open(os.path.join(HERE, "logs", f"robot_{o['tag']}.log"), "w") as f:
        f.write(p.stdout[-5000:] + "\n---stderr---\n" + p.stderr[-5000:])
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    with open(out, "a") as f:
        f.write((lines[-1] if lines else json.dumps(dict(o=o, error=p.stderr[-1500:]))) + "\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(RUNS)
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(one, RUNS))
