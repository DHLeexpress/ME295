# Run timing.py over the nu=0.45 models (plus the robot's raw 11x15x3), NPROC at a time (default 16, to keep the
# wall-clock numbers comparable on a shared machine). Appends to timing.jsonl. Usage: python timing_driver.py [nproc]
import sys, os, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from driver import configs, cost
from common import cfg_id
PY = "/home/dohyun/miniforge3/bin/python"
OUT = os.path.join(HERE, "timing.jsonl")


def go(c):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([PY, os.path.join(HERE, "timing.py"), json.dumps(c), OUT], capture_output=True, text=True,
                       env=env, cwd=HERE)
    line = [l for l in r.stdout.splitlines() if l.startswith(cfg_id(c))]
    print(f"[{time.strftime('%H:%M:%S')}]", line[-1] if line else "FAILED " + r.stderr[-300:], flush=True)


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["id"] for l in open(OUT) if l.strip()}
    cs = [c for c in configs() if c["nu"] == 0.45 and (c["variant"] != "full_raw" or (c["nt"] == 2 and c["nz"] == 3))]
    cs = [c for c in cs if cfg_id(c) not in done]
    cs.sort(key=cost, reverse=True)
    print(len(cs), "timing runs", flush=True)
    with ThreadPoolExecutor(nproc) as ex:
        list(ex.map(go, cs))
