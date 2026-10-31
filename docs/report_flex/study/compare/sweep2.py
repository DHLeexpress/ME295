"""Parallel runner for rig2.py. Usage: python sweep2.py <plan.py-group> [--nproc N]
Groups are defined in plan.py (function groups() -> dict name -> list of cfg dicts). Each run is a separate
single-threaded process (rig2.py), cached by cfg id in runs/<id>.json. Log: logs/sweep_<groups>.log."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig2  # noqa: E402
import plan  # noqa: E402

PY = sys.executable


def full_cfg(c):
    cfg = rig2.default_cfg()
    cfg.update(c)
    return cfg


def run_one(c, log):
    cfg = full_cfg(c)
    rid = rig2.cfg_id(cfg)
    out = os.path.join(HERE, "runs", rid + ".json")
    if os.path.exists(out):
        return rid, "cached"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    try:
        p = subprocess.run([PY, os.path.join(HERE, "rig2.py"), json.dumps(c)], capture_output=True, text=True,
                           timeout=c.get("timeout_s", 7200), env=env, cwd=HERE)
        tail = [ln for ln in p.stdout.splitlines() if ln.startswith("SUMMARY")]
        msg = tail[-1] if tail else ("rc=%d " % p.returncode) + p.stderr[-400:].replace("\n", " | ")
    except subprocess.TimeoutExpired:
        msg = "TIMEOUT"
        json.dump(dict(id=rid, cfg=cfg, status="timeout"), open(out, "w"))
    with open(log, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {time.time()-t0:7.1f}s {c.get('tag','')} {msg}\n")
    return rid, msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--nproc", type=int, default=32)
    a = ap.parse_args()
    G = plan.groups()
    cfgs = []
    for g in a.groups:
        for c in G[g]:
            c = dict(c)
            c.setdefault("tag", g)
            cfgs.append(c)
    # slowest first (rough cost proxy: number of cells)
    cfgs.sort(key=lambda c: -(c.get("cells", [10, 14, 2])[0] * c.get("cells", [10, 14, 2])[1]
                              * c.get("cells", [10, 14, 2])[2] * (3 if c.get("variant") == "quadratic" else 1)))
    log = os.path.join(HERE, "logs", "sweep_" + "_".join(a.groups) + ".log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a") as f:
        f.write(f"=== {time.ctime()} {len(cfgs)} cfgs nproc={a.nproc}\n")
    with ThreadPoolExecutor(a.nproc) as ex:
        list(ex.map(lambda c: run_one(c, log), cfgs))
    with open(log, "a") as f:
        f.write(f"=== done {time.ctime()}\n")
