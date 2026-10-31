"""Sweep driver for bench-dynamic: builds the run matrix (groups below), runs each config of rig.py in its own
process (<= NPROC at a time, single-threaded numpy), stores runs/<id>.json (+ .npz time series) and the MuJoCo text
warnings printed by the process.

Usage:  python sweep.py GROUP [GROUP ...] [--nproc 90] [--dry]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rig import cfg_id, default_cfg  # noqa: E402

PY = "/home/dohyun/miniforge3/bin/python"
IP = [(5, 7), (10, 14), (20, 28), (30, 42), (40, 56)]       # in-plane cells: 10.2, 5.1, 2.55, 1.7, 1.275 mm
M30 = [(nx, ny, nz) for (nx, ny) in IP for nz in range(1, 7)]
ALL7 = ["C1", "C1f", "C2", "C4", "C5", "C6", "C6f"]
BASE = dict(hold_s=0.2)


def mk(**kw):
    c = dict(BASE)
    c.update(kw)
    c["cells"] = list(c["cells"])
    return c


def group(name):
    out = []
    if name == "main":          # (i) full flex, F1-corrected, robot options (CG, 100 iterations)
        out = [mk(variant="full_corr", cells=g, case=c) for g in M30 for c in ALL7]
    elif name == "conv":        # (i) with a 10x iteration budget (convergence reference)
        out = [mk(variant="full_corr", cells=g, case=c, iterations=1000) for g in M30 for c in ALL7]
    elif name == "raw":         # (ii) uncorrected full flex, a few meshes
        out = [mk(variant="full_raw", cells=g, case=c, iterations=it)
               for g in [(5, 7, 1), (10, 14, 2), (20, 28, 3), (40, 56, 4)] for c in ALL7 for it in (100, 1000)]
    elif name == "tri":         # (iii) trilinear, cellcount = cells
        out = [mk(variant="trilinear", cells=g, case=c) for g in M30 for c in ALL7]
    elif name == "quad":        # (iv) quadratic
        out = [mk(variant="quadratic", cells=g, case=c) for g in M30 for c in ALL7]
    elif name == "split":       # 50-contact cap workaround: indenter split into s x s geoms
        out = [mk(variant=v, cells=g, case=c, split=s) for v in ("trilinear", "quadratic")
               for g in [(5, 7, 1), (10, 14, 2), (20, 28, 3)] for c in ["C1", "C1f", "C2", "C6", "C6f"]
               for s in (2, 3, 5)]
    elif name == "contact":     # contact settings vs penetration
        sets = [dict(contact="default"), dict(contact="radius"), dict(contact="margin"), dict(contact="stiffest"),
                dict(contact="stiff", mass_scale=10.0), dict(contact="stiff", mass_scale=100.0)]
        out = [mk(variant="full_corr", cells=g, case=c, iterations=it, **s)
               for s in sets for g in [(10, 14, 2), (20, 28, 3)] for c in ALL7 for it in (100, 1000)]
    elif name == "newton":      # Newton once for comparison (servo indenter = a hub body; mocap without it)
        out = [mk(variant="full_corr", cells=g, case=c, solver="Newton")
               for g in [(5, 7, 1), (10, 14, 2), (20, 28, 3), (20, 28, 4)] for c in ALL7]
        out += [mk(variant="full_corr", cells=g, case=c, solver="Newton", driver="mocap")
                for g in [(10, 14, 2), (20, 28, 3)] for c in ["C1", "C1f", "C4", "C5", "C6f"]]
    elif name == "nu49":        # nu = 0.49
        out = [mk(variant="full_corr", cells=g, case=c, nu=0.49, iterations=it)
               for g in M30 for c in ["C1", "C1f", "C4", "C6f"] for it in (100, 1000)]
        out += [mk(variant=v, cells=g, case=c, nu=0.49, split=s)
                for v in ("trilinear", "quadratic") for g in M30 for (c, s) in [("C1f", 3), ("C4", 1)]]
    elif name == "iter":        # CG iteration budget
        out = [mk(variant="full_corr", cells=g, case=c, iterations=it)
               for g in [(10, 14, 2), (20, 28, 3), (30, 42, 4), (40, 56, 4)] for c in ["C1", "C1f", "C2"]
               for it in (30, 300, 3000, 10000)]
    elif name == "driver":      # kinematic (mocap) indenter instead of the servo
        out = [mk(variant="full_corr", cells=g, case=c, iterations=it, driver="mocap")
               for g in [(10, 14, 2), (20, 28, 3)] for c in ALL7 for it in (100, 1000)]
    elif name == "margin2":     # corrected margin setting (flex margin only), see rig.CONTACT["margin"] note
        out = [mk(variant="full_corr", cells=g, case=c, iterations=it, contact="margin_flex")
               for g in [(10, 14, 2), (20, 28, 3)] for c in ALL7 for it in (100, 1000)]
    elif name == "timing":      # cost re-measurement at low machine load (run with --nproc 4); rep makes new ids
        out = [mk(variant="full_corr", cells=g, case=c, rep=1)
               for g in [(10, 14, 2), (20, 28, 3), (30, 42, 4), (40, 56, 3)] for c in ("C1", "C1f")]
        out += [mk(variant="trilinear", cells=(20, 28, 3), case="C1f", rep=1),
                mk(variant="quadratic", cells=(10, 14, 2), case="C1f", rep=1)]
    elif name == "smoke":
        out = [mk(variant="full_corr", cells=(5, 7, 1), case=c, hold_s=0.05) for c in ALL7]
    else:
        raise ValueError(name)
    return out


def est_cost(c):
    nx, ny, nz = c["cells"]
    if c.get("variant") == "quadratic":
        n = (2 * nx + 1) * (2 * ny + 1) * (2 * nz)
    else:
        n = (nx + 1) * (ny + 1) * nz
    return n * (c.get("iterations", 100) ** 0.5) * (5 if c.get("solver") == "Newton" else 1)


def run_one(c, timeout):
    full = default_cfg()
    full.update(c)
    rid = cfg_id(full)
    path = os.path.join(HERE, "runs", rid + ".json")
    if os.path.exists(path):
        return rid, "cached", 0.0
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MUJOCO_GL="egl")
    t0 = time.time()
    try:
        r = subprocess.run([PY, os.path.join(HERE, "rig.py"), json.dumps(c)], capture_output=True, text=True,
                           timeout=timeout, cwd=os.path.join(HERE, "logs"), env=env)
        out, err, code = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err, code = f"timeout after {timeout} s", -9
    wall = time.time() - t0
    warn = sorted(set(l.strip() for l in (out + "\n" + err).splitlines() if "WARNING" in l or "Warning" in l))
    if os.path.exists(path):
        rec = json.load(open(path))
    else:
        rec = dict(id=rid, cfg=full, status="crashed" if code != -9 else "timeout", returncode=code,
                   stderr_tail=err[-3000:], stdout_tail=out[-2000:])
    rec["text_warnings"] = warn
    rec["process_wall_s"] = wall
    with open(path, "w") as f:
        json.dump(rec, f)
    return rid, rec["status"], wall


def branch_key(c):
    full = default_cfg()
    full.update(c)
    return json.dumps({k: full[k] for k in sorted(full) if k not in ("cells", "tag")}, sort_keys=True)


def dominated(c, slow):
    """True if a mesh with <= cells (componentwise) in the same branch was too slow (> max_step_s per step)."""
    for cells in slow.get(branch_key(c), []):
        if all(a <= b for a, b in zip(cells, c["cells"])):
            return cells
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--nproc", type=int, default=90)
    ap.add_argument("--timeout", type=float, default=4 * 3600)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    cfgs = []
    seen = set()
    for g in a.groups:
        for c in group(g):
            full = default_cfg()
            full.update(c)
            rid = cfg_id(full)
            if rid in seen:
                continue
            seen.add(rid)
            c["tag"] = g
            cfgs.append(c)
    cfgs.sort(key=est_cost)              # small first: too-slow information prunes the larger meshes
    print(f"{len(cfgs)} configs", flush=True)
    if a.dry:
        return
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = open(os.path.join(HERE, "logs", f"sweep_{'_'.join(a.groups)}.log"), "a")
    # too-slow records already on disk
    slow = {}
    for c in cfgs:
        full = default_cfg()
        full.update(c)
        p = os.path.join(HERE, "runs", cfg_id(full) + ".json")
        if os.path.exists(p) and json.load(open(p)).get("status") == "too_slow":
            slow.setdefault(branch_key(c), []).append(c["cells"])
    t0 = time.time()
    done = 0
    queue = list(cfgs)
    running = {}

    def report(c, rid, st, wall):
        nonlocal done
        done += 1
        line = (f"[{done}/{len(cfgs)} {time.time()-t0:7.0f}s] {rid} {st} {wall:7.1f}s "
                f"{c.get('variant')} {c['cells']} {c['case']} {c.get('contact', 'stiff')} it={c.get('iterations', 100)} "
                f"{c.get('solver', 'CG')} split={c.get('split', 1)} nu={c.get('nu', 0.45)} "
                f"m={c.get('mass_scale', 1)} drv={c.get('driver', 'servo')}")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    with ProcessPoolExecutor(a.nproc) as ex:
        while queue or running:
            while queue and len(running) < a.nproc:
                c = queue.pop(0)
                dom = dominated(c, slow)
                if dom is not None:
                    full = default_cfg()
                    full.update(c)
                    rid = cfg_id(full)
                    path = os.path.join(HERE, "runs", rid + ".json")
                    if not os.path.exists(path):
                        json.dump(dict(id=rid, cfg=full, status="skipped_branch_too_slow", dominated_by_cells=dom),
                                  open(path, "w"))
                    slow.setdefault(branch_key(c), []).append(c["cells"])
                    report(c, rid, "skipped_branch_too_slow", 0.0)
                    continue
                running[ex.submit(run_one, c, a.timeout)] = c
            if not running:
                break
            fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fu in fin:
                c = running.pop(fu)
                try:
                    rid, st, wall = fu.result()
                except Exception as e:  # noqa
                    rid, st, wall = "?", f"driver_error {e}", 0
                if st == "too_slow" or (st == "cached" and False):
                    slow.setdefault(branch_key(c), []).append(c["cells"])
                report(c, rid, st, wall)


if __name__ == "__main__":
    main()
