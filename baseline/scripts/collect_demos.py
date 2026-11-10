"""Collect scripted-expert demonstrations of the pick-up task for fine-tuning a VLA.

    python baseline/scripts/collect_demos.py --n-random 240 --n-nominal 60 --workers 48

Each episode: build a scene variation, run the privileged expert (baseline/sim/expert.py) and record at 10 Hz
both cameras (224 x 224), joints, gripper, TCP pose, wrist F/T, the PDMS skin maps and the action labels. Only
successful episodes are kept (one .npz per episode in --out); failures are listed in collect_log.jsonl.

Seeds: random layouts use scene seeds [0, n_random) (evaluation uses 100000+); nominal-layout demos (the simulator's
own scene, start pose perturbed) use seeds 50000+. Rendering uses EGL, one context per worker, spread over GPUs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _worker_init(gpus):
    ident = mp.current_process()._identity
    wid = ident[0] if ident else 0
    os.environ["MUJOCO_GL"] = "egl"
    if gpus:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpus[wid % len(gpus)])
    sys.path.insert(0, str(ROOT))


def make_variation(kind: str, seed: int):
    import numpy as np
    from baseline.sim import scene
    if kind == "random":
        return scene.sample(np.random.default_rng(seed), split="train")
    rng = np.random.default_rng(seed)
    v = scene.nominal()
    v.split, v.seed = "train_nominal", seed
    v.start_offset = tuple(float(s) for s in np.r_[rng.uniform(-0.03, 0.03, 3), rng.uniform(-0.06, 0.06, 3)])
    return v


def collect_one(job):
    kind, seed, out = job
    import numpy as np
    from baseline.sim.env import EnvConfig, LibrarianPickEnv
    from baseline.sim.expert import Recorder, run_expert
    t0 = time.time()
    try:
        v = make_variation(kind, seed)
        v.seed = seed
        env = LibrarianPickEnv(v, EnvConfig())
        rec = Recorder(env)
        ok = run_expert(env, np.random.default_rng(seed + 10_000), rec)
        rec.detach()
        met = env.metrics()
        ep = rec.episode()
        env.close()
        path = None
        if ok:
            path = Path(out) / f"{kind}_{seed:06d}.npz"
            np.savez_compressed(path, **ep, variation=v.to_json(), instruction=met["instruction"],
                                metrics=json.dumps(met))
        return {"kind": kind, "seed": seed, "ok": ok, "steps": len(ep["actions"]), "wall": time.time() - t0,
                "path": str(path) if path else None, "metrics": met}
    except Exception as e:  # keep the pool alive; record the failure
        return {"kind": kind, "seed": seed, "ok": False, "error": repr(e), "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-random", type=int, default=240)
    ap.add_argument("--n-nominal", type=int, default=60)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--gpus", default="0,1,2,3", help="EGL devices for rendering ('' = default device)")
    ap.add_argument("--out", type=Path, default=ROOT / "baseline" / "data" / "demos")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [("random", s, str(args.out)) for s in range(args.n_random)]
    jobs += [("nominal", 50_000 + s, str(args.out)) for s in range(args.n_nominal)]
    jobs = [j for j in jobs if not (args.out / f"{j[0]}_{j[1]:06d}.npz").exists()]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    print(f"{len(jobs)} episodes on {args.workers} workers -> {args.out}", flush=True)
    t0, n_ok = time.time(), 0
    log = open(args.out / "collect_log.jsonl", "a")
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init, initargs=(gpus,)) as pool:
        for i, r in enumerate(pool.imap_unordered(collect_one, jobs), 1):
            n_ok += bool(r["ok"])
            log.write(json.dumps(r) + "\n")
            log.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {n_ok} kept, {time.time() - t0:.0f} s", flush=True)
    print(f"kept {n_ok}/{len(jobs)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
