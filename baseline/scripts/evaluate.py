"""Evaluate a policy on the pick-up task over fixed test suites (held-out seeds), in parallel.

    # privileged scripted expert (upper bound; same fixed grip force as the VLA)
    python baseline/scripts/evaluate.py --policy expert --suites nominal,random,narrow,heavy_lowmu
    # pi0.5 fine-tuned on the demos (server: python -m baseline.openpi_ext.cli serve pi05_librarian_lora ...)
    python baseline/scripts/evaluate.py --policy pi05_ft --port 8000 --suites nominal,random,narrow,heavy_lowmu
    # zero-shot pi0.5-DROID (server: ... serve pi05_droid --dir gs://openpi-assets/checkpoints/pi05_droid)
    python baseline/scripts/evaluate.py --policy pi05_droid --port 8001 --suites nominal,random -n 20

Suites (scene seeds never used for training data, which uses 0..1000 and 50000+):
    nominal      the simulator's own scene (dark blue 40 mm book, 10 mm gaps, 0.8 kg), start pose perturbed
    random       randomised layouts from the training distribution (gaps 8-12 mm, 0.4-1.2 kg, colours, 25% distractor)
    narrow       as random, gaps 6.5-7.5 mm (narrower than any demo)
    heavy_lowmu  as random, 1.6-2.4 kg books and PDMS-cover friction 0.5 instead of 1.0
Writes one JSON line per episode to <out>/<policy>/<suite>.jsonl (+ MP4s of the first --videos episodes per suite).
Media re-runs for figures (high-resolution close / top / wide views, sim/media.py; keep them out of results/eval):
    python baseline/scripts/evaluate.py --policy pi05_ft --port 8000 --suites random --indices 5,27 --media \
        --out baseline/data/media_runs
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
SUITES = {"nominal": 100_000, "random": 200_000, "narrow": 300_000, "heavy_lowmu": 400_000}


def _worker_init(gpus):
    ident = mp.current_process()._identity
    wid = ident[0] if ident else 0
    os.environ["MUJOCO_GL"] = "egl"
    if gpus:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpus[wid % len(gpus)])
    sys.path.insert(0, str(ROOT))


def make_case(suite: str, i: int):
    """(Variation, EnvConfig overrides) of test case i of a suite."""
    import numpy as np
    from baseline.sim import scene
    seed = SUITES[suite] + i
    rng = np.random.default_rng(seed)
    kw = {}
    if suite == "nominal":
        v = scene.nominal()
        v.start_offset = tuple(float(s) for s in np.r_[rng.uniform(-0.03, 0.03, 3), rng.uniform(-0.06, 0.06, 3)])
    elif suite == "random":
        v = scene.sample(rng, split=suite)
    elif suite == "narrow":
        v = scene.sample(rng, split=suite, gap_range=(0.0065, 0.0075))
    elif suite == "heavy_lowmu":
        v = scene.sample(rng, split=suite, mass_range=(1.6, 2.4))
        kw["pdms_mu"] = 0.5
    else:
        raise ValueError(suite)
    v.seed, v.split = seed, suite
    return v, kw


def run_one(job):
    policy, suite, i, host, port, out, video, name, stop, media = job
    import numpy as np
    from baseline.sim.env import EnvConfig, LibrarianPickEnv
    t0 = time.time()
    rec = {"policy": policy, "suite": suite, "index": i}
    env = client = recm = None
    try:
        v, kw = make_case(suite, i)
        hz = 15.0 if policy == "pi05_droid" else 10.0
        env = LibrarianPickEnv(v, EnvConfig(control_hz=hz, keep_frames=video, protective_stop=stop, **kw))
        if media:
            from baseline.sim.media import MediaRecorder
            recm = MediaRecorder(env, Path(out) / name / "media" / f"{suite}_{i:03d}", fps=hz)
            rec["media"] = {k: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                            for k, p in recm.paths.items()}
        if policy == "expert":
            from baseline.sim.expert import run_expert
            tick = {"next": env.rob.time}

            def monitor(rob):   # metrics (and video frames) at the policy rate while the expert moves
                if rob.time + 1e-9 >= tick["next"]:
                    tick["next"] += 1.0 / hz
                    env.update_metrics()
                    if video:
                        env.observe()
                    if recm is not None:
                        recm.capture()
            env.rob.callbacks.append(monitor)
            run_expert(env, np.random.default_rng(v.seed + 10_000))
        else:
            from baseline.policies.adapters import DroidRetargetPolicy, JointChunkPolicy
            from baseline.policies.remote import RemotePolicy
            client = RemotePolicy(host, port)
            pol = DroidRetargetPolicy(client, env.ee) if policy == "pi05_droid" else JointChunkPolicy(client)
            pol.reset()
            obs = env.observe()
            if recm is not None:
                recm.capture()
            while not env.done():
                obs = env.step(pol.act(obs, env))
                if recm is not None:
                    recm.capture()
            rec["policy_calls"] = client.n_calls
            rec["policy_latency_s"] = client.latency / max(1, client.n_calls)
            if policy == "pi05_droid":
                rec["ik_fail_steps"] = pol.ik_fail
                rec["virtual_ik_err_max"] = pol.max_ik_err
        rec.update(env.metrics())
        rec["variation"] = json.loads(v.to_json())
        if env.trace:   # t, TCP xyz, book xyz, gripper, |F| at the policy rate, for failure analysis
            tdir = Path(out) / name / "traces"
            tdir.mkdir(parents=True, exist_ok=True)
            np.save(tdir / f"{suite}_{i:03d}.npy", np.asarray(env.trace, np.float32))
        if video and env.frames:
            import imageio
            path = Path(out) / name / "videos" / f"{suite}_{i:03d}_{'ok' if rec['success'] else 'fail'}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(path, env.frames, fps=int(hz), quality=7, macro_block_size=8)
            rec["video"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    except Exception as e:
        import traceback
        rec["error"] = repr(e)
        rec["traceback"] = traceback.format_exc()[-2000:]
    finally:
        if client is not None:
            client.close()
        if recm is not None:
            recm.close()
        if env is not None:
            env.close()
    rec["wall_s"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", choices=["expert", "pi05_ft", "pi05_droid"], required=True)
    ap.add_argument("--suites", default="nominal,random,narrow,heavy_lowmu")
    ap.add_argument("-n", type=int, default=50, help="episodes per suite")
    ap.add_argument("--n-per-suite", default="", help="override per suite, e.g. narrow=30,heavy_lowmu=30")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--videos", type=int, default=4, help="record the first N episodes of each suite")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--tag", default="", help="results sub-folder suffix (e.g. a checkpoint step)")
    ap.add_argument("--protective-stop", type=float, default=150.0,
                    help="wrist force (N) that ends an episode as a UR protective stop (1e9 = off, diagnostic only)")
    ap.add_argument("--out", type=Path, default=ROOT / "baseline" / "results" / "eval")
    ap.add_argument("--indices", default="", help="run only these test cases of each suite, e.g. 5,27")
    ap.add_argument("--media", action="store_true",
                    help="also record high-resolution close / top / wide views (sim/media.py) of every episode")
    args = ap.parse_args()
    n_override = dict(kv.split("=") for kv in args.n_per_suite.split(",") if kv)
    name = args.policy + (f"_{args.tag}" if args.tag else "")
    (args.out / name).mkdir(parents=True, exist_ok=True)
    jobs = []
    for suite in args.suites.split(","):
        n = int(n_override.get(suite, args.n))
        done = set()
        f = args.out / name / f"{suite}.jsonl"
        if f.exists():
            done = {json.loads(line)["index"] for line in open(f) if "error" not in json.loads(line)}
        idx = [int(k) for k in args.indices.split(",") if k.strip()] or range(n)
        jobs += [(args.policy, suite, i, args.host, args.port, str(args.out), i < args.videos, name,
                  args.protective_stop, args.media)
                 for i in idx if i not in done]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    print(f"{name}: {len(jobs)} episodes on {args.workers} workers", flush=True)
    t0 = time.time()
    stats = {}
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init, initargs=(gpus,)) as pool:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            with open(args.out / name / f"{r['suite']}.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            s = stats.setdefault(r["suite"], [0, 0, 0])
            s[0] += 1
            s[1] += bool(r.get("success"))
            s[2] += "error" in r
            if k % 5 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} ({time.time() - t0:.0f} s): " +
                      ", ".join(f"{su} {v[1]}/{v[0]}" + (f" ({v[2]} err)" if v[2] else "") for su, v in stats.items()),
                      flush=True)


if __name__ == "__main__":
    main()
