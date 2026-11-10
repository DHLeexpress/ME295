"""How well does the wrist ZED see the shelf? Shelf perception vs simulation ground truth on the VLA baseline's
held-out test cases (same seeds as baseline/scripts/evaluate.py), in parallel.

    python vision/scripts/eval_sim_perception.py                                  # nominal + random + narrow, 3 depth models
    python vision/scripts/eval_sim_perception.py --suites random -n 50 --depth exact

Per case: build the scene with the calibrated ZED on the flange, move to the look pose (aimed at the row centre: the
target is not known yet), grab one frame, estimate the shelf, pick "the {colour} book" of the instruction, and
compare with the designated book (centre, thickness, both gaps, spine plane, top). Depth models: exact (the
renderer), noise (0.3 px disparity noise), fatten (noise + 2 px foreground fattening, a pessimistic stereo matcher).
Writes <out>/cases.jsonl, <out>/summary.md and a contact sheet of the first cases.
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
DEPTH_MODELS = {"exact": {}, "noise": {"disparity_noise_px": 0.3}, "fatten": {"disparity_noise_px": 0.3,
                                                                           "edge_fatten_px": 2}}


def _init(gpus):
    ident = mp.current_process()._identity
    wid = ident[0] if ident else 0
    os.environ["MUJOCO_GL"] = "egl"
    if gpus:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpus[wid % len(gpus)])
    for p in (ROOT, ROOT / "src", ROOT / "scripts_sim", ROOT / "baseline" / "scripts"):
        sys.path.insert(0, str(p))


def ground_truth(m, d, info):
    """Designated book and its actual gaps after settling (sim state)."""
    import numpy as np
    from baseline.sim import scene
    b = m.body(info.designated).id
    x, half = float(d.xpos[b][0]), info.half
    left = right = None
    for n in scene.UPPER:
        if n == info.designated or n not in [m.body(i).name for i in range(m.nbody)]:
            continue
        j = m.body(n).id
        hx = float(m.geom_size[m.geom(n).id][0])
        xj = float(d.xpos[j][0])
        if xj < x:
            g = (x - half[0]) - (xj + hx)
            left = g if left is None else min(left, g)
        else:
            g = (xj - hx) - (x + half[0])
            right = g if right is None else min(right, g)
    return {"x": x, "thickness": float(2 * half[0]), "gap_left": left, "gap_right": right,
            "spine_y": float(d.xpos[b][1] - half[1]), "top_z": float(d.xpos[b][2] + half[2])}


def run_case(job):
    suite, i, depth, scale, keep_img = job
    import numpy as np
    import evaluate
    import run_librarian as rl
    from baseline.sim import scene
    from shear_agent.robot import SimRobot, mat_to_pose, pose_to_mat
    from vision import WristCamera
    from vision.look import look_pose
    from vision.shelf import PerceptionError, estimate_shelf
    from vision.sim_zed import SimZedCamera, add_wrist_zed
    t0 = time.time()
    v, _ = evaluate.make_case(suite, i)
    wrist = WristCamera.active(warn=False)
    m, info = scene.build(v, edit=lambda s: add_wrist_zed(s, wrist))
    rob = SimRobot(model=m, verbose=False)
    rob.set_tcp(rl.TCP)
    rob.reset(rl.start_q(rob))
    T = look_pose(wrist, (0.0, scene.SPINE_Y, 0.53))
    rob.movel(list(mat_to_pose(T @ pose_to_mat(rl.TCP))), acc=0.5, vel=0.25)
    rob.sleep(0.3)
    gt = ground_truth(m, rob.d, info)
    cam = SimZedCamera(m, scale=scale, seed=v.seed, **DEPTH_MODELS[depth])
    frame = cam.grab(rob.d)
    cam.close()
    rec = {"suite": suite, "index": i, "seed": v.seed, "depth": depth, "colour": v.color, "gt": gt,
           "distractor": v.distractor is not None}
    try:
        t1 = time.time()
        est = estimate_shelf(frame)
        rec["t_estimate"] = time.time() - t1
        tgt = est.select(v.color)
        rec["n_books"] = len(est.books)
        rec["spine_err_mm"] = 1e3 * (est.spine_y - gt["spine_y"])
        rec["yaw_deg"] = est.yaw_deg
        if tgt is None:
            rec["status"] = "no candidate"
        else:
            rec.update({
                "status": "ok" if abs(tgt.center[0] - gt["x"]) < 0.5 * gt["thickness"] else "wrong book",
                "x_err_mm": 1e3 * (tgt.center[0] - gt["x"]),
                "thick_err_mm": 1e3 * (tgt.thickness - gt["thickness"]),
                "gapL_err_mm": 1e3 * (tgt.gap_left - gt["gap_left"]) if gt["gap_left"] is not None else None,
                "gapR_err_mm": 1e3 * (tgt.gap_right - gt["gap_right"]) if gt["gap_right"] is not None else None,
                "top_err_mm": 1e3 * (tgt.z_top - gt["top_z"]), "top_seen": tgt.top_seen,
                "est_colour": tgt.colour})
        if keep_img:
            from vision.shelf import draw
            img = draw(frame, est, tgt)
            np.save(Path(keep_img), img[::2, ::2])
    except PerceptionError as e:
        rec["status"] = f"error: {e}"
    rec["wall"] = time.time() - t0
    return rec


def summarise(recs, path):
    import numpy as np
    lines = ["# Wrist ZED shelf perception vs simulation ground truth", "",
             "Look pose aimed at the row centre (target unknown), one frame, HD720 intrinsics of the calibrated ZED Mini.",
             "Errors in mm (estimate - truth): median [5th, 95th percentile]. found = the instruction's book picked.",
             "", "| suite | depth | n | found | wrong book | no candidate | centre x | thickness | gap left | gap right |"
                 " spine plane | top |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def stat(xs):
        xs = np.array([x for x in xs if x is not None], float)
        if not len(xs):
            return "–"
        return f"{np.median(xs):+.1f} [{np.percentile(xs, 5):+.1f}, {np.percentile(xs, 95):+.1f}]"
    keys = sorted({(r["suite"], r["depth"]) for r in recs}, key=lambda k: (k[0], list(DEPTH_MODELS).index(k[1])))
    for su, de in keys:
        rs = [r for r in recs if r["suite"] == su and r["depth"] == de]
        ok = [r for r in rs if r["status"] == "ok"]
        lines.append(f"| {su} | {de} | {len(rs)} | {len(ok)} | {sum(r['status'] == 'wrong book' for r in rs)} | "
                     f"{sum(r['status'] == 'no candidate' or r['status'].startswith('error') for r in rs)} | "
                     + " | ".join(stat([r.get(k) for r in ok]) for k in
                                  ("x_err_mm", "thick_err_mm", "gapL_err_mm", "gapR_err_mm"))
                     + f" | {stat([r.get('spine_err_mm') for r in rs])} | "
                       f"{stat([r.get('top_err_mm') for r in ok if r.get('top_seen')])} |")
    bad = [r for r in recs if r["status"] != "ok"]
    if bad:
        lines += ["", "Misses:", ""] + [f"- {r['suite']} #{r['index']} ({r['depth']}): {r['status']}"
                                        f"{'' if 'est_colour' not in r else ', picked ' + r['est_colour']}"
                                        f" (asked {r['colour']}{', distractor present' if r['distractor'] else ''})"
                                        for r in bad]
    Path(path).write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suites", default="nominal,random,narrow")
    ap.add_argument("-n", type=int, default=30, help="cases per suite")
    ap.add_argument("--depth", default=",".join(DEPTH_MODELS), help="depth models: " + ",".join(DEPTH_MODELS))
    ap.add_argument("--scale", type=float, default=1.0, help="render scale of HD720")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--sheet", type=int, default=8, help="cases in the contact sheet (from the first suite / depth)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "vision" / "sim_perception")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    gpus = [g for g in a.gpus.split(",") if g]
    depths = a.depth.split(",")
    jobs = []
    for su in a.suites.split(","):
        for de in depths:
            for i in range(a.n):
                keep = str(a.out / f"_sheet_{su}_{i:03d}.npy") if (de == depths[0] and i < a.sheet) else None
                jobs.append((su, i, de, a.scale, keep))
    t0 = time.time()
    recs = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(gpus,)) as pool:
        for r in pool.imap_unordered(run_case, jobs):
            recs.append(r)
            if len(recs) % 20 == 0:
                print(f"  {len(recs)}/{len(jobs)} cases, {time.time() - t0:.0f} s", flush=True)
    recs.sort(key=lambda r: (r["suite"], r["depth"], r["index"]))
    with open(a.out / "cases.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(summarise(recs, a.out / "summary.md"))
    import numpy as np
    tiles = sorted(a.out.glob("_sheet_*.npy"))
    if tiles:
        import cv2
        imgs = [np.load(t) for t in tiles]
        while len(imgs) % 4:
            imgs.append(np.zeros_like(imgs[0]))
        sheet = np.vstack([np.hstack(imgs[k:k + 4]) for k in range(0, len(imgs), 4)])
        cv2.imwrite(str(a.out / "contact_sheet.jpg"), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        for t in tiles:
            t.unlink()
    print(f"{len(recs)} cases in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
