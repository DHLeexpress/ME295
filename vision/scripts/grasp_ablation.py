"""Grasp ablation in the simulated tabletop cell: same perception (wrist ZED, vision.tabletop), same gripper
sequence and safety filter; the controllers differ in how they approach (vision/controllers).

    python vision/scripts/grasp_ablation.py                         # 3 controllers x 4 conditions x 30 objects
    python vision/scripts/grasp_ablation.py -n 5 --conditions nominal --video results/vision/grasp_demo.mp4

Conditions:
    nominal  stereo depth noise (0.3 px disparity), exact hand-eye calibration
    calib    the perception's hand-eye calibration is off by 0.75 deg and 2 mm (random axis; the size of the change
             between the 2026-10-09 and 2026-10-13 calibrations)
    calib2   off by 2 deg and 4 mm (a bumped camera that was not recalibrated)
    moved    the object is pushed 4 cm in a random direction 1 s after the start (someone bumps it)
    fatten   stereo edge fattening (2 px) on top of the noise: objects look wider

Start: the top-down view configuration (vision/sim/tabletop.py Q_VIEW), object somewhere in the workspace.
Success: both pads on the object and the object lifted > 5 cm after the hold.
Writes <out>/episodes.jsonl and <out>/summary.md.
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
CONDITIONS = ("nominal", "calib", "calib2", "moved", "fatten")


def _init(gpus):
    ident = mp.current_process()._identity
    wid = ident[0] if ident else 0
    os.environ["MUJOCO_GL"] = "egl"
    if gpus:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpus[wid % len(gpus)])
    for p in (ROOT, ROOT / "src"):
        sys.path.insert(0, str(p))


def perturbed(wrist, rng, deg=0.75, mm=2.0):
    import numpy as np
    from dataclasses import replace
    from vision.controllers.base import rotvec_to_mat
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    T = wrist.T_flange_cam.copy()
    T[:3, :3] = T[:3, :3] @ rotvec_to_mat(np.radians(deg) * axis)
    T[:3, 3] += 1e-3 * mm * d
    return replace(wrist, T_flange_cam=T)


def episode(controller: str, condition: str, i: int, video: str = None, scale: float = 0.5):
    import numpy as np
    from baseline.embodiment import EndEffector
    from vision.controllers import CONTROLLERS, run_grasp
    from vision.controllers.base import rotvec_to_mat, tool_yaw
    from vision.safety import SafetyConfig, SafetyFilter
    from vision.sim.tabletop import WORKSPACE_CENTER, TabletopEnv, sample_object
    from vision.tabletop import plan_topdown_grasp, segment_objects
    from vision.wrist import WristCamera
    t_wall = time.time()
    seed = 7000 + i
    rng = np.random.default_rng(seed)
    obj = sample_object(rng)
    belief = None
    if condition in ("calib", "calib2"):
        belief = perturbed(WristCamera.active(warn=False), rng, *((0.75, 2.0) if condition == "calib" else (2.0, 4.0)))
    env = TabletopEnv(obj, depth="fatten" if condition == "fatten" else "noise", scale=scale, seed=seed,
                      wrist_belief=belief)
    ee = EndEffector.load("assembly_ur10e_2f140_dohyun_v1.toml")
    safety = SafetyFilter(SafetyConfig(z_table=0.0))
    rob, gripper = env.rob, env.gripper
    gripper.set_force(80)
    state = {"prev": None}

    def perceive():
        f = env.grab()
        objs, zt = segment_objects(f, rob.flange_pose())
        if not objs:
            return None
        prev = state["prev"]
        o = min(objs, key=lambda o: np.linalg.norm(o.center[:2] - prev)) if prev is not None else objs[0]
        state["prev"] = o.center[:2].copy()
        yaw = tool_yaw(rotvec_to_mat(rob.getl()[3:]))
        return plan_topdown_grasp(o, safety.cfg.z_table, prefer_yaw=yaw, stamp=rob.time)
    events = []
    if condition == "moved":
        ang = rng.uniform(-np.pi, np.pi)
        d = 0.04 * np.array([np.cos(ang), np.sin(ang)])
        c = np.asarray(WORKSPACE_CENTER)
        p = env.rob.d.xpos[env.body][:2]
        if np.any(np.abs(p + d - c) > np.array([0.11, 0.08])):   # keep it in the workspace
            d = -d
        events.append((1.0, lambda: env.move_object(d)))
    writer = None
    on_step = None
    if video:
        from vision.camera import depth_colormap  # noqa: F401
        import imageio_ffmpeg
        import mujoco
        r = mujoco.Renderer(env.m, 360, 640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [obj.xy[0], obj.xy[1], 0.08]
        cam.distance, cam.azimuth, cam.elevation = 1.1, 200.0, -25.0
        writer = imageio_ffmpeg.write_frames(video, (1280, 360), fps=25, macro_block_size=1, quality=7)
        writer.send(None)
        nxt = {"t": 0.0}

        def on_step(obs, cmd):
            if rob.time < nxt["t"]:
                return
            nxt["t"] = rob.time + 0.04
            r.update_scene(rob.d, camera=cam)
            a = r.render()
            r.update_scene(rob.d, camera="zed_cam")
            b = r.render()
            import cv2
            for im, txt in ((a, f"{controller} | {condition} | {cmd.phase} | t={rob.time:5.2f}s"), (b, "wrist ZED (sim)")):
                cv2.putText(im, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            writer.send(np.ascontiguousarray(np.concatenate([a, b], 1)))
    ctrl = CONTROLLERS[controller](ee.register_for_width)
    try:
        out = run_grasp(ctrl, rob, gripper, perceive, safety, lambda w: tuple(ee.tcp_pose6(width=w)),
                        rob.flange_pose, events=events, on_step=on_step, say=lambda s: None, max_time=30.0)
        for _ in range(25):          # let it settle in the hold
            rob.sleep(0.02)
            if on_step:
                on_step(None, type("C", (), {"phase": "end"})())
    finally:
        if writer:
            writer.close()
            r.close()
    m = env.metrics()
    rec = {"controller": controller, "condition": condition, "index": i, "seed": seed, "object": obj.to_dict(),
           "status": out["status"], **m, "time": out.get("time"), "perceptions": out.get("perceptions"),
           "filter_updates": out.get("filter_updates"), "safety_interventions": out.get("safety_interventions"),
           "mpc_fails": getattr(ctrl, "fails", None), "wall": time.time() - t_wall}
    if out.get("goal") is not None:
        rec["goal_err_mm"] = 1e3 * float(np.linalg.norm(np.asarray(out["goal"][:2]) - env.p0[:2]))
    env.close()
    return rec


def _job(a):
    try:
        return episode(*a)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"controller": a[0], "condition": a[1], "index": a[2], "status": f"error: {e!r}",
                "trace": traceback.format_exc()[-1500:], "success": False}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarise(recs, path, controllers, conditions):
    import numpy as np
    lines = ["# Top-down grasp ablation (simulated tabletop, wrist ZED perception)", "",
             "Success = both pads on the object and lifted > 5 cm. Same perception, gripper sequence and safety filter "
             "for every controller. 95 % Wilson intervals.", "",
             "| controller | " + " | ".join(conditions) + " |", "|---|" + "---|" * len(conditions)]
    for c in controllers:
        cells = []
        for k in conditions:
            rs = [r for r in recs if r["controller"] == c and r["condition"] == k]
            s = sum(bool(r.get("success")) for r in rs)
            lo, hi = wilson(s, len(rs))
            cells.append(f"{s}/{len(rs)} [{100 * lo:.0f}, {100 * hi:.0f} %]")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "Medians over all episodes of a controller (time: start to the end of the hold):", "",
              "| controller | time s | goal error mm | pushed before closing mm | gripper-body contact N | safety interventions | perceptions |",
              "|---|---|---|---|---|---|---|"]
    med = lambda xs: f"{np.median([x for x in xs if x is not None]):.1f}" if any(x is not None for x in xs) else "–"
    for c in controllers:
        rs = [r for r in recs if r["controller"] == c]
        lines.append(f"| {c} | {med([r.get('time') for r in rs])} | {med([r.get('goal_err_mm') for r in rs])} | "
                     f"{med([r.get('pushed_before_close_mm') for r in rs])} | {med([r.get('hard_contact_max_N') for r in rs])} | "
                     f"{med([r.get('safety_interventions') for r in rs])} | {med([r.get('perceptions') for r in rs])} |")
    fails = [r for r in recs if not r.get("success")]
    if fails:
        lines += ["", "Failures:", ""]
        for r in fails:
            o = r.get("object", {})
            lines.append(f"- {r['controller']} / {r['condition']} #{r['index']}: {r['status']}; "
                         f"{o.get('shape', '?')} {1e3 * 2 * min(o.get('size', [0])[:2]) if o else 0:.0f} mm, "
                         f"lift {1e3 * r.get('lift_m', 0):.0f} mm, pushed {r.get('pushed_before_close_mm', 0):.0f} mm, "
                         f"goal error {r.get('goal_err_mm', float('nan')):.1f} mm")
    Path(path).write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--controllers", default="scripted,pbvs,mpc")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--video", help="also record episode 0 of each controller x condition (MP4 path prefix)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "vision" / "grasp_ablation")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ctrls, conds = a.controllers.split(","), a.conditions.split(",")
    jobs = [(c, k, i, (f"{a.video}_{c}_{k}.mp4" if a.video and i == 0 else None)) for c in ctrls for k in conds
            for i in range(a.n)]
    t0 = time.time()
    recs = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=([g for g in a.gpus.split(",") if g],)) as pool:
        for r in pool.imap_unordered(_job, jobs):
            recs.append(r)
            if len(recs) % 20 == 0:
                print(f"  {len(recs)}/{len(jobs)} episodes, {time.time() - t0:.0f} s", flush=True)
    recs.sort(key=lambda r: (r["controller"], r["condition"], r["index"]))
    with open(a.out / "episodes.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(summarise(recs, a.out / "summary.md", ctrls, conds))
    print(f"{len(recs)} episodes in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
