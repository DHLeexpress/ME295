"""Librarian demo: approach the shelf horizontally, slide the blades beside one book, pull it out,
lift it and stand it on the desk. Written against the python-urx API, so the same task runs in
MuJoCo (default) or on the real UR10e (--backend urx).

    python scripts_sim/run_librarian.py                      # headless; urx-style log + CSV + plot
    python scripts_sim/run_librarian.py --viewer             # interactive viewer with live HUD
                                                             #   (macOS: mjpython scripts_sim/run_librarian.py --viewer)
    python scripts_sim/run_librarian.py --video out.mp4      # three-panel video: skin | MuJoCo | camera
    python scripts_sim/run_librarian.py --controller mppi    # plug-in slot (raises until implemented)
    python scripts_sim/run_librarian.py --backend urx --host 192.168.1.10   # real robot (see real_ur.py)

Phases: gross motion with urx calls (movej/movel, a guarded movel, gripper_action), then the
contact-rich part (close -> pull -> lift -> hold) in a controller loop (speedl + Robotiq
registers) whose rule-based placeholder is shear_agent.control.ShearMarginGrasp.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_sim"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from shear_agent.control import CONTROLLERS, run  # noqa: E402

SCENE = ROOT / "assembly" / "scene_bookshelf.xml"
# shelf layout (assembly/scene_bookshelf.xml): book spines at y = 0.82, target book 40 mm thick
BOOK_CENTER = np.array([0.0, 0.905, 0.535])
BOOK_THICKNESS = 0.040
SPINE_Y = 0.82
TCP = (0.0, 0.0, 0.292, 0.0, 0.0, 0.0)   # between the PDMS faces, on the flange axis
HORIZONTAL = (-np.pi / 2, 0.0, 0.0)       # tool z -> +y (into the shelf), blade width vertical
INSERT_DEPTH = 0.020                       # TCP behind the spines: blades 59 mm in; the clip posts on the
                                           #   blades' back (24-35 mm before the TCP) stay outside the shelf
CLEARANCE = 0.003                          # air between each PDMS face and the book while inserting
                                           #   (leaves ~3.8 mm between each blade's back and the neighbour)
PULL, LIFT = 0.25, 0.08
PLACE = np.array([0.45, 0.60, 0.118])      # book stands on the desk (centre 0.115 above it)


def pose(p, r=HORIZONTAL):
    return [float(v) for v in (*p, *r)]


def gap_to_register(model: mujoco.MjModel, gap: float) -> int:
    """Robotiq position register for a PDMS-to-PDMS gap (gripper kinematics from the MuJoCo model)."""
    import sim_gripper as sg
    d = mujoco.MjData(model)
    left, right = model.geom("left_pdms").id, model.geom("right_pdms").id
    lo, hi = 0.0, 0.7
    for _ in range(40):
        q = 0.5 * (lo + hi)
        sg.set_fingers_kinematic(model, d, q)
        lo, hi = (q, hi) if mujoco.mj_geomDistance(model, d, left, right, 0.5, None) > gap else (lo, q)
    return int(round(q / 0.7 * 255))


def task(rob, gripper, controller, model, rec=None, skins=None, say=print):
    """The librarian task, urx calls only (+ the controller loop)."""
    def phase(name):
        if rec is not None:
            rec.phase = name
        say(f"--- {name}")

    open_reg = gap_to_register(model, BOOK_THICKNESS + 2 * CLEARANCE)
    pre = BOOK_CENTER.copy()
    pre[1] = SPINE_Y - 0.10
    insert = BOOK_CENTER.copy()
    insert[1] = SPINE_Y + INSERT_DEPTH

    phase("approach")
    gripper.gripper_action(open_reg)
    rob.movel(pose(pre), acc=0.5, vel=0.15)

    phase("insert")  # guarded move, python-urx style: stop if the blades hit something
    rob.movel(pose(insert), acc=0.2, vel=0.03, wait=False)
    rob.sleep(0.1)  # python-urx: give the controller time to start the program before polling
    while rob.is_program_running():
        if rob.get_force() > 15.0:
            rob.stopl()
            say("!!! unexpected contact while inserting, stopped")
            return False
        rob.sleep(0.02)
    rob.sleep(0.3)

    phase("grasp & pull (controller)")
    last = {"phase": None}

    def on_step(obs, cmd):
        if cmd.phase != last["phase"]:
            last["phase"] = cmd.phase
            if rec is not None:
                rec.phase = f"controller: {cmd.phase}"
                if cmd.phase == "pull":
                    rec.mark_grasp()
    run(controller, rob, gripper, skins=skins, max_time=25.0, on_step=on_step)

    phase("place")
    rob.movel(pose((PLACE[0], PLACE[1], 0.40)), acc=0.4, vel=0.12)
    rob.movel(pose(PLACE + (0, 0, 0.03)), acc=0.2, vel=0.05)
    f0 = rob.get_tcp_force()[2]
    rob.movel(pose(PLACE - (0, 0, 0.01)), acc=0.1, vel=0.015, wait=False)  # guarded: stop on touch-down
    rob.sleep(0.1)
    while rob.is_program_running():
        if rob.get_tcp_force()[2] - f0 > 3.0:  # desk pushes the book up
            rob.stopl(acc=0.5)
            say("book touched the desk")
            break
        rob.sleep(0.01)
    rob.sleep(0.2)
    gripper.gripper_action(open_reg)
    if rec is not None:
        rec.release()
    phase("retreat")
    rob.translate((0, -0.12, 0), acc=0.3, vel=0.08)
    rob.translate((0, 0, 0.25), acc=0.4, vel=0.12)
    phase("done")
    return True


def start_q(rob) -> np.ndarray:
    """Joint configuration for the start pose (elbow up, tool horizontal toward the shelf)."""
    from shear_agent.robot import pose_to_mat
    T = pose_to_mat(pose((0.0, 0.66, 0.66))) @ np.linalg.inv(pose_to_mat(TCP))
    q, err = rob.ik(T, [-1.97, -1.8, 2.0, -0.2, 1.17, 3.14], iters=200)
    assert err < 1e-6, err
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["sim", "urx"], default="sim")
    ap.add_argument("--host", help="robot IP for --backend urx")
    ap.add_argument("--controller", choices=sorted(CONTROLLERS), default="margin")
    ap.add_argument("--viewer", action="store_true", help="interactive MuJoCo viewer with live HUD")
    ap.add_argument("--realtime", type=float, default=1.0, help="viewer speed (1 = real time)")
    ap.add_argument("--video", type=Path, help="write the three-panel MP4 here")
    ap.add_argument("--renderer", choices=["auto", "filament", "classic"], default="auto")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "sim_runs" / "librarian")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    controller = CONTROLLERS[args.controller](**({"pull_dist": PULL, "lift": LIFT} if args.controller == "margin" else {}))

    if args.backend == "urx":
        from shear_agent.robot.real_ur import connect
        if not args.host:
            ap.error("--backend urx needs --host")
        if input(f"Move the REAL robot at {args.host}? Clear the workspace, hand on the e-stop. [yes/N] ") != "yes":
            return
        model = mujoco.MjModel.from_xml_path(str(SCENE))  # gripper geometry for the width -> register map
        rob, gripper = connect(args.host, tcp=TCP)
        rob.movej(list(start_q_from_model(model)), acc=0.5, vel=0.3)
        task(rob, gripper, controller, model)
        rob.close()
        return

    from librarian_media import Compositor, Recorder, VideoWriter, ViewerHUD
    from shear_agent.robot import PDMSSkin, SimRobot, SimRobotiqGripper, SkinSensor

    rob = SimRobot(SCENE, verbose=not args.quiet)
    rob.reset(start_q(rob))
    gripper = SimRobotiqGripper(rob, speed=128, force=0)
    rob.set_tcp(TCP)
    skins = {s: SkinSensor(PDMSSkin(rob.m, s), rate_hz=20.0) for s in ("left", "right")}
    for sensor in skins.values():
        rob.callbacks.append(sensor.tick)
    others = [rob.m.body(b).name for b in range(rob.m.nbody)
              if rob.m.body(b).name.startswith("book_") and rob.m.body(b).name != "book_target"]
    rec = Recorder(rob, gripper, skins, others=others)
    hud = writer = comp = None
    if args.viewer:
        hud = ViewerHUD(rec, realtime=args.realtime)
    if args.video:
        from shear_agent.robot.render import SceneRenderer
        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = SceneRenderer(rob.m, args.renderer)

        def orbit(t):  # slow arc from the robot's left-front, looking down at the desk and shelf front
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (0.12, 0.62, 0.36)
            cam.distance, cam.elevation = 1.55, -27.0
            cam.azimuth = 22.0 + 0.55 * t
            return cam

        follow = {"p": None}

        def close_up(t):  # follows the TCP, low-pass filtered so it glides
            p = np.asarray(rob.getl()[:3]) + (0.0, 0.025, -0.01)
            follow["p"] = p if follow["p"] is None else 0.92 * follow["p"] + 0.08 * p
            cam = mujoco.MjvCamera()
            cam.lookat[:] = follow["p"]
            cam.distance, cam.azimuth, cam.elevation = 0.42, 38.0, -30.0
            return cam
        comp = Compositor(renderer, orbit, close_up,
                          title="Librarian demo  |  UR10e + Robotiq 2F-140 + PDMS blade fingers  |  MuJoCo")
        writer = VideoWriter(args.video, fps=args.fps)
        t_frame = {"next": 0.0, "n": 0, "wall": time.time()}

        def grab(r):
            if r.time + 1e-9 < t_frame["next"]:
                return
            t_frame["next"] += 1.0 / args.fps
            writer.write(comp.frame(rec))
            t_frame["n"] += 1
            if t_frame["n"] % 150 == 0:
                print(f"   video: {t_frame['n']} frames, sim t = {r.time:.1f} s, "
                      f"{time.time() - t_frame['wall']:.0f} s wall", flush=True)
        rob.callbacks.append(grab)

    ok = False
    try:
        rob.sleep(0.5)
        ok = task(rob, gripper, controller, rob.m, rec=rec, skins=skins, say=lambda s: rob._say(s))
        rob.sleep(1.0)
    except KeyboardInterrupt:
        print("stopped (viewer closed or Ctrl+C)")
    finally:
        if writer:
            writer.close()
            comp.r.close()  # free the GL contexts while EGL is still up
            print(f"video -> {args.video}")
        if hud:
            hud.close()
    args.out.mkdir(parents=True, exist_ok=True)
    rec.log.save_csv(args.out / "log.csv")
    summary(rec, args.out / "summary.png")
    print(f"task {'succeeded' if ok else 'aborted'}; book now at {np.round(rec.d.xpos[rec.book], 3)}; "
          f"log -> {args.out}")


def start_q_from_model(model):
    from shear_agent.robot import SimRobot
    return start_q(SimRobot(model=model, verbose=False))


def summary(rec, path):
    """Static plot of the run: skin forces, margin, arm F/T, slip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = rec.log
    t = L.column("t")
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for side, c in (("left", "#1f77b4"), ("right", "#d62728")):
        ax[0].plot(t, L.column(f"{side}_normal"), c=c, label=f"{side} normal")
        ax[1].plot(t, L.column(f"{side}_shear"), c=c, label=f"{side} |shear|")
        ax[2].plot(t, np.clip(L.column(f"{side}_margin"), -0.2, 1.0), c=c, label=f"q {side} (patch)")
    ax[2].plot(t, L.column("gripper_force") / 255.0, c="#e0a000", label="force register / 255")
    ax[3].plot(t, -L.column("fz"), label="-Fz arm F/T (N)")
    ax[3].plot(t, -L.column("fy"), label="-Fy arm F/T (N)")
    ax[3].plot(t, L.column("slip_mm"), label="book slip (mm)")
    ax[3].plot(t, L.column("neighbours_mm"), label="neighbour books moved (mm)")
    for a, ylab in zip(ax, ["N", "N", "Q", "N / mm"]):
        a.set_ylabel(ylab)
        a.legend(fontsize=8, ncol=4, loc="upper left")
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("time (s)")
    fig.suptitle("Librarian run: what the skin and the arm feel")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
