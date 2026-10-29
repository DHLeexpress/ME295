"""Librarian demo: take one book out of shelf A by approaching horizontally and shelve it in the free slot of
shelf B. Written against the python-urx API, so the same task runs in MuJoCo (default) or on the real UR10e
(--backend urx).

    python scripts_sim/run_librarian.py                      # headless; urx-style log + CSV + plot
    python scripts_sim/run_librarian.py --viewer             # interactive viewer with live HUD
                                                             #   (macOS: mjpython scripts_sim/run_librarian.py --viewer)
    python scripts_sim/run_librarian.py --video out.mp4      # three-panel video: skin | MuJoCo | camera
    python scripts_sim/run_librarian.py --controller fixed --grip-force 128   # the no-skin baseline
    python scripts_sim/run_librarian.py --controller mppi    # plug-in slot (raises until implemented)
    python scripts_sim/run_librarian.py --backend urx --host 192.168.1.10   # real robot (see real_ur.py)

Phases (urx calls; the contact-rich part runs in a controller loop):
  approach   movel to 10 cm in front of shelf A's spines, gripper opened to the book + 2 x 3 mm
  insert     guarded movel: the blades slide 20 mm past the spines beside the book (stop on > 15 N)
  grasp      controller loop at 50 Hz (speedl + Robotiq registers): close, pull the book out 21 cm, lift 8 cm
             (shear_agent.control.ShearMarginGrasp, or FixedForceGrasp without the skin)
  carry      movel to 22 cm in front of shelf B's slot, turning the tool 45 deg on the way
  shelve     guarded movel into the slot (stop on an unexpected 10 N), guarded descent onto the board (stop when
             the board pushes back 3 N), open to the book + 2 x 3 mm
  retreat    movel back out along the shelf axis and up
If the controller aborts (full force and the skin still says the book is about to slip), the book is pushed back
into shelf A instead.
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
from shear_agent.robot import mat_to_rotvec, rotvec_to_mat  # noqa: E402

SCENE = ROOT / "assembly" / "scene_bookshelf.xml"
# shelf A (source, assembly/scene_bookshelf.xml): book spines at y = 0.82, target book 40 x 170 x 230 mm
BOOK = "book_target"
BOOK_CENTER = np.array([0.0, 0.905, 0.535])
BOOK_THICKNESS = 0.040
BOOK_HALF_HEIGHT = 0.115
SPINE_Y = 0.82
TCP = (0.0, 0.0, 0.292, 0.0, 0.0, 0.0)   # between the PDMS faces, on the flange axis
HORIZONTAL = (-np.pi / 2, 0.0, 0.0)       # tool z -> +y (into shelf A), blade width vertical
INSERT_DEPTH = 0.020                       # TCP behind the spines: blades 59 mm in; the clip posts on the
                                           #   blades' back (24-35 mm before the TCP) stay outside the shelf
CLEARANCE = 0.003                          # air between each PDMS face and the book while inserting
                                           #   (leaves ~3.8 mm between each blade's back and the neighbour)
PULL, LIFT = 0.21, 0.08                    # pulled 21 cm the book clears the shelf front by 2 cm; further, the
                                           #   wrist nears the base axis (shoulder singularity)
# shelf B (destination): turned 45 deg toward the robot; 70 mm slot in its upper tier
B_FRONT = np.array([0.612, 0.612, 0.0])   # front edge centre, desk level
B_HEADING = np.radians(-45.0)             # into-shelf axis = Rz(B_HEADING) (0, 1, 0)
B_SPINE = 0.02                             # spines 20 mm behind the front edge
B_BOARD = 0.33                             # top of the middle board
SET_DOWN_CLEARANCE = 0.020                 # book bottom above the board while sliding in (a 2.4 kg book droops ~10 mm)


def pose(p, r=HORIZONTAL):
    return [float(v) for v in (*p, *r)]


def facing(heading: float):
    """Tool rotation vector for a horizontal approach along Rz(heading) (0, 1, 0), blade width vertical."""
    c, s = np.cos(heading), np.sin(heading)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return mat_to_rotvec(Rz @ rotvec_to_mat(np.array(HORIZONTAL)))


def shelf_b():
    """Shelf B's into-shelf axis, the tool rotation vector facing it, and the TCP where the book is shelved."""
    axis = np.array([-np.sin(B_HEADING), np.cos(B_HEADING), 0.0])
    slot = B_FRONT + axis * (B_SPINE + INSERT_DEPTH) + np.array([0.0, 0.0, B_BOARD + BOOK_HALF_HEIGHT])
    return axis, facing(B_HEADING), slot


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


def guarded_movel(rob, target, acc, vel, limit, axis=None) -> bool:
    """python-urx style guarded move: movel(wait=False), then poll the TCP wrench and stopl() once the force
    has changed by more than `limit` N since the start (axis given: only that component, increases only).
    Returns True if it stopped on contact."""
    f0 = np.asarray(rob.get_tcp_force()[:3])
    rob.movel(target, acc=acc, vel=vel, wait=False)
    rob.sleep(0.1)  # python-urx: give the controller time to start the program before polling
    while rob.is_program_running():
        df = np.asarray(rob.get_tcp_force()[:3]) - f0
        if (df[axis] if axis is not None else np.linalg.norm(df)) > limit:
            rob.stopl(acc=0.5)
            return True
        rob.sleep(0.01)
    return False


def task(rob, gripper, controller, model, rec=None, skins=None, say=print):
    """The librarian task, urx calls only (+ the controller loop). Returns True if the book was shelved."""
    def phase(name):
        if rec is not None:
            rec.phase = name
        say(f"--- {name}")

    open_reg = gap_to_register(model, BOOK_THICKNESS + 2 * CLEARANCE)
    pre = BOOK_CENTER.copy()
    pre[1] = SPINE_Y - 0.10
    insert = BOOK_CENTER.copy()
    insert[1] = SPINE_Y + INSERT_DEPTH

    phase("approach shelf A")
    gripper.gripper_action(open_reg)
    rob.movel(pose(pre), acc=0.5, vel=0.15)

    phase("insert")  # guarded move, python-urx style: stop if the blades hit something
    rob.movel(pose(insert), acc=0.2, vel=0.03, wait=False)
    rob.sleep(0.1)
    while rob.is_program_running():
        if rob.get_force() > 15.0:
            rob.stopl()
            say("!!! unexpected contact while inserting, stopped")
            return False
        rob.sleep(0.02)
    rob.sleep(0.3)

    phase("grasp, pull, lift (controller)")
    last = {"phase": None}

    def on_step(obs, cmd):
        if cmd.phase != last["phase"]:
            last["phase"] = cmd.phase
            if rec is not None:
                rec.phase = f"controller: {cmd.phase}"
                if cmd.phase == "pull":
                    rec.mark_grasp()
    history = run(controller, rob, gripper, skins=skins, max_time=25.0, on_step=on_step)
    if history and history[-1].phase == "abort":
        phase("abort: put the book back")
        say("!!! full grip force and the skin still says the book is about to slip: putting it back")
        p = np.asarray(rob.getl()[:3])
        rob.movel(pose((p[0], p[1], insert[2])), acc=0.2, vel=0.03)
        rob.movel(pose(insert), acc=0.2, vel=0.03)
        gripper.gripper_action(open_reg)
        if rec is not None:
            rec.release()
        rob.movel(pose(pre), acc=0.3, vel=0.08)
        return False

    axis, rot_b, slot = shelf_b()
    up = np.array([0.0, 0.0, SET_DOWN_CLEARANCE])
    phase("carry to shelf B")  # via a waypoint right of the robot that keeps the wrist off the base axis
    rob.movel(pose((0.30, 0.60, 0.60), facing(np.radians(-25.0))), acc=0.3, vel=0.12)
    rob.movel(pose(slot - 0.22 * axis + up, rot_b), acc=0.3, vel=0.12)

    phase("shelve")
    if guarded_movel(rob, pose(slot + up, rot_b), acc=0.2, vel=0.03, limit=10.0):
        say("!!! unexpected contact while sliding into shelf B, stopped")
        return False
    rob.sleep(0.2)
    if guarded_movel(rob, pose(slot - 0.006 * np.array([0, 0, 1]), rot_b), acc=0.1, vel=0.01, limit=3.0, axis=2):
        say("book touched the board")
    rob.sleep(0.2)
    gripper.gripper_action(open_reg)
    if rec is not None:
        rec.release()
    phase("retreat")
    p = np.asarray(rob.getl()[:3])
    rob.movel(pose(p - 0.20 * axis, rot_b), acc=0.3, vel=0.08)
    rob.translate((0, 0, 0.15), acc=0.4, vel=0.12)
    phase("done")
    return True


def start_q(rob) -> np.ndarray:
    """Joint configuration for the start pose (elbow up, tool horizontal toward shelf A)."""
    from shear_agent.robot import pose_to_mat
    T = pose_to_mat(pose((0.0, 0.66, 0.66))) @ np.linalg.inv(pose_to_mat(TCP))
    q, err = rob.ik(T, [-1.97, -1.8, 2.0, -0.2, 1.17, 3.14], iters=200)
    assert err < 1e-6, err
    return q


def shelved(m, d, book: str = BOOK) -> dict:
    """Where the book ended up relative to shelf B's slot (simulation ground truth)."""
    axis, _, slot = shelf_b()
    centre = slot + axis * (BOOK_CENTER[1] - SPINE_Y - INSERT_DEPTH)   # book centre 65 mm past the TCP
    centre[2] = B_BOARD + BOOK_HALF_HEIGHT
    b = m.body(book).id
    R = d.xmat[b].reshape(3, 3)
    err = d.xpos[b] - centre
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    yaw = np.degrees(np.arctan2(R[1, 1], R[0, 1]) - np.arctan2(axis[1], axis[0]))
    yaw = (yaw + 180) % 360 - 180
    ok = bool(np.linalg.norm(err[:2]) < 0.015 and abs(err[2]) < 0.008 and tilt < 5.0 and abs(yaw) < 5.0)
    return {"shelved": ok, "err_mm": 1e3 * err, "tilt_deg": float(tilt), "yaw_deg": float(yaw)}


def make_sim(controller="margin", grip_force=128, mu=None, mu_assumed=None, book_mass=None, verbose=True,
             skin_rate=20.0):
    """SimRobot + gripper + skins (+ Recorder) for the librarian scene. mu: true PDMS friction (model);
    mu_assumed: what the skin's margin assumes (default: the model's); book_mass: kg (inertia scaled)."""
    from librarian_media import Recorder
    from shear_agent.robot import PDMSSkin, SimRobot, SimRobotiqGripper, SkinSensor
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    if book_mass is not None:
        b = m.body(BOOK).id
        m.body_inertia[b] *= book_mass / m.body_mass[b]
        m.body_mass[b] = book_mass
    if mu is not None:
        for g in range(m.ngeom):
            if "_pdms_" in (m.geom(g).name or ""):
                m.geom_friction[g, 0] = mu
    rob = SimRobot(model=m, verbose=verbose)
    mujoco.mj_setConst(rob.m, rob.d)  # masses changed: recompute the constraint solver's inertia estimates
    rob.reset(start_q(rob))
    gripper = SimRobotiqGripper(rob, speed=128, force=0)
    rob.set_tcp(TCP)
    skins = {s: SkinSensor(PDMSSkin(rob.m, s, mu=mu_assumed), rate_hz=skin_rate) for s in ("left", "right")}
    for sensor in skins.values():
        rob.callbacks.append(sensor.tick)
    others = [rob.m.body(b).name for b in range(rob.m.nbody)
              if rob.m.body(b).name.startswith("book") and rob.m.body(b).name != BOOK]
    rec = Recorder(rob, gripper, skins, book=BOOK, others=others)
    kw = {"pull_dist": PULL, "lift": LIFT}
    if controller == "fixed":
        kw["force"] = grip_force
    ctrl = CONTROLLERS[controller](**(kw if controller in ("margin", "fixed") else {}))
    return rob, gripper, skins, rec, ctrl


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["sim", "urx"], default="sim")
    ap.add_argument("--host", help="robot IP for --backend urx")
    ap.add_argument("--controller", choices=sorted(CONTROLLERS), default="margin")
    ap.add_argument("--grip-force", type=int, default=128, help="force register 0..255 for --controller fixed")
    ap.add_argument("--viewer", action="store_true", help="interactive MuJoCo viewer with live HUD")
    ap.add_argument("--realtime", type=float, default=1.0, help="viewer speed (1 = real time)")
    ap.add_argument("--video", type=Path, help="write the three-panel MP4 here")
    ap.add_argument("--renderer", choices=["auto", "filament", "classic"], default="auto")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "sim_runs" / "librarian")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.backend == "urx":
        from shear_agent.robot.real_ur import connect
        if not args.host:
            ap.error("--backend urx needs --host")
        name = args.controller
        if name == "margin":  # no skin on the robot yet: the margin controller would hold at its 10 N start
            print(f"no skin on the real robot yet: using the fixed grip (--grip-force {args.grip_force})")
            name = "fixed"
        kw = {"pull_dist": PULL, "lift": LIFT}
        if name == "fixed":
            kw["force"] = args.grip_force
        controller = CONTROLLERS[name](**(kw if name in ("margin", "fixed") else {}))
        if input(f"Move the REAL robot at {args.host}? Clear the workspace, hand on the e-stop. [yes/N] ") != "yes":
            return
        model = mujoco.MjModel.from_xml_path(str(SCENE))  # gripper geometry for the width -> register map
        rob, gripper = connect(args.host, tcp=TCP)
        rob.movej(list(start_q_from_model(model)), acc=0.5, vel=0.3)
        task(rob, gripper, controller, model)
        rob.close()
        return

    from librarian_media import Compositor, VideoWriter, ViewerHUD

    rob, gripper, skins, rec, controller = make_sim(args.controller, args.grip_force, verbose=not args.quiet)
    hud = writer = comp = None
    if args.viewer:
        hud = ViewerHUD(rec, realtime=args.realtime)
    if args.video:
        from librarian_media import close_up_camera, orbit_camera
        from shear_agent.robot.render import SceneRenderer
        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = SceneRenderer(rob.m, args.renderer)
        comp = Compositor(renderer, orbit_camera, close_up_camera(rob),
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
    res = shelved(rob.m, rob.d)
    print(f"task {'completed' if ok else 'aborted'}; book {'shelved in shelf B' if res['shelved'] else 'NOT shelved'} "
          f"(centre error {np.round(res['err_mm'], 1)} mm, tilt {res['tilt_deg']:.1f} deg, yaw {res['yaw_deg']:.1f} deg); "
          f"log -> {args.out}")


def start_q_from_model(model):
    from shear_agent.robot import SimRobot
    return start_q(SimRobot(model=model, verbose=False))


def summary(rec, path):
    """Static plot of the run: skin forces and deformation, margin, arm F/T, slip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = rec.log
    t = L.column("t")
    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for side, c in (("left", "#1f77b4"), ("right", "#d62728")):
        ax[0].plot(t, L.column(f"{side}_normal"), c=c, label=f"{side} normal")
        ax[0].plot(t, L.column(f"{side}_shear"), c=c, ls="--", label=f"{side} |shear|")
        ax[1].plot(t, L.column(f"{side}_comp_um"), c=c, label=f"{side} max compression")
        ax[1].plot(t, L.column(f"{side}_sdisp_um"), c=c, ls="--", label=f"{side} max shear displacement")
        ax[2].plot(t, np.clip(L.column(f"{side}_margin"), -0.2, 1.0), c=c, label=f"q {side} (patch)")
    ax[2].plot(t, L.column("gripper_force") / 255.0, c="#e0a000", label="force register / 255")
    ax[3].plot(t, -L.column("fz"), label="-Fz arm F/T (N)")
    ax[3].plot(t, -L.column("fy"), label="-Fy arm F/T (N)")
    ax[4].plot(t, L.column("slip_mm"), label="book slip in the grasp (mm)")
    ax[4].plot(t, L.column("slip_deg"), label="book rotation in the grasp (deg)")
    ax[4].plot(t, L.column("neighbours_mm"), label="other books moved (mm)")
    for a, ylab in zip(ax, ["N", "um", "q", "N", "mm / deg"]):
        a.set_ylabel(ylab)
        a.legend(fontsize=8, ncol=4, loc="upper left")
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("time (s)")
    fig.suptitle("Librarian run: contact force and deformation of each PDMS layer, margin, arm F/T, slip")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
