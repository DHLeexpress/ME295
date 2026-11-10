"""Top-down grasp on the real UR10e with the wrist ZED: the same perception, controllers and safety filter as the
simulated ablation (vision/scripts/grasp_ablation.py).

    python vision/scripts/grasp_real.py                          # DRY RUN (default): look, plan, check; no motion
    python vision/scripts/grasp_real.py --controller mpc --execute     # asks for a typed "yes", then grasps

Assumes (as agreed) the arm already stands in the top-down view configuration (vision/sim/tabletop.py Q_VIEW,
printed below) with an object on the table in view; --goto-view moves there first (movej, path-checked, asks).
The robot's TCP must be the flange (the calibration's assumption; checked). Speeds are scaled by --speed-scale
(default 0.5). Every twist goes through vision.safety (table + workspace barrier, speed caps); Ctrl+C stops the arm.
Before the first --execute: e-stop in hand, clear workspace, gripper type and fingers as in the model (2F-140 with
the blade fingers; another gripper needs its own envelope / width table).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from vision import WristCamera  # noqa: E402
from vision.controllers import CONTROLLERS, run_grasp  # noqa: E402
from vision.controllers.base import rotvec_to_mat, tool_yaw  # noqa: E402
from vision.safety import SafetyConfig, SafetyFilter  # noqa: E402
from vision.sim.tabletop import Q_VIEW  # noqa: E402
from vision.tabletop import TabletopParams, draw_tabletop  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="192.168.0.2")
    ap.add_argument("--controller", choices=sorted(CONTROLLERS), default="mpc")
    ap.add_argument("--execute", action="store_true", help="actually move (asks first); default is a dry run")
    ap.add_argument("--goto-view", action="store_true", help="movej to the view configuration first (asks)")
    ap.add_argument("--speed-scale", type=float, default=0.5)
    ap.add_argument("--force", type=int, default=80, help="Robotiq force register 0..255")
    ap.add_argument("--z-table", type=float, default=0.0065)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "vision" / "grasp_real" / time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    import cv2
    from baseline.embodiment import EndEffector
    from vision.real_robot import AsyncGraspEstimator, RTDERobot
    from vision.stream import CalibratedStream, RTDEPoses
    from vision.zed import ZedCamera
    from shear_agent.robot import SimRobot
    a.out.mkdir(parents=True, exist_ok=True)
    s = a.speed_scale
    safety = SafetyFilter(SafetyConfig(z_table=a.z_table, v_max=0.20 * s, w_max=0.6 * s))
    ee = EndEffector.load("assembly_ur10e_2f140_dohyun_v1.toml")
    wrist = WristCamera.active()
    print(f"view configuration Q_VIEW (deg): {np.round(np.degrees(Q_VIEW), 1).tolist()}")
    poses = RTDEPoses(a.host)
    mm, deg = poses.check_tcp()
    _, Tf, q = poses.latest()
    print(f"robot: q (deg) {np.round(np.degrees(q), 1).tolist()}; controller vs FK {mm:.1f} mm / {deg:.2f} deg")
    if mm > 6:
        sys.exit("the robot's TCP does not look like the flange: set it to zero on the pendant")
    rep = safety.check_q(q)
    print(f"current pose: {rep.message}")
    dq = np.degrees(np.abs((q - Q_VIEW + np.pi) % (2 * np.pi) - np.pi)).max()
    if dq > 3.0 and not a.goto_view:
        print(f"note: the arm is {dq:.0f} deg (largest joint) from the view configuration; perceiving from here")
    rob = None
    try:
        if a.goto_view:
            ok, r, i = safety.check_path(q, Q_VIEW)
            print(f"path to the view configuration: {'safe' if ok else 'BLOCKED at step ' + str(i)}: {r.message}")
            if not ok:
                return
            if not a.execute:
                print("dry run: would movej to the view configuration")
            elif input("movej to the view configuration? [yes/N] ") == "yes":
                rob = RTDERobot(a.host)
                rob.movej(Q_VIEW, acc=0.3, vel=0.2 * s)
        cam = ZedCamera(serial=wrist.serial or None)
        stream = CalibratedStream(cam, poses, wrist)
        f = stream.read()
        from vision.tabletop import plan_topdown_grasp, segment_objects
        objs, zt = segment_objects(f, f.meta["T_base_flange"], TabletopParams(z_table_guess=a.z_table))
        if zt is not None and abs(zt - a.z_table) > 0.01:
            print(f"WARNING: vision puts the table at z = {zt:+.4f}, the safety filter at {a.z_table:+.4f} (check the calibration)")
        if not objs:
            cv2.imwrite(str(a.out / "look.jpg"), f.rgb[..., ::-1])
            sys.exit(f"no object found on the table (table z {zt}); image -> {a.out / 'look.jpg'}")
        g = plan_topdown_grasp(objs[0], a.z_table, prefer_yaw=tool_yaw(f.meta["T_base_flange"][:3, :3]))
        cv2.imwrite(str(a.out / "look.jpg"), draw_tabletop(f, objs, g, zt)[..., ::-1])
        f.save(a.out / "look.npz")
        print(objs[0].summary())
        print(g.summary())
        if not g.ok:
            sys.exit("not graspable")
        # static checks of the plan: grasp, pre-grasp and lift poses (tool points + links via IK on nominal kinematics)
        T_tcp = np.eye(4)
        T_tcp[:3, :] = np.c_[np.eye(3), ee.tcp_pose6(width=g.width)[:3]][:3]
        sim = SimRobot(verbose=False)
        lift = g.T_tcp.copy()
        lift[2, 3] += 0.10
        checks = {}
        for name, T in (("pre-grasp", g.T_pre), ("grasp", g.T_tcp), ("lift", lift)):
            Tfl = T @ np.linalg.inv(T_tcp)
            qq, err = sim.ik(Tfl, q, iters=300)
            r = safety.check_q(qq, T_tcp) if err < 1e-5 else None
            checks[name] = {"ik_err": float(err), "safe": bool(r and r.safe), "msg": r.message if r else "no IK"}
            print(f"  {name:9s}: {'OK ' if r and r.safe else 'NO '} {checks[name]['msg']}")
        (a.out / "plan.json").write_text(json.dumps({"grasp": g.summary(), "checks": checks}, indent=2))
        if not all(c["safe"] for c in checks.values()):
            sys.exit("the plan fails the safety checks: not executing")
        if not a.execute:
            print(f"DRY RUN: would run the {a.controller} controller at speed scale {s}; overlay -> {a.out / 'look.jpg'}")
            return
        if input(f"EXECUTE the {a.controller} grasp on the real robot? Clear workspace, e-stop in hand. [yes/N] ") != "yes":
            return
        from shear_agent.robot.real_ur import RobotiqSocket
        rob = rob or RTDERobot(a.host)
        gripper = RobotiqSocket(a.host, speed=128, force=a.force)
        gripper.activate()        # no-op if already active; otherwise the fingers open/close once
        est = AsyncGraspEstimator(stream, TabletopParams(z_table_guess=a.z_table),
                                  yaw_hint=lambda: tool_yaw(rotvec_to_mat(rob.getl()[3:])), z_table=a.z_table)
        kw = {"scripted": {"v_approach": 0.12 * s, "v_descend": 0.04 * s},
              "pbvs": {"v_max": 0.12 * s, "v_descend": 0.04 * s},
              "mpc": {"v_max": (0.12 * s, 0.12 * s, 0.06 * s)}}[a.controller]
        ctrl = CONTROLLERS[a.controller](ee.register_for_width, force=a.force, **kw)

        def watchdog(obs, cmd):
            if est.stale(0.5) and cmd.phase == "approach":
                raise RuntimeError("camera stream stalled during the approach")
        t_wait = time.time() + 3
        while est.latest is None and time.time() < t_wait:
            time.sleep(0.05)
        out = run_grasp(ctrl, rob, gripper, est.perceive, safety, lambda w: tuple(ee.tcp_pose6(width=w)),
                        rob.flange_pose, max_time=a.max_time, on_step=watchdog, clock=time.monotonic)
        est.close()
        print({k: v for k, v in out.items() if k != "log"})
        (a.out / "result.json").write_text(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=2, default=str))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if rob is not None:
            rob.close()
        poses.close()


if __name__ == "__main__":
    main()
