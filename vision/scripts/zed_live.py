"""Live calibrated wrist-ZED view: what the camera sees, placed in the robot base frame in real time (read-only:
never commands the robot).

    python vision/scripts/zed_live.py                       # ZED + robot pose over RTDE (192.168.0.2)
    python vision/scripts/zed_live.py --grasp               # + objects on the table and the planned top-down grasp
    python vision/scripts/zed_live.py --no-robot            # camera only (no base frame)
    python vision/scripts/zed_live.py --headless 60         # no window: 60 frames, saves the last panel + stats
    python vision/scripts/zed_live.py --sim --headless 30   # same pipeline on the simulated tabletop

Panels: RGB with the robot base axes, a 10 cm grid on the table plane and the calibration board's recorded outline
projected through the live calibration (they should sit on the real table / board); depth; a top-down height map
of the point cloud in the base frame (workspace box, 2 mm / px) with the TCP. If the ChArUco board is in view, its
live pose is compared with where the calibration session saw it (mm / deg): a direct check of the calibration.
Keys: q quit, s save frame (npz + png), g toggle the grasp preview.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from vision import WristCamera, depth_colormap  # noqa: E402
from vision.charuco import board_outline, detect_board  # noqa: E402
from vision.tabletop import TabletopParams, draw_tabletop, plan_topdown_grasp, segment_objects  # noqa: E402

TABLE_Z = 0.0065


def topdown_map(frame, params: TabletopParams, z_table: float, res: float = 0.002, tcp=None, objs=(), grasp=None):
    import cv2
    x0, x1 = params.x_range
    y0, y1 = params.y_range
    W, H = int((x1 - x0) / res), int((y1 - y0) / res)
    img = np.zeros((H, W, 3), np.uint8)
    pts, _ = frame.points_base(stride=2)
    sel = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1) & (pts[:, 2] < 0.6)
    p = pts[sel]
    if len(p):
        u = ((p[:, 0] - x0) / res).astype(int)
        v = ((y1 - p[:, 1]) / res).astype(int)          # +y up in the image
        order = np.argsort(p[:, 2])                       # higher points drawn last
        h = np.clip((p[order, 2] - z_table) / 0.25, 0, 1)
        col = depth_colormap(1.0 - h[None, :], 0.0, 1.0)[0]
        img[v[order].clip(0, H - 1), u[order].clip(0, W - 1)] = col
    for x in np.arange(np.ceil(x0 * 10) / 10, x1, 0.1):
        cv2.line(img, (int((x - x0) / res), 0), (int((x - x0) / res), H), (60, 60, 60), 1)
    for y in np.arange(np.ceil(y0 * 10) / 10, y1, 0.1):
        cv2.line(img, (0, int((y1 - y) / res)), (W, int((y1 - y) / res)), (60, 60, 60), 1)
    toimg = lambda x, y: (int((x - x0) / res), int((y1 - y) / res))
    for o in objs:
        uu = np.array([np.cos(o.yaw), np.sin(o.yaw)])
        vv = np.array([-uu[1], uu[0]])
        cs = [toimg(*(o.center[:2] + a * 0.5 * o.length * uu + b * 0.5 * o.width * vv)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cv2.polylines(img, [np.array(cs)], True, (0, 255, 0), 1)
    if grasp is not None:
        x = grasp.T_tcp[:3, 0]
        cv2.line(img, toimg(*(grasp.position[:2] - 0.5 * grasp.width * x[:2])),
                 toimg(*(grasp.position[:2] + 0.5 * grasp.width * x[:2])), (255, 0, 255), 2)
    if tcp is not None:
        cv2.drawMarker(img, toimg(tcp[0], tcp[1]), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.putText(img, f"base frame top view x[{x0},{x1}] y[{y0},{y1}] m, colour = height", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def overlay(frame, calib_board, z_table, objs=(), grasp=None, board=None):
    import cv2
    img = draw_tabletop(frame, list(objs), grasp, z_table) if frame.T_base_cam is not None else frame.rgb.copy()
    if frame.T_base_cam is None:
        return img
    Ti = np.linalg.inv(frame.T_base_cam)

    def px(p):
        pc = Ti[:3, :3] @ p + Ti[:3, 3]
        return None if pc[2] < 0.05 else tuple(int(v) for v in np.round(frame.intr.project(pc[None])[0]))
    o = px(np.zeros(3))
    for a, c in zip(np.eye(3) * 0.15, ((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        e = px(a)
        if o and e:
            cv2.line(img, o, e, c, 3, cv2.LINE_AA)
    if calib_board is not None:
        cs = [px(c) for c in board_outline(calib_board)]
        if all(cs):
            cv2.polylines(img, [np.array(cs)], True, (0, 255, 255), 2, cv2.LINE_AA)
    if board is not None:
        cs = [px(c) for c in board_outline(board)]
        if all(cs):
            cv2.polylines(img, [np.array(cs)], True, (255, 128, 0), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="192.168.0.2")
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--tcp", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0],
                    help="TCP set on the robot (the calibration assumes the flange)")
    ap.add_argument("--sim", action="store_true", help="simulated tabletop instead of the real camera / robot")
    ap.add_argument("--depth-mode", default="NEURAL")
    ap.add_argument("--grasp", action="store_true", help="segment objects and preview the top-down grasp")
    ap.add_argument("--headless", type=int, default=0, help="no window: run N frames and save")
    ap.add_argument("--z-table", type=float, default=TABLE_Z)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "vision" / "live")
    a = ap.parse_args()
    import cv2
    a.out.mkdir(parents=True, exist_ok=True)
    wrist = WristCamera.active()
    calib_board = None
    try:
        zdir = Path(wrist.source).parent if Path(wrist.source).is_absolute() else ROOT / "vision" / wrist.source
        calib_board = np.load(zdir / "hand_eye_calibration_ur10e_result.npz")["T_base_board"]
    except Exception:  # noqa: BLE001
        pass
    from vision.stream import CalibratedStream, SimPoses
    if a.sim:
        from vision.sim.tabletop import TabletopEnv, sample_object
        env = TabletopEnv(sample_object(np.random.default_rng(3)), scale=1.0)
        cam = type("C", (), {"grab": lambda self: env.cam.grab(env.rob.d), "close": lambda self: env.close()})()
        poses = SimPoses(env.rob)
        a.z_table = 0.0
    else:
        from vision.zed import ZedCamera
        cam = ZedCamera(serial=wrist.serial or None, depth_mode=a.depth_mode)
        print(cam.describe())
        d = np.abs(np.array([cam.intr.fx - wrist.intr.fx, cam.intr.fy - wrist.intr.fy, cam.intr.cx - wrist.intr.cx,
                             cam.intr.cy - wrist.intr.cy]))
        if d.max() > 1.0:
            print(f"note: live intrinsics differ from vision/zed_mini_wrist.json by up to {d.max():.1f} px "
                  f"(zed_check.py --snapshot updates it); the live ones are used")
        poses = None
        if not a.no_robot:
            from vision.stream import RTDEPoses
            poses = RTDEPoses(a.host, tcp=a.tcp)
            mm, deg = poses.check_tcp()
            print(f"robot pose over RTDE: controller vs nominal FK {mm:.1f} mm / {deg:.2f} deg "
                  f"({'TCP = flange as assumed' if mm < 6 else 'CHECK the TCP set on the robot (--tcp)'})")
    stream = CalibratedStream(cam, poses, wrist) if poses is not None else None
    show_grasp = a.grasp
    stats = []
    n = 0
    try:
        while True:
            t0 = time.time()
            f = stream.read() if stream else cam.grab()
            objs, zt, grasp, board, berr = [], a.z_table, None, None, None
            if f.T_base_cam is not None:
                Tcb, npts, _ = detect_board(f.rgb, f.intr.K)
                if Tcb is not None:
                    board = f.T_base_cam @ Tcb
                    if calib_board is not None:
                        berr = (1e3 * float(np.linalg.norm(board[:3, 3] - calib_board[:3, 3])),
                                float(np.degrees(np.arccos(np.clip((np.trace(board[:3, :3].T @ calib_board[:3, :3]) - 1) / 2, -1, 1)))))
                if show_grasp:
                    objs, zt_est = segment_objects(f, f.meta.get("T_base_flange"), TabletopParams(z_table_guess=a.z_table))
                    zt = zt_est if zt_est is not None else a.z_table
                    if objs:
                        grasp = plan_topdown_grasp(objs[0])
            left = overlay(f, calib_board, zt, objs, grasp, board)
            mid = depth_colormap(f.depth, 0.15, 1.2)
            panels = [cv2.resize(left, (640, 360)), cv2.resize(mid, (640, 360))]
            if f.T_base_cam is not None:
                tcp = f.meta.get("T_base_flange")
                top = topdown_map(f, TabletopParams(), zt, tcp=None if tcp is None else tcp[:3, 3], objs=objs, grasp=grasp)
                panels.append(cv2.resize(top, (int(360 * top.shape[1] / top.shape[0]), 360)))
            img = np.concatenate(panels, 1)
            valid = float(np.mean(np.isfinite(f.depth)))
            m = f.meta or {}
            hud = (f"{m.get('fps', 0):4.1f} fps  frame age {1e3 * m.get('frame_age_s', 0):3.0f} ms  pose skew "
                   f"{1e3 * m.get('pose_skew_s', 0):3.1f} ms  robot {1e3 * m.get('robot_speed', 0):4.0f} mm/s  depth valid "
                   f"{100 * valid:3.0f}%  table z {zt:+.4f}" + (f"  BOARD vs calibration {berr[0]:.1f} mm {berr[1]:.2f} deg"
                                                                if berr else "  board not in view"))
            cv2.putText(img, hud, (8, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            stats.append({"fps": m.get("fps"), "frame_age_ms": 1e3 * m.get("frame_age_s", 0), "valid": valid,
                          "loop_ms": 1e3 * (time.time() - t0), "board_err": berr,
                          "grasp": None if grasp is None else grasp.summary()})
            n += 1
            if a.headless:
                if n >= a.headless:
                    cv2.imwrite(str(a.out / "live_last.jpg"), img[..., ::-1])
                    f.save(a.out / "live_last.npz")
                    break
                continue
            cv2.imshow("wrist ZED, calibrated (q quit, s save, g grasp preview)", img[..., ::-1])
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(a.out / f"live_{stamp}.jpg"), img[..., ::-1])
                f.save(a.out / f"live_{stamp}.npz")
                print("saved", a.out / f"live_{stamp}.*")
            if k == ord("g"):
                show_grasp = not show_grasp
    finally:
        cam.close()
        if poses is not None:
            poses.close()
        cv2.destroyAllWindows()
    s = stats[len(stats) // 3:] or stats
    summary = {"frames": n, "fps_median": float(np.median([x["fps"] or 0 for x in s])),
               "frame_age_ms_median": float(np.median([x["frame_age_ms"] for x in s])),
               "loop_ms_median": float(np.median([x["loop_ms"] for x in s])),
               "depth_valid_median": float(np.median([x["valid"] for x in s])),
               "board_err_last": stats[-1]["board_err"] if stats else None, "grasp_last": stats[-1]["grasp"] if stats else None}
    (a.out / "live_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
