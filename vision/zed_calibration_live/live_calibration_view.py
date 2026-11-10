"""Live calibration view for the wrist ZED Mini: see the ChArUco board, check the active hand-eye calibration, capture
poses for a new one. A side tool next to vision/zed_calibration (that folder and the lab's ../ZED are never modified).
Read-only towards the robot: it reads the pose over RTDE and never commands motion (move the arm with the pendant /
freedrive).

    python vision/zed_calibration_live/live_calibration_view.py            # window on the desktop
    python vision/zed_calibration_live/live_calibration_view.py --solve vision/zed_calibration_live/captures/<session>

What it shows
  - the board detection (markers, corners; "board OK" as in ur10e_hand_eye_calibration.py: >= 8 points);
  - the board's pose in the robot base frame through the ACTIVE calibration: T_base_board = T_base_flange @
    T_flange_cam @ T_cam_board;
  - consistency: each time the arm stands still at a new pose (> 2 cm or > 5 deg from the previous sample) with the
    board in view, the base-frame board pose is added; the board does not move, so the spread over poses
    (mean +- std distance to the average, mm / deg) measures the calibration. The original session had 3.4 +- 2.1 mm.
Keys:  y  save a capture (image + T_base_ee + T_cam_board + q, in the poses.npz format of the original script)
       h  solve the hand-eye calibration from this session's captures (Tsai, as the original), saved in the session
       r  reset the consistency samples      q  quit
Camera: the ZED SDK (rectified image, factory intrinsics) when pyzed is importable; otherwise the raw UVC stream,
which is NOT rectified: the lens distortion biases the board poses, so numbers in raw mode are approximate (the HUD
says so) and captures from it are for setting up only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from vision import WristCamera  # noqa: E402
from vision.charuco import detect_board  # noqa: E402
from vision.safety import _rotvec  # noqa: E402

MIN_POINTS_SAMPLE = 30


def rot_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(_rotvec(Ra.T @ Rb))))


def average_T(Ts):
    """Chordal mean rotation + mean translation (as average_transforms in the original script)."""
    M = sum(T[:3, :3] for T in Ts) / len(Ts)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def spread(Ts):
    m = average_T(Ts)
    t = [1e3 * np.linalg.norm(T[:3, 3] - m[:3, 3]) for T in Ts]
    r = [rot_deg(m[:3, :3], T[:3, :3]) for T in Ts]
    return m, (float(np.mean(t)), float(np.std(t)), float(np.mean(r)), float(np.std(r)))


def solve_session(session: Path, wrist: WristCamera) -> dict:
    """Tsai AX = XB on a session's poses.npz (same math as ur10e_hand_eye_calibration.compute_and_save)."""
    import cv2
    d = np.load(session / "poses.npz")
    A, B = list(d["T_base_ee"]), list(d["T_camw_board"])
    if len(A) < 4:
        raise SystemExit(f"{len(A)} captures in {session}: need at least 4 (15-20 recommended)")
    R, t = cv2.calibrateHandEye([T[:3, :3] for T in A], [T[:3, 3] for T in A], [T[:3, :3] for T in B],
                                [T[:3, 3] for T in B], method=cv2.CALIB_HAND_EYE_TSAI)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t.ravel()
    board, (tm, ts, rm, rs) = spread([a @ X @ b for a, b in zip(A, B)])
    _, (tm0, ts0, rm0, rs0) = spread([a @ wrist.T_flange_cam @ b for a, b in zip(A, B)])
    res = {"num_poses": len(A), "consistency_new_mm": [tm, ts], "consistency_new_deg": [rm, rs],
           "consistency_active_mm": [tm0, ts0], "consistency_active_deg": [rm0, rs0],
           "new_vs_active_mm": 1e3 * float(np.linalg.norm(X[:3, 3] - wrist.T_flange_cam[:3, 3])),
           "new_vs_active_deg": rot_deg(X[:3, :3], wrist.T_flange_cam[:3, :3]),
           "source": json.loads((session / "meta.json").read_text()).get("source", "?") if (session / "meta.json").exists() else "?"}
    res["recommendation"] = ("new result is more consistent: consider adopting it (edit active_calibration.txt)"
                             if tm < 0.8 * tm0 else "keep the active calibration (the new solve is not better)")
    np.savez(session / "hand_eye_calibration_ur10e_result.npz", T_ee_cam_wrist=X, T_base_board=board, num_poses=len(A))
    (session / "solve.json").write_text(json.dumps(res, indent=2))
    return res


class Session:
    def __init__(self, root: Path, source: str, K):
        self.dir = root / time.strftime("%Y%m%d_%H%M%S")
        self.A, self.B, self.Q = [], [], []
        self.source, self.K = source, K

    def add(self, img_rgb, T_base_ee, T_cam_board, q):
        import cv2
        self.dir.mkdir(parents=True, exist_ok=True)
        self.A.append(T_base_ee)
        self.B.append(T_cam_board)
        self.Q.append(q)
        cv2.imwrite(str(self.dir / f"pose{len(self.A):02d}_wrist.png"), img_rgb[..., ::-1])
        np.savez(self.dir / "poses.npz", T_base_ee=np.stack(self.A), T_camw_board=np.stack(self.B), q=np.stack(self.Q))
        (self.dir / "meta.json").write_text(json.dumps({"source": self.source, "K": np.asarray(self.K).tolist(),
                                                        "note": "T_base_ee = RTDE actual TCP pose with TCP = flange"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="192.168.0.2")
    ap.add_argument("--uvc", action="store_true", help="force the raw UVC stream even if pyzed is available")
    ap.add_argument("--solve", type=Path, help="only solve a saved session and print the comparison")
    ap.add_argument("--captures", type=Path, default=HERE / "captures")
    ap.add_argument("--snapshot", type=Path, help="save one annotated frame here and exit (no window)")
    a = ap.parse_args()
    wrist = WristCamera.active()
    if a.solve:
        print(json.dumps(solve_session(a.solve, wrist), indent=2))
        return
    import cv2
    from vision.stream import RTDEPoses
    cam, source, K = None, "", None
    if not a.uvc:
        try:
            from vision.zed import ZedCamera
            cam = ZedCamera(serial=wrist.serial or None, depth_mode="NONE")
            source, K = "ZED SDK, rectified", cam.intr.K
            grab = lambda: cam.grab().rgb
        except Exception as e:  # noqa: BLE001
            print(f"ZED SDK not available ({str(e).splitlines()[0][:90]}...): using the raw UVC stream")
            cam = None
    if cam is None:
        from vision.zed import UVCStereo
        cam = UVCStereo()
        source, K = "RAW UVC (not rectified): board poses APPROXIMATE", wrist.intr.K
        grab = lambda: cam.read()[0]
    poses = RTDEPoses(a.host)
    tcp_mm, tcp_deg = poses.check_tcp()
    samples, last_T, still_since = [], None, None
    sess = Session(a.captures, source, K)
    msg, msg_t = "", 0.0
    win = "wrist ZED - live calibration view (y capture, h solve, r reset, q quit)"
    if not a.snapshot:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 720)
    try:
        while True:
            img = grab()
            t_img = time.time()
            T_flange, info = poses.at(t_img)
            q = info["q"]
            Tcb, npts, (mc, mi) = detect_board(img, K)
            vis = np.ascontiguousarray(img[..., ::-1].copy())
            if mi is not None and len(mi):
                cv2.aruco.drawDetectedMarkers(vis, mc, mi)
            moving = info["speed"] > 0.006            # RTDE jitter alone reads ~3 mm/s
            still_since = None if moving else (still_since or t_img)
            settled = still_since is not None and t_img - still_since > 0.5
            T_bb = None
            if Tcb is not None:
                cv2.drawFrameAxes(vis, K, np.zeros(5), cv2.Rodrigues(Tcb[:3, :3])[0], Tcb[:3, 3], 0.2, 3)
                T_bb = T_flange @ wrist.T_flange_cam @ Tcb
                if settled and npts >= MIN_POINTS_SAMPLE and (
                        last_T is None or np.linalg.norm(T_flange[:3, 3] - last_T[:3, 3]) > 0.02
                        or rot_deg(T_flange[:3, :3], last_T[:3, :3]) > 5.0):
                    samples.append(T_bb)
                    last_T = T_flange.copy()
            ok = Tcb is not None
            col = (0, 200, 0) if ok else (0, 0, 255)
            cv2.putText(vis, f"{'board OK' if ok else 'board not enough'} ({0 if mi is None else len(mi)} markers, {npts} pts)",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
            # side panel
            P = np.full((720, 640, 3), 30, np.uint8)
            lines = [("CAMERA", source), ("calibration", WristCamera.active(warn=False).source.split("captures_ur10e/")[-1]),
                     ("robot", f"{'MOVING' if moving else ('still' if settled else 'settling')}  "
                               f"{1e3 * info['speed']:.0f} mm/s"),
                     ("q (deg)", str(np.round(np.degrees(q), 1).tolist())),
                     ("TCP check", f"controller vs FK {tcp_mm:.1f} mm / {tcp_deg:.2f} deg"
                                   f" ({'flange: OK' if tcp_mm < 6 else 'NOT flange!'})")]
            if Tcb is not None:
                dist = float(np.linalg.norm(Tcb[:3, 3]))
                lines += [("board-camera", f"{dist:.2f} m" + ("  (far: aim for 0.4-1.0 m)" if dist > 1.2 else "")),
                          ("board in base", f"origin {np.round(T_bb[:3, 3], 3).tolist()} m"),
                          ("board normal", f"{np.round(T_bb[:3, 2], 3).tolist()}")]
            if len(samples) >= 2:
                _, (tm, ts, rm, rs) = spread(samples)
                lines += [("CONSISTENCY", f"{len(samples)} poses: {tm:.1f} +- {ts:.1f} mm, {rm:.2f} +- {rs:.2f} deg"),
                          ("", "(original calibration: 3.4 +- 2.1 mm)")]
            else:
                lines += [("CONSISTENCY", f"{len(samples)} pose(s): move the arm to another view of the board")]
            lines += [("captures", f"{len(sess.A)} in {sess.dir.relative_to(ROOT) if sess.A else '(none yet)'}")]
            if msg and time.time() - msg_t < 6:
                lines += [("", msg)]
            y = 30
            for k, v in lines:
                cv2.putText(P, k, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1, cv2.LINE_AA)
                for j in range(0, max(1, len(v)), 44):
                    cv2.putText(P, v[j:j + 44], (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
                    y += 24
                y += 8
            if "RAW" in source:
                cv2.putText(P, "RAW image: lens distortion not removed,", (12, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)
                cv2.putText(P, "numbers are approximate (needs ZED SDK)", (12, 676), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)
            frame = np.concatenate([cv2.resize(vis, (1280, 720)), P], 1)
            if a.snapshot:
                cv2.imwrite(str(a.snapshot), frame)
                print("\n".join(f"{k}: {v}" for k, v in lines))
                break
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                samples, last_T = [], None
                msg, msg_t = "consistency samples reset", time.time()
            if k == ord("y"):
                if Tcb is None or moving:
                    msg, msg_t = "not captured: board not detected or arm moving", time.time()
                else:
                    sess.add(img, T_flange, Tcb, q)
                    msg, msg_t = f"captured pose {len(sess.A)}", time.time()
            if k == ord("h"):
                try:
                    r = solve_session(sess.dir, wrist)
                    msg = (f"solved {r['num_poses']}: new {r['consistency_new_mm'][0]:.1f} mm vs active "
                           f"{r['consistency_active_mm'][0]:.1f} mm -> {r['recommendation']}")
                except SystemExit as e:
                    msg = str(e)
                msg_t = time.time()
                print(msg)
    finally:
        cam.close()
        poses.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
