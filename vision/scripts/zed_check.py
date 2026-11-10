"""Check the wrist ZED Mini (and, read-only, its calibration against the robot). Nothing moves.

    python vision/scripts/zed_check.py                 # SDK: camera info, intrinsics vs snapshot, fps + depth per mode
    python vision/scripts/zed_check.py --robot         # + robot pose over RTDE: TCP sanity, base-frame cloud saved
    python vision/scripts/zed_check.py --snapshot      # write the live intrinsics into vision/zed_mini_wrist.json
    python vision/scripts/zed_check.py --uvc           # without the SDK: raw side-by-side stream (no depth)

Outputs in results/vision/zed_check/<time>/: left image, depth colormap, frame .npz (re-runnable offline), and with
--robot the point cloud in the base frame (.ply) and the calibration overlay.
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

from vision import WristCamera, depth_colormap  # noqa: E402


def write_ply(path, pts, rgb):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, rgb):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--uvc", action="store_true")
    ap.add_argument("--robot", action="store_true", help="read the robot pose over RTDE (read-only)")
    ap.add_argument("--host", default="192.168.0.2")
    ap.add_argument("--modes", default="NEURAL,ULTRA", help="depth modes to time")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "vision" / "zed_check" / time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    import cv2
    a.out.mkdir(parents=True, exist_ok=True)
    wrist = WristCamera.active()
    if a.uvc:
        from vision.zed import UVCStereo
        with UVCStereo() as cam:
            t0 = time.time()
            for _ in range(a.frames):
                left, right = cam.read()
            fps = a.frames / (time.time() - t0)
            cv2.imwrite(str(a.out / "uvc_left_right.jpg"), np.concatenate([left, right], 1)[..., ::-1])
            print(f"UVC {cam.device} {cam.size[0]}x{cam.size[1]} side by side, {fps:.1f} fps -> {a.out}")
        return
    from vision.zed import ZedCamera
    report = {}
    for mode in a.modes.split(","):
        with ZedCamera(serial=wrist.serial or None, depth_mode=mode) as cam:
            if not report:
                print(cam.describe())
                report["camera"] = cam.describe()
                d = {k: getattr(cam.intr, k) - getattr(wrist.intr, k) for k in ("fx", "fy", "cx", "cy")}
                print("intrinsics live - snapshot (px):", {k: round(v, 2) for k, v in d.items()})
                report["intrinsics_delta_px"] = d
                if a.snapshot:
                    wrist.intr = cam.intr
                    wrist.serial = cam.serial
                    wrist.notes["intrinsics"] = f"{cam.intr.width}x{cam.intr.height} from the camera {time.strftime('%Y-%m-%d')}"
                    wrist.save()
                    print("vision/zed_mini_wrist.json updated")
            for _ in range(5):
                cam.grab()
            t0 = time.time()
            for _ in range(a.frames):
                f = cam.grab()
            fps = a.frames / (time.time() - t0)
            z = f.depth
            valid = np.isfinite(z)
            report[mode] = {"fps": fps, "valid": float(valid.mean()),
                            "depth_median_m": float(np.nanmedian(z)) if valid.any() else None}
            print(f"{mode:8s} {fps:5.1f} fps, depth valid {100 * valid.mean():.0f} %, median {report[mode]['depth_median_m']}")
            cv2.imwrite(str(a.out / f"left_{mode}.jpg"), f.rgb[..., ::-1])
            cv2.imwrite(str(a.out / f"depth_{mode}.jpg"), depth_colormap(z, 0.15, 1.5)[..., ::-1])
            last = f
    if a.robot:
        from vision.stream import RTDEPoses
        poses = RTDEPoses(a.host)
        mm, deg = poses.check_tcp()
        T, info = poses.at(time.time())
        poses.close()
        print(f"robot: controller vs nominal FK {mm:.1f} mm / {deg:.2f} deg; speed {1e3 * info['speed']:.0f} mm/s")
        f = last.with_pose(wrist.T_base_cam(T))
        f.save(a.out / "frame.npz")
        pts, rgb = f.points_base(stride=4)
        write_ply(a.out / "cloud_base.ply", pts, rgb)
        sys.path.insert(0, str(ROOT / "vision" / "scripts"))
        from zed_live import overlay
        cv2.imwrite(str(a.out / "calibration_overlay.jpg"), overlay(f, None, 0.0065)[..., ::-1])
        report["robot"] = {"fk_vs_controller_mm": mm, "deg": deg, "camera_position": f.T_base_cam[:3, 3].tolist()}
    else:
        last.save(a.out / "frame.npz")
    (a.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print("->", a.out)


if __name__ == "__main__":
    main()
