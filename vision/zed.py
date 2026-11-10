"""The real wrist ZED Mini.

ZedCamera   ZED SDK (pyzed): rectified left image + depth -> RGBDFrame, same settings as
            vision/zed_calibration/perception/capture_pointclouds.py (HD720, NEURAL depth, metres, IMAGE coordinates). Needs the ZED SDK and its Python
            API in this environment, and read access to /usr/local/zed (on this PC: membership of group `zed`).
UVCStereo   the camera as a plain UVC webcam (OpenCV, no SDK): raw side-by-side left|right images, unrectified,
            no depth. Enough to check that the camera streams and what it sees on the mount.
"""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .camera import Intrinsics, RGBDFrame

SDK_HELP = ("pyzed (ZED SDK Python API) is not importable. Install it into this environment with the SDK's script "
            "(python /usr/local/zed/get_python_api.py) and make sure this user can read /usr/local/zed "
            "(here: sudo usermod -aG zed $USER, then log in again).")


class ZedCamera:
    def __init__(self, serial: Optional[int] = None, resolution: str = "HD720", fps: int = 30,
                 depth_mode: str = "NEURAL", min_depth: float = 0.1, max_depth: float = 1.5,
                 confidence: Optional[int] = None):
        try:
            import pyzed.sl as sl
        except ImportError as e:
            raise RuntimeError(SDK_HELP) from e
        self.sl = sl
        init = sl.InitParameters()
        if serial:
            init.set_from_serial_number(int(serial))
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.camera_fps = fps
        init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        init.depth_minimum_distance = min_depth
        init.depth_maximum_distance = max_depth
        self.cam = sl.Camera()
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"could not open the ZED{f' {serial}' if serial else ''}: {status}")
        info = self.cam.get_camera_information()
        conf = info.camera_configuration
        c = conf.calibration_parameters.left_cam
        self.intr = Intrinsics(int(conf.resolution.width), int(conf.resolution.height), float(c.fx), float(c.fy),
                               float(c.cx), float(c.cy))
        self.baseline = float(conf.calibration_parameters.get_camera_baseline()) \
            if hasattr(conf.calibration_parameters, "get_camera_baseline") else float("nan")
        self.serial = int(info.serial_number)
        self.model = str(info.camera_model)
        self.firmware = getattr(conf, "firmware_version", None)
        self.sdk = sl.Camera.get_sdk_version()
        self.depth_mode = depth_mode
        self.runtime = sl.RuntimeParameters()
        if confidence is not None:
            self.runtime.confidence_threshold = int(confidence)
        self._img, self._xyz = sl.Mat(), sl.Mat()

    def grab(self, T_base_cam: Optional[np.ndarray] = None, retries: int = 5) -> RGBDFrame:
        sl = self.sl
        for _ in range(retries):
            if self.cam.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("ZED grab failed")
        self.cam.retrieve_image(self._img, sl.VIEW.LEFT)
        bgra = self._img.get_data()
        rgb = np.ascontiguousarray(bgra[..., 2::-1])
        if self.depth_mode == "NONE":            # image only (e.g. the calibration view)
            xyz = np.full((*rgb.shape[:2], 3), np.nan, np.float32)
        else:
            self.cam.retrieve_measure(self._xyz, sl.MEASURE.XYZ)
            xyz = np.array(self._xyz.get_data()[..., :3], dtype=np.float32)
            xyz[~np.isfinite(xyz).all(-1)] = np.nan
        stamp = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds() / 1e3
        return RGBDFrame(rgb=rgb, xyz=xyz, intr=self.intr, T_base_cam=T_base_cam, stamp=stamp,
                         source=f"zed:{self.serial}:{self.depth_mode}")

    def describe(self) -> str:
        i = self.intr
        return (f"{self.model} SN {self.serial}, SDK {self.sdk}, firmware {self.firmware}, {i.width}x{i.height}, "
                f"fx {i.fx:.1f} fy {i.fy:.1f} cx {i.cx:.1f} cy {i.cy:.1f} (fov {i.fovx_deg:.0f} x {i.fovy_deg:.0f} deg), "
                f"baseline {1e3 * self.baseline:.1f} mm, depth {self.depth_mode}")

    def close(self):
        if self.cam is not None:
            self.cam.close()
            self.cam = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def find_uvc_device(name: str = "ZED") -> Optional[str]:
    """/dev/videoN of the first V4L2 device whose name contains `name` (capture node: the lowest index)."""
    for d in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(p.rsplit("video", 1)[1])):
        try:
            if name.lower() in Path(d, "name").read_text().lower():
                return "/dev/" + os.path.basename(d)
        except OSError:
            continue
    return None


class UVCStereo:
    """Raw side-by-side frames over UVC (no SDK, no rectification, no depth)."""

    def __init__(self, device: Optional[str] = None, width: int = 2560, height: int = 720, fps: int = 30):
        import cv2
        self.cv2 = cv2
        self.device = device or find_uvc_device() or "/dev/video0"
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {self.device} (in group video, or logged in at the seat?)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("UVC read failed")
        w = f.shape[1] // 2
        return f[:, :w, ::-1].copy(), f[:, w:, ::-1].copy()   # RGB left, right

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
