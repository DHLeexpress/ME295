"""The wrist ZED Mini in MuJoCo: a camera with the real intrinsics at the calibrated pose on the flange, rendering
the same RGBDFrame the real camera gives (left image + planar depth -> XYZ in the IMAGE frame).

    spec = mujoco.MjSpec.from_file("assembly/scene_bookshelf.xml")
    add_wrist_zed(spec, WristCamera.active())          # camera (+ a visual-only housing) on body "flange"
    m = spec.compile(); ...
    cam = SimZedCamera(m)                              # renders at the camera's own resolution (scale= to subsample)
    frame = cam.grab(d)                                # RGBDFrame with T_base_cam from the simulation

MuJoCo convention, checked in vision/tests/test_vision.py: a camera looks along its -z with +y up, so its frame is
the IMAGE frame times diag(1, -1, -1); `principal_pixel` is the principal point's offset from the image centre in
continuous coordinates (pixel centres at +0.5), so for OpenCV / ZED intrinsics (pixel centres at integers) it is
(W/2 - cx - 0.5, H/2 - cy - 0.5); depth renders as planar z.
"""
from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
from typing import Optional  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .camera import Intrinsics, RGBDFrame  # noqa: E402
from .wrist import WristCamera  # noqa: E402

ZED_CAM = "zed_cam"
FLIP = np.diag([1.0, -1.0, -1.0])                 # IMAGE frame <-> MuJoCo camera frame
ZED_MINI_SIZE = (0.1245, 0.0305, 0.0265)          # housing w x h x d (m); left lens 31.5 mm left of centre
ZED_MINI_BASELINE = 0.063
PIXEL_PITCH = 2e-6                                # only sets MuJoCo's sensor size; any positive value works


def _mat2quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def add_wrist_zed(spec: mujoco.MjSpec, wrist: Optional[WristCamera] = None, name: str = ZED_CAM,
                  body: str = "flange", housing: bool = True) -> mujoco.MjsCamera:
    """Add the calibrated ZED Mini to `spec` (in place). The housing is visual only (no contacts, no mass), so the
    dynamics of the scene are unchanged."""
    wrist = wrist or WristCamera.active()
    T = wrist.T_flange_cam
    b = spec.body(body)
    cam = b.add_camera()
    cam.name = name
    cam.pos = T[:3, 3].tolist()
    cam.quat = _mat2quat(T[:3, :3] @ FLIP).tolist()
    i = wrist.intr
    cam.resolution = [i.width, i.height]
    cam.sensor_size = [i.width * PIXEL_PITCH, i.height * PIXEL_PITCH]
    cam.focal_pixel = [i.fx, i.fy]
    cam.principal_pixel = [0.5 * i.width - i.cx - 0.5, 0.5 * i.height - i.cy - 0.5]
    if housing:
        g = b.add_geom()
        g.name = f"{name}_housing"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.5 * s for s in ZED_MINI_SIZE]
        g.pos = (T[:3, :3] @ np.array([0.5 * ZED_MINI_BASELINE, 0.0, -0.5 * ZED_MINI_SIZE[2] + 0.004]) + T[:3, 3]).tolist()
        g.quat = _mat2quat(T[:3, :3]).tolist()
        g.contype = g.conaffinity = 0
        g.density = 0.0
        g.group = 2
        g.rgba = [0.12, 0.12, 0.13, 1.0]
    return cam


def camera_intrinsics(m: mujoco.MjModel, name: str = ZED_CAM) -> Intrinsics:
    """Pixel intrinsics of a MuJoCo camera that was given sensor size / focal / principal point."""
    c = m.camera(name).id
    w, h = (int(v) for v in m.cam_resolution[c])
    sx, sy = m.cam_sensorsize[c]
    if sy <= 0:   # fovy camera: symmetric frustum
        fy = 0.5 * h / np.tan(np.radians(m.cam_fovy[c]) / 2)
        return Intrinsics(w, h, fy, fy, 0.5 * w - 0.5, 0.5 * h - 0.5)
    fxl, fyl, pxl, pyl = m.cam_intrinsic[c]
    fx, fy = fxl / sx * w, fyl / sy * h
    return Intrinsics(w, h, fx, fy, 0.5 * w - pxl / sx * w - 0.5, 0.5 * h - pyl / sy * h - 0.5)


def camera_pose(m: mujoco.MjModel, d: mujoco.MjData, name: str = ZED_CAM) -> np.ndarray:
    """T_base_cam (IMAGE convention) of a MuJoCo camera in the current state."""
    c = m.camera(name).id
    T = np.eye(4)
    T[:3, :3] = d.cam_xmat[c].reshape(3, 3) @ FLIP
    T[:3, 3] = d.cam_xpos[c]
    return T


def _min_filter(z: np.ndarray, k: int) -> np.ndarray:
    """(2k+1) x (2k+1) minimum filter (numpy only): nearer depth spreads k pixels."""
    out = z.copy()
    pad = np.pad(z, k, mode="edge")
    h, w = z.shape
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            np.minimum(out, pad[k + dy:k + dy + h, k + dx:k + dx + w], out=out)
    return out


class SimZedCamera:
    """Renders RGBDFrames from a MuJoCo camera. scale < 1 renders fewer pixels (K scaled to match).
    Depth outside [min_depth, max_depth] is NaN, like the ZED SDK's depth range. Two stereo artefacts, both off by
    default (exact depth): disparity_noise_px adds sigma_z = z^2 / (f B) * sigma_d; edge_fatten_px grows nearer
    surfaces into farther ones by that many pixels (a window-based matcher's "foreground fattening": books look
    thicker and gaps narrower)."""

    def __init__(self, m: mujoco.MjModel, name: str = ZED_CAM, scale: float = 1.0, min_depth: float = 0.1,
                 max_depth: float = 1.5, disparity_noise_px: float = 0.0, edge_fatten_px: int = 0, seed: int = 0):
        self.m, self.name = m, name
        self.intr_full = camera_intrinsics(m, name)
        self.intr = self.intr_full.scaled(scale) if scale != 1.0 else self.intr_full
        w, h = self.intr.width, self.intr.height
        if w > m.vis.global_.offwidth or h > m.vis.global_.offheight:
            raise ValueError(f"offscreen buffer {m.vis.global_.offwidth}x{m.vis.global_.offheight} < {w}x{h}: "
                             f"raise <visual><global offwidth/offheight>")
        self.renderer = mujoco.Renderer(m, h, w)
        self.min_depth, self.max_depth = min_depth, max_depth
        self.sigma_d = disparity_noise_px
        self.fatten = int(edge_fatten_px)
        self.rng = np.random.default_rng(seed)

    def grab(self, d: mujoco.MjData, segmentation: bool = False) -> RGBDFrame:
        r = self.renderer
        r.update_scene(d, camera=self.name)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, camera=self.name)
        z = r.render().astype(np.float32)
        r.disable_depth_rendering()
        if self.fatten > 0:
            z = _min_filter(z, self.fatten)
        z[(z < self.min_depth) | (z > self.max_depth)] = np.nan
        if self.sigma_d > 0:
            z = z + self.rng.normal(0.0, 1.0, z.shape).astype(np.float32) * z ** 2 / (
                self.intr.fx * ZED_MINI_BASELINE) * self.sigma_d
        meta = {}
        if segmentation:
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=self.name)
            meta["segmentation"] = r.render()[..., :2].copy()   # (objid, objtype) per pixel
            r.disable_segmentation_rendering()
        return RGBDFrame(rgb=rgb, xyz=self.intr.deproject(z), intr=self.intr, T_base_cam=camera_pose(self.m, d, self.name),
                         stamp=float(d.time), source=f"sim:{self.name}", meta=meta)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
