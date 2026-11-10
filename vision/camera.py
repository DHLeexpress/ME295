"""Camera-agnostic RGB-D frames: the same type comes from the real ZED Mini (zed.py) and from MuJoCo (sim_zed.py).

Camera frame = the ZED SDK's IMAGE coordinate system (= OpenCV): x right, y down, z forward, metres, origin at the
LEFT camera's optical centre. That is the frame of `MEASURE.XYZ` point clouds and of the hand-eye calibration in
../ZED (T_ee_cam_wrist), so a frame from either source goes to the robot base frame the same way:

    p_base = T_base_cam @ p_cam,   T_base_cam = T_base_flange @ T_flange_cam      (wrist.py)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics of the left camera (rectified image; the ZED SDK reports zero distortion after
    rectification). u = fx x / z + cx, v = fy y / z + cy, pixel centres at integer coordinates."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def fovy_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(0.5 * self.height / self.fy)))

    @property
    def fovx_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(0.5 * self.width / self.fx)))

    def scaled(self, s: float) -> "Intrinsics":
        """Intrinsics of the same camera sampled at s x the resolution (pixel-centre convention kept)."""
        return Intrinsics(int(round(self.width * s)), int(round(self.height * s)), self.fx * s, self.fy * s,
                          (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5)

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        """(N, 3) camera-frame points -> (N, 2) pixel coordinates (u, v)."""
        p = np.atleast_2d(p_cam)
        return np.c_[self.fx * p[:, 0] / p[:, 2] + self.cx, self.fy * p[:, 1] / p[:, 2] + self.cy]

    def deproject(self, depth: np.ndarray) -> np.ndarray:
        """(H, W) planar depth z (m; 0 / NaN / inf = none) -> (H, W, 3) camera-frame XYZ, NaN where no depth."""
        h, w = depth.shape
        u = np.arange(w, dtype=np.float32)[None, :]
        v = np.arange(h, dtype=np.float32)[:, None]
        z = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float32)
        return np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], axis=-1)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy, "cx": self.cx,
                "cy": self.cy}


def transform_points(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (..., 3) points."""
    return p @ T[:3, :3].T + T[:3, 3]


@dataclass
class RGBDFrame:
    """One registered colour + depth frame from the left camera.

    rgb  (H, W, 3) uint8, left image
    xyz  (H, W, 3) float32, camera frame (x right, y down, z forward), NaN where there is no depth
    T_base_cam: camera pose in the robot base frame when the frame was taken (None if the robot pose is unknown)
    """
    rgb: np.ndarray
    xyz: np.ndarray
    intr: Intrinsics
    T_base_cam: Optional[np.ndarray] = None
    stamp: float = 0.0
    source: str = ""
    meta: dict = None

    @property
    def depth(self) -> np.ndarray:
        return self.xyz[..., 2]

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.xyz).all(axis=-1)

    def points_base(self, mask: Optional[np.ndarray] = None, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Valid points in the robot base frame and their colours: ((N, 3) float64, (N, 3) uint8)."""
        if self.T_base_cam is None:
            raise ValueError("frame has no camera pose (T_base_cam): set it from the robot's flange pose")
        keep = self.valid if mask is None else (self.valid & mask)
        if stride > 1:
            sub = np.zeros_like(keep)
            sub[::stride, ::stride] = True
            keep &= sub
        return transform_points(self.T_base_cam, self.xyz[keep].astype(np.float64)), self.rgb[keep]

    def with_pose(self, T_base_cam: np.ndarray) -> "RGBDFrame":
        return replace(self, T_base_cam=np.asarray(T_base_cam, float))

    def save(self, path) -> None:
        """Compressed .npz with everything needed to re-run perception offline."""
        np.savez_compressed(path, rgb=self.rgb, xyz=self.xyz.astype(np.float32), K=self.intr.K,
                            size=np.array([self.intr.width, self.intr.height]),
                            T_base_cam=np.full((4, 4), np.nan) if self.T_base_cam is None else self.T_base_cam,
                            stamp=self.stamp, source=self.source)

    @classmethod
    def load(cls, path) -> "RGBDFrame":
        z = np.load(path, allow_pickle=False)
        K, (w, h) = z["K"], z["size"]
        T = z["T_base_cam"]
        return cls(rgb=z["rgb"], xyz=z["xyz"], intr=Intrinsics(int(w), int(h), K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
                   T_base_cam=None if np.isnan(T).any() else T, stamp=float(z["stamp"]), source=str(z["source"]))


def depth_colormap(depth: np.ndarray, near: float = 0.2, far: float = 1.2) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) uint8 turbo-like colours (black = no depth), numpy only."""
    t = np.clip((np.nan_to_num(depth, nan=far) - near) / (far - near), 0, 1)
    # polynomial fit of the turbo colormap (Mikhailov 2019), good to a few percent
    r = 0.1357 + t * (4.5974 - t * (42.3277 - t * (130.5887 - t * (150.5666 - t * 58.1375))))
    g = 0.0914 + t * (2.1856 + t * (4.8052 - t * (14.0195 - t * (4.2109 + t * 2.7747))))
    b = 0.1067 + t * (12.5925 - t * (60.1097 - t * (109.0745 - t * (88.5066 - t * 26.8183))))
    rgb = (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)
    rgb[~(np.isfinite(depth) & (depth > 0))] = 0
    return rgb
