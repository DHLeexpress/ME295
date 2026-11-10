"""Where to put the arm so the wrist ZED sees the shelf: the "look" pose before the task.

The calibrated ZED sits 10 cm off the tool axis, tilted 21 deg toward it, so the gripper fills the image half on the
tool-axis side (27-34 % of the image with the 2F-140 + blades). look_pose() therefore puts the shelf at a chosen pixel
in the free half and makes the line of sight to it horizontal and square to the spines, so the camera looks straight
into the gaps (the strongest depth contrast between a spine and the air beside it).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .wrist import WristCamera

HORIZONTAL = (-np.pi / 2, 0.0, 0.0)   # run_librarian.py: tool z -> +y (into shelf A), blade width vertical


def _rotvec_to_mat(r) -> np.ndarray:
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def look_pose(wrist: WristCamera, target: Sequence[float], into: Sequence[float] = (0.0, 1.0, 0.0),
              distance: float = 0.45, pixel: Tuple[float, float] = (0.5, 0.27),
              tool_rot: Sequence[float] = HORIZONTAL) -> np.ndarray:
    """T_base_flange such that the ray through `pixel` (fractions of width, height) runs along `into` and hits
    `target` at `distance` from the camera. Starts from the tool orientation `tool_rot` (turned to face `into`) and
    adds only a pitch about the row axis and a yaw about vertical, so the blades stay vertical-ish."""
    into = np.asarray(into, float)
    into = into / np.linalg.norm(into)
    up = np.array([0.0, 0.0, 1.0])
    row = np.cross(into, up)
    Q = np.stack([row, into, up], 1)                       # local (row, into, up) -> base
    heading = np.arctan2(-into[0], into[1])                 # into = Rz(heading) (0, 1, 0)
    R0 = _rz(heading) @ _rotvec_to_mat(tool_rot)
    i = wrist.intr
    d_cam = np.array([(pixel[0] * i.width - i.cx) / i.fx, (pixel[1] * i.height - i.cy) / i.fy, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    v = Q.T @ (R0 @ wrist.T_flange_cam[:3, :3] @ d_cam)
    th = np.arctan2(-v[2], v[1])
    rho = np.hypot(v[1], v[2])
    psi = np.arctan2(v[0], rho)
    R = Q @ _rz(psi) @ _rx(th) @ Q.T @ R0
    cam = np.asarray(target, float) - distance * into
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cam - R @ wrist.T_flange_cam[:3, 3]
    return T


TOOL_DOWN = (np.pi, 0.0, 0.0)   # tool z -> -z (gripper pointing at the table), tool x -> +x


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def view_pose(wrist: WristCamera, target: Sequence[float], direction: Sequence[float] = (0.0, 0.0, -1.0),
              distance: float = 0.5, pixel: Tuple[float, float] = (0.5, 0.30), tool_rot: Sequence[float] = TOOL_DOWN,
              yaw: float = 0.0) -> np.ndarray:
    """T_base_flange such that the ray through `pixel` runs along `direction` (default straight down: a top-down
    view) and hits `target` at `distance`. Orientation: tool_rot turned by `yaw` about vertical, then the smallest
    rotation that aligns the pixel's ray with `direction`."""
    g = np.asarray(direction, float)
    g = g / np.linalg.norm(g)
    R0 = _rz(yaw) @ _rotvec_to_mat(tool_rot)
    i = wrist.intr
    d_cam = np.array([(pixel[0] * i.width - i.cx) / i.fx, (pixel[1] * i.height - i.cy) / i.fy, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    v = R0 @ wrist.T_flange_cam[:3, :3] @ d_cam
    R = _align(v, g) @ R0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(target, float) - distance * g - R @ wrist.T_flange_cam[:3, 3]
    return T
