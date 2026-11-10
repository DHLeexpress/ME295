"""Small SE(3) helpers (numpy only, no MuJoCo), so the embodiment module also runs on the robot PC."""
from __future__ import annotations

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF roll-pitch-yaw (fixed axes X, then Y, then Z): R = Rz(y) Ry(p) Rx(r). Same as MuJoCo eulerseq="XYZ"."""
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def axis_angle(axis, angle: float) -> np.ndarray:
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def quat_to_mat(q) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(1.0 + t)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def rotvec_to_mat(r) -> np.ndarray:
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    return np.eye(3) if th < 1e-12 else axis_angle(r / th, th)


def mat_to_rotvec(R) -> np.ndarray:
    q = mat_to_quat(R)
    s = np.linalg.norm(q[1:])
    if s < 1e-12:
        return np.zeros(3)
    return 2.0 * np.arctan2(s, q[0]) * q[1:] / s


def make(R=None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def from_pos_rpy(pos=(0, 0, 0), rpy_=(0, 0, 0)) -> np.ndarray:
    return make(rpy(*rpy_), pos)


def from_pos_quat(pos=(0, 0, 0), quat=(1, 0, 0, 0)) -> np.ndarray:
    return make(quat_to_mat(quat), pos)


def inv(T) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def pose6(T) -> np.ndarray:
    """4x4 -> urx pose [x, y, z, rx, ry, rz] (rotation vector)."""
    return np.r_[T[:3, 3], mat_to_rotvec(T[:3, :3])]


def from_pose6(p) -> np.ndarray:
    return make(rotvec_to_mat(p[3:6]), p[:3])


def rot6d(R) -> np.ndarray:
    """First two columns of R (continuous 6D rotation representation, Zhou et al. 2019)."""
    return np.asarray(R)[:, :2].T.reshape(6)


def rot6d_to_mat(v) -> np.ndarray:
    a, b = np.asarray(v, float).reshape(2, 3)
    x = a / np.linalg.norm(a)
    b = b - (x @ b) * x
    y = b / np.linalg.norm(b)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def twist_error(T_now, T_goal) -> np.ndarray:
    """6-vector [dp, dw] (base frame) that moves T_now toward T_goal."""
    dp = T_goal[:3, 3] - T_now[:3, 3]
    dw = T_now[:3, :3] @ mat_to_rotvec(T_now[:3, :3].T @ T_goal[:3, :3])
    return np.r_[dp, dw]
