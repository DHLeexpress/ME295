"""The hand-eye calibration target (Boston Dynamics Spot ChArUco board: 9 x 4 squares of 115 mm, 90 mm markers,
DICT_4X4_50, legacy pattern) and its detection, as in vision/zed_calibration/calibration/ur10e_hand_eye_calibration.py
(full board: interpolated ChArUco corners + every visible marker's corners in one solvePnP). Used for a live check of
the calibration: the board seen now, through the calibration, should land where the calibration session put it
(T_base_board in the result npz) if the board has not moved."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

BOARD = dict(squares=(9, 4), square=0.115, marker=0.090, legacy=True)
MIN_POINTS = 8


def _board():
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    b = cv2.aruco.CharucoBoard(BOARD["squares"], BOARD["square"], BOARD["marker"], d)
    b.setLegacyPattern(BOARD["legacy"])
    det = cv2.aruco.CharucoDetector(b, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters())
    obj = {int(i): np.asarray(o, float).reshape(4, 3) for i, o in zip(b.getIds().ravel(), b.getObjPoints())}
    return b, det, obj


_CACHE = {}


def detect_board(rgb: np.ndarray, K: np.ndarray) -> Tuple[Optional[np.ndarray], int, tuple]:
    """(T_cam_board or None, number of points, (marker_corners, marker_ids)) from an RGB image (rectified, K)."""
    import cv2
    if "b" not in _CACHE:
        _CACHE["b"] = _board()
    board, det, idobj = _CACHE["b"]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cc, ci, mc, mi = det.detectBoard(gray)
    if mi is None:
        return None, 0, (None, None)
    O, I = [], []
    if ci is not None and len(ci):
        o, i = board.matchImagePoints(cc, ci)
        O.append(o.reshape(-1, 3))
        I.append(i.reshape(-1, 2))
    for c, k in zip(mc, mi.ravel()):
        if int(k) in idobj:
            O.append(idobj[int(k)])
            I.append(c.reshape(4, 2))
    O, I = np.concatenate(O).astype(np.float64), np.concatenate(I).astype(np.float64)
    if len(O) < MIN_POINTS:
        return None, len(O), (mc, mi)
    ok, rvec, tvec = cv2.solvePnP(O, I, K, np.zeros(5))
    if not ok:
        return None, len(O), (mc, mi)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, len(O), (mc, mi)


def board_outline(T_base_board: np.ndarray) -> np.ndarray:
    """(4, 3) corners of the board in the base frame."""
    w, h = BOARD["squares"][0] * BOARD["square"], BOARD["squares"][1] * BOARD["square"]
    return np.array([T_base_board[:3, :3] @ np.array([x, y, 0.0]) + T_base_board[:3, 3]
                     for x, y in ((0, 0), (w, 0), (w, h), (0, h))])
