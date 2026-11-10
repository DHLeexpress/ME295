"""
Hand-eye calibration for a wrist-mounted ZED Mini on a UR10e, using the SAME
Boston Dynamics Spot calibration target (a ChArUco board -- 9x4 squares,
115mm square/90mm marker, DICT_4X4_50) that hand_eye_calibration_3cam.py
uses -- see that file for how the board geometry was confirmed. This is a
separate, self-contained script (not an import of hand_eye_calibration.py or
hand_eye_calibration_3cam.py) so it has no dependency on franky (the FR3's
SDK) being installed at all -- only urx + math3d, which you get from cloning
https://github.com/Caltech-AMBER/python-urx into this conda env.

Uses the FULL board (every visible marker + interpolated ChArUco corners),
not a single marker -- an earlier version of this script tracked just one
marker (ID 1) at an assumed 10cm size copied from hand_eye_calibration.py's
DIFFERENT physical tag, which was both the wrong size for this board's
actual 90mm markers AND inherently noisy (a single planar marker's 4 points
are known to have an occasional pose ambiguity). Reprocessing one capture
session's images confirmed it directly: T_base_board consistency across 20
poses went from 19-36mm (single marker, depending on assumed size) to
2.76+/-1.11mm using the full board (9-17 markers, 46-91 points per frame).

Run this with an env that has urx, math3d, opencv (with contrib/aruco), and
pyzed installed:
python ur10e_hand_eye_calibration.py

Setup assumed:
- The board is fixed in the workspace (e.g. taped to a table) and does not
move for the entire session.
- The ZED Mini is mounted on the UR10e's tool flange and moves with it.
- You manually move the UR10e between captures (e.g. freedrive/teach
pendant); the script shows a live preview (no timer) and waits for you
to press 'y' whenever you're ready for that pose to be captured (or 'q'
to stop early).

What it computes:
T_ee_cam_wrist: wrist camera pose expressed in the tool/end-effector
frame (i.e. p_ee = T_ee_cam_wrist @ p_cam_wrist), via classic AX=XB
hand-eye calibration (cv2.calibrateHandEye). "ee" here means the UR10e's
TCP/tool frame -- same name as hand_eye_calibration.py's FR3 output for
consistency across the codebase, even though UR's own docs call it "tool."

Robot connection: urx.Robot(ROBOT_IP) opens a live connection to read state
(get_pose()/getj()) -- this script never commands motion, same as
hand_eye_calibration.py's franky connection. You move the arm yourself
(e.g. via freedrive) between captures.

get_ee_pose() below uses trans = robot.get_pose() directly as T_base_ee
(trans.array is the 4x4 base->tool matrix) -- confirmed correct for this
use case; not doing any extra csys/frame juggling on top of it.
"""

import argparse
import datetime
import os
import sys
import time

import cv2
import numpy as np
import pyzed.sl as sl
import urx

# ==============================================================================
# CONFIGURATION
# ==============================================================================
WRIST_SERIAL = 18982088  # set the ZED Mini's serial number (see sl.Camera.get_device_list()
# or ZED Explorer) -- 0 is a placeholder and will fail to open.

# --- Spot calibration target (Boston Dynamics p/n 03-001036-001-A01) --------
# SAME physical board hand_eye_calibration_3cam.py uses -- see that file's
# module docstring for how these were confirmed. Duplicated here (not
# imported) to keep this script franky-free -- see module docstring.
BOARD_DICT = cv2.aruco.DICT_4X4_50
BOARD_SQUARES_X = 9
BOARD_SQUARES_Y = 4
BOARD_SQUARE_LENGTH_M = 0.115
BOARD_MARKER_LENGTH_M = 0.090
BOARD_LEGACY_PATTERN = True

# Minimum total (interpolated-ChArUco-corner + raw-marker-corner) points
# required before solvePnP is even attempted -- a single marker's 4 points
# is excluded on purpose: a single planar marker has a known pose ambiguity
# (confirmed directly on this UR10e's own captures -- see module docstring).
MIN_TOTAL_POINTS = 8

NUM_POSES = 20  # number of robot poses to sample
MIN_POSES = 4  # cv2.calibrateHandEye needs >= 3; want some margin
HAND_EYE_METHOD = cv2.CALIB_HAND_EYE_TSAI

ROBOT_IP = "192.168.0.2"  # was 172.16.0.20 until 2026-10; override with --robot-ip

CAPTURES_ROOT = "captures_ur10e"  # each run gets its own captures_ur10e/<timestamp>/ subfolder
POSES_FILENAME = "poses.npz"
OUTPUT_FILENAME = "hand_eye_calibration_ur10e_result.npz"


# ==============================================================================
# ROBOT POSE FEEDBACK (urx/UR10e connection)
# ==============================================================================
_arm = None


def _connect_robot():
    # Opens urx's live connection. In the default (manual) mode this only reads
    # state; only --replay (replay_capture below) commands motion.
    global _arm
    if _arm is None:
        _arm = urx.Robot(ROBOT_IP)
    return _arm


def get_ee_pose():
    """
    Return the robot's current tool/end-effector pose in the base frame as
    a 4x4 numpy float64 matrix T_base_ee, i.e. p_base_homog = T_base_ee @
    p_ee_homog.

    robot.get_pose() (urx/math3d) already returns exactly this: a
    math3d.Transform whose .array is the 4x4 base->tool matrix (assuming
    the default identity csys, i.e. no set_csys() call anywhere in this
    session) -- used directly, no extra conversion needed.
    """
    arm = _connect_robot()
    trans = arm.get_pose()
    return np.array(trans.array, dtype=np.float64)


def get_joint_positions():
    """
    Return the robot's current joint angles (radians), shape (6,) float64 --
    UR10e has 6 joints (vs. the FR3's 7). Live/actual state (urx's getj()
    reads the secondary monitor's q_actual, not a commanded target).
    """
    arm = _connect_robot()
    return np.asarray(arm.getj(), dtype=np.float64)


# ==============================================================================
# GEOMETRY HELPERS (byte-for-byte the same as hand_eye_calibration.py's --
# duplicated, not imported, to keep this script free of any franky
# dependency; see module docstring)
# ==============================================================================
def rt_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def average_rotations(rotations):
    """Chordal L2 mean of rotation matrices via orthogonal Procrustes (SVD)."""
    M = sum(rotations) / len(rotations)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def average_transforms(transforms):
    R_avg = average_rotations([T[:3, :3] for T in transforms])
    t_avg = np.mean([T[:3, 3] for T in transforms], axis=0)
    T = np.eye(4)
    T[:3, :3] = R_avg
    T[:3, 3] = t_avg
    return T


def rotation_angle_deg(R):
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def transform_spread(transforms, mean_T):
    """Mean/std of translation error (m) and rotation error (deg) vs mean_T."""
    t_errs = [np.linalg.norm(T[:3, 3] - mean_T[:3, 3]) for T in transforms]
    ang_errs = [rotation_angle_deg(mean_T[:3, :3].T @ T[:3, :3]) for T in transforms]
    return np.mean(t_errs), np.std(t_errs), np.mean(ang_errs), np.std(ang_errs)


# ==============================================================================
# CAMERA HELPERS (raw pyzed.sl -- same as hand_eye_calibration.py's ZedCam)
# ==============================================================================
class ZedCam:
    def __init__(self, serial_number):
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(serial_number)
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE  # only need RGB for tag detection
        init_params.coordinate_units = sl.UNIT.METER

        self.cam = sl.Camera()
        status = self.cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED {serial_number}: {status}")

        calib = self.cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.K = np.array([
            [calib.fx, 0.0, calib.cx],
            [0.0, calib.fy, calib.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.runtime_params = sl.RuntimeParameters()
        self.image_mat = sl.Mat()

    def grab_bgr(self):
        if self.cam.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None
        self.cam.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        return cv2.cvtColor(self.image_mat.get_data(), cv2.COLOR_BGRA2BGR)

    def close(self):
        self.cam.close()


def open_camera(serial_number):
    return ZedCam(serial_number)


# ==============================================================================
# CHARUCO BOARD DETECTION -> POSE (full or partial view)
# ==============================================================================
_BOARD_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(BOARD_DICT)
_BOARD = cv2.aruco.CharucoBoard(
    (BOARD_SQUARES_X, BOARD_SQUARES_Y), BOARD_SQUARE_LENGTH_M, BOARD_MARKER_LENGTH_M, _BOARD_ARUCO_DICT
)
_BOARD.setLegacyPattern(BOARD_LEGACY_PATTERN)
_ARUCO_DETECT_PARAMS = cv2.aruco.DetectorParameters()
_CHARUCO_DETECTOR = cv2.aruco.CharucoDetector(_BOARD, cv2.aruco.CharucoParameters(), _ARUCO_DETECT_PARAMS)
_ZERO_DIST = np.zeros(5, dtype=np.float64)

# Per-marker-ID object points (4x3, in the board's own frame), so any subset
# of visible markers -- not just fully-interpolated checkerboard corners --
# can contribute to solvePnP.
_BOARD_ID_TO_OBJ = {
    int(marker_id): np.asarray(obj_pts, dtype=np.float64).reshape(4, 3)
    for marker_id, obj_pts in zip(_BOARD.getIds().ravel(), _BOARD.getObjPoints())
}


def detect_board_pose(image_bgr, K):
    """
    Return (found, T_cam_board, num_points, num_markers, marker_corners, marker_ids)
    where p_cam_homog = T_cam_board @ p_board_homog.

    Combines interpolated ChArUco checkerboard corners (precise, but need
    several neighboring markers to interpolate) with raw per-marker corners
    (cruder, but work from however many individual tags are visible) into
    one solvePnP call, so a partial view of the board still yields a pose.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = _CHARUCO_DETECTOR.detectBoard(gray)
    if marker_ids is None:
        return False, None, 0, 0, None, None
    n_markers = len(marker_ids)

    obj_pts_list, img_pts_list = [], []
    if charuco_ids is not None and len(charuco_ids) > 0:
        charuco_obj, charuco_img = _BOARD.matchImagePoints(charuco_corners, charuco_ids)
        obj_pts_list.append(charuco_obj.reshape(-1, 3))
        img_pts_list.append(charuco_img.reshape(-1, 2))

    for corner_set, marker_id in zip(marker_corners, marker_ids.ravel()):
        obj = _BOARD_ID_TO_OBJ.get(int(marker_id))
        if obj is None:
            continue  # shouldn't happen -- detected marker not part of this board
        obj_pts_list.append(obj)
        img_pts_list.append(corner_set.reshape(4, 2))

    obj_pts = np.concatenate(obj_pts_list, axis=0).astype(np.float64)
    img_pts = np.concatenate(img_pts_list, axis=0).astype(np.float64)
    n_points = len(obj_pts)
    if n_points < MIN_TOTAL_POINTS:
        return False, None, n_points, n_markers, marker_corners, marker_ids

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, _ZERO_DIST)
    if not ok:
        return False, None, n_points, n_markers, marker_corners, marker_ids
    return True, rt_to_T(rvec, tvec), n_points, n_markers, marker_corners, marker_ids


def wait_for_capture_keypress(wrist_cam, pose_idx, total_poses):
    """
    Live preview, no timer -- reposition the robot at your own pace, then
    press 'y' in the preview window to attempt capturing this pose (retries
    in place if the board isn't visible enough right then), or 'q' to stop
    the whole session early.

    Returns (T_camw_board, img_w) on a successful capture, or None if the
    user pressed 'q'.
    """
    print(f"\n=== Pose {pose_idx}/{total_poses}: reposition the robot, then press 'y' "
          f"in the preview window to capture (or 'q' to stop early). ===")
    while True:
        img_w = wrist_cam.grab_bgr()

        result = (False, None, 0, 0, None, None)
        if img_w is not None:
            result = detect_board_pose(img_w, wrist_cam.K)
        found_w, _T, n_points, n_markers, marker_corners, marker_ids = result

        # NOTE(reconstruction): the pasted source lost all indentation, so
        # this file was rebuilt from context. This guard (skip the
        # detect/display block when the grab failed, instead of calling
        # img_w.copy() on a None) was added because it's the only way to
        # avoid a crash on a failed grab while still matching the "Camera
        # grab failed just now, try again." message below -- please
        # double-check this matches your original intent.
        if img_w is not None:
            # Detect ONCE, annotate a COPY for display -- drawDetectedMarkers
            # stamps ID text on/near the marker itself, so re-detecting on an
            # already-annotated image can wrongly lose markers on the next pass
            # (confirmed directly in hand_eye_calibration_3cam.py).
            display = img_w.copy()
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            status = f"{n_markers} tags, {n_points} pts" if n_markers else "no tags"
            label = f"board OK ({status})" if found_w else f"not enough ({status})"
            cv2.putText(display, f"wrist: {label} ('y'=capture, 'q'=stop)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if found_w else (0, 0, 255), 2)
            cv2.imshow("Wrist Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            return None

        if key in (ord("y"), ord("Y")):
            if img_w is None:
                print(" Camera grab failed just now, try again.")
                continue
            found_w, T_camw_board, n_points, n_markers, _mc, _mi = result
            if not found_w:
                print(f" Board not visible enough ({n_markers} tags, {n_points} pts, need "
                      f">={MIN_TOTAL_POINTS}) -- not captured, reposition and press 'y' again.")
                continue
            return T_camw_board, img_w


# ==============================================================================
# PERSISTENCE (save every capture as it happens -- a crash in the math below
# should never cost you a re-run of the physical capture session)
# ==============================================================================
def new_session_dir():
    session_dir = os.path.join(CAPTURES_ROOT, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def save_poses(session_dir, T_base_ee_list, T_camw_board_list):
    np.savez(
        os.path.join(session_dir, POSES_FILENAME),
        T_base_ee=np.stack(T_base_ee_list),
        T_camw_board=np.stack(T_camw_board_list),
    )


def load_poses(session_dir):
    data = np.load(os.path.join(session_dir, POSES_FILENAME))
    return list(data["T_base_ee"]), list(data["T_camw_board"])


# ==============================================================================
# CAPTURE
# ==============================================================================
def capture_poses(session_dir, num_new_poses=NUM_POSES, start_index=1,
                   T_base_ee_list=None, T_camw_board_list=None):
    """
    Capture num_new_poses poses, appending them to the given lists (if any)
    and to session_dir's saved images/poses.npz. Pose image filenames and
    progress messages are numbered starting at start_index, so this can
    extend an existing session (start_index = len(existing) + 1) without
    clobbering its earlier pose%02d_wrist.png files.
    """
    T_base_ee_list = list(T_base_ee_list) if T_base_ee_list is not None else []
    T_camw_board_list = list(T_camw_board_list) if T_camw_board_list is not None else []

    wrist_cam = open_camera(WRIST_SERIAL)

    end_index = start_index + num_new_poses - 1

    try:
        pose_idx = start_index - 1
        while pose_idx < end_index:
            pose_idx += 1
            result = wait_for_capture_keypress(wrist_cam, pose_idx, end_index)
            if result is None:
                print(f"\nStopped by 'q' after {len(T_base_ee_list)} total poses (saved to {session_dir}).")
                break
            T_camw_board, img_w = result

            T_base_ee = get_ee_pose()

            cv2.imwrite(os.path.join(session_dir, f"pose{pose_idx:02d}_wrist.png"), img_w)

            T_base_ee_list.append(T_base_ee)
            T_camw_board_list.append(T_camw_board)
            save_poses(session_dir, T_base_ee_list, T_camw_board_list)
            print(f"  Captured pose {pose_idx}: board seen, EE pose recorded and saved to {session_dir}.")

    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(T_base_ee_list)} total poses (saved to {session_dir}).")

    finally:
        cv2.destroyAllWindows()
        wrist_cam.close()

    return T_base_ee_list, T_camw_board_list


def replay_capture(session_dir, T_targets, acc=0.3, vel=0.2, settle_s=1.5, confirm_each=False, q_targets=None):
    """
    Automatic capture: drive the arm through previously recorded tool poses (e.g. the
    T_base_ee list from an earlier session's poses.npz, which were all chosen by hand to
    see the board), and capture wherever the board is detected. Each move is a joint-space
    move to the Cartesian pose (URScript movej(p[...]); the controller solves the joint
    angles), slow by default. If q_targets (one 6-joint configuration per pose, solved offline,
    e.g. SESSION_DIR/replay_joints_elbow_up.npy) is given, plain joint movej's are sent instead:
    no IK on the controller, so the arm configuration is exactly the one checked offline. The ACTUAL pose after the move is what gets recorded, so the
    replay doesn't need to be exact. Poses where the board isn't detected are skipped.
    Ctrl+C stops and keeps what was captured.
    """
    import math3d as m3d  # comes with urx

    arm = _connect_robot()
    wrist_cam = open_camera(WRIST_SERIAL)
    T_base_ee_list, T_camw_board_list = [], []
    n = len(T_targets)
    try:
        here = get_ee_pose()
        d0 = np.linalg.norm(T_targets[0][:3, 3] - here[:3, 3])
        print(f"\nReplaying {n} recorded poses at acc={acc}, vel={vel} (joint-space moves"
              f"{', to precomputed joint targets' if q_targets is not None else ''}).")
        if q_targets is not None:
            dq0 = np.degrees(np.abs(np.asarray(q_targets[0]) - get_joint_positions()).max())
            print(f"Largest joint move to the first target: {dq0:.0f} deg.")
        print(f"First target is {d0*100:.1f} cm from the current tool position. Make sure the path "
              f"is clear (the hand is on the flange) and keep the e-stop in reach.")
        input("Press ENTER to start moving (Ctrl+C to abort). ")
        for i, T_target in enumerate(T_targets, start=1):
            pose_vec = np.round(np.asarray(m3d.Transform(T_target).pose_vector.array), 4)
            if confirm_each:
                input(f"\nPose {i}/{n}: ENTER to move to {pose_vec} (Ctrl+C to stop). ")
            else:
                print(f"\nPose {i}/{n}: moving to {pose_vec}")
            if q_targets is not None:
                arm.movej(list(np.asarray(q_targets[i - 1], float)), acc=acc, vel=vel, wait=True)
            else:
                arm.set_pose(m3d.Transform(T_target), acc=acc, vel=vel, wait=True, command="movej")

            # settle, keeping the preview live
            t_end = time.monotonic() + settle_s
            img_w, result = None, (False, None, 0, 0, None, None)
            while time.monotonic() < t_end or img_w is None:
                img = wrist_cam.grab_bgr()
                if img is not None:
                    img_w = img
                    result = detect_board_pose(img_w, wrist_cam.K)
                    display = img_w.copy()
                    if result[5] is not None and len(result[5]) > 0:
                        cv2.aruco.drawDetectedMarkers(display, result[4], result[5])
                    cv2.putText(display, f"replay {i}/{n}: {'board OK' if result[0] else 'no board'} "
                                f"({result[3]} tags, {result[2]} pts)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0) if result[0] else (0, 0, 255), 2)
                    cv2.imshow("Wrist Camera", display)
                cv2.waitKey(1)

            found_w, T_camw_board, n_points, n_markers, _mc, _mi = result
            if not found_w:
                print(f"  Board not visible enough here ({n_markers} tags, {n_points} pts) -- skipped.")
                continue
            T_base_ee = get_ee_pose()  # the actual pose reached, not the target
            err_mm = np.linalg.norm(T_base_ee[:3, 3] - T_target[:3, 3]) * 1000
            idx = len(T_base_ee_list) + 1
            cv2.imwrite(os.path.join(session_dir, f"pose{idx:02d}_wrist.png"), img_w)
            T_base_ee_list.append(T_base_ee)
            T_camw_board_list.append(T_camw_board)
            save_poses(session_dir, T_base_ee_list, T_camw_board_list)
            print(f"  Captured ({n_markers} tags, {n_points} pts; reached target within {err_mm:.1f} mm) "
                  f"-> pose {idx}, saved.")
    except KeyboardInterrupt:
        print(f"\nStopped after {len(T_base_ee_list)} captured poses (saved to {session_dir}).")
        arm.stopj(acc)
    finally:
        cv2.destroyAllWindows()
        wrist_cam.close()
    return T_base_ee_list, T_camw_board_list


# ==============================================================================
# CALIBRATION MATH
# ==============================================================================
def compute_and_save(session_dir, T_base_ee_list, T_camw_board_list):
    if len(T_base_ee_list) < MIN_POSES:
        raise RuntimeError(
            f"Only {len(T_base_ee_list)} pose(s) captured; need at least "
            f"{MIN_POSES} for hand-eye calibration."
        )

    # --- Hand-eye calibration (wrist camera <-> tool/end effector) -------------
    R_gripper2base = [T[:3, :3] for T in T_base_ee_list]
    t_gripper2base = [T[:3, 3] for T in T_base_ee_list]
    R_target2cam = [T[:3, :3] for T in T_camw_board_list]
    t_target2cam = [T[:3, 3] for T in T_camw_board_list]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=HAND_EYE_METHOD,
    )
    T_ee_cam_wrist = np.eye(4)
    T_ee_cam_wrist[:3, :3] = R_cam2gripper
    T_ee_cam_wrist[:3, 3] = t_cam2gripper.ravel()

    # --- Static board pose in the base frame, from every pose's wrist-cam chain
    # (Not needed for T_ee_cam_wrist itself -- kept only as a diagnostic: if the
    # board genuinely never moved, T_base_board should agree tightly across poses.)
    T_base_board_samples = [
        T_base_ee @ T_ee_cam_wrist @ T_camw_board
        for T_base_ee, T_camw_board in zip(T_base_ee_list, T_camw_board_list)
    ]
    T_base_board = average_transforms(T_base_board_samples)

    t_mean_err, t_std_err, ang_mean_err, ang_std_err = transform_spread(
        T_base_board_samples, T_base_board)
    print(f"\nT_base_board consistency across {len(T_base_board_samples)} poses: "
          f"translation {t_mean_err*1000:.1f}+/-{t_std_err*1000:.1f} mm, "
          f"rotation {ang_mean_err:.2f}+/-{ang_std_err:.2f} deg "
          f"(large spread => noisy detections or board/robot moved).")

    np.set_printoptions(precision=4, suppress=True)
    print("\nT_ee_cam_wrist (wrist camera frame -> tool/end-effector frame):")
    print(T_ee_cam_wrist)

    output_path = os.path.join(session_dir, OUTPUT_FILENAME)
    np.savez(
        output_path,
        T_ee_cam_wrist=T_ee_cam_wrist,
        T_base_board=T_base_board,
        num_poses=len(T_base_ee_list),
    )
    print(f"\nSaved results to {output_path}")


def main():
    global ROBOT_IP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session_dir", nargs="?", default=None,
        help="An existing captures_ur10e/<timestamp> dir. With no --add, just redoes "
             "the calibration math from its saved poses (no camera/robot needed). "
             "Omit entirely to start a brand-new capture session.",
    )
    parser.add_argument(
        "--add", type=int, default=None, metavar="N",
        help="Capture N more poses and append them to session_dir's existing ones "
             "(continuing the pose numbering), then recompute from all of them. "
             "Requires session_dir.",
    )
    parser.add_argument(
        "--replay", metavar="SESSION_DIR", default=None,
        help="AUTOMATIC capture: start a new session by driving the arm through the poses "
             "recorded in SESSION_DIR/poses.npz (e.g. the last calibration), capturing wherever "
             "the board is detected, then compute the calibration.",
    )
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--hand-delay", type=float, default=5.0,
                        help="LEAP hand stays limp this many seconds at startup (move the fingers out of the "
                             "camera's view), then holds them there for the whole session (default 5s)")
    parser.add_argument("--no-hand", action="store_true", help="don't touch the LEAP hand")
    parser.add_argument("--acc", type=float, default=0.3, help="--replay: joint acceleration (rad/s^2)")
    parser.add_argument("--vel", type=float, default=0.2, help="--replay: joint speed (rad/s)")
    parser.add_argument("--settle", type=float, default=1.5, help="--replay: seconds to settle before capturing")
    parser.add_argument("--confirm-each", action="store_true", help="--replay: press ENTER before every move")
    parser.add_argument("--cartesian", action="store_true",
                        help="--replay: ignore SESSION_DIR/replay_joints_elbow_up.npy and let the controller "
                             "solve IK for each pose (movej to the pose)")
    args = parser.parse_args()

    ROBOT_IP = args.robot_ip
    needs_capture = args.replay is not None or args.add is not None or args.session_dir is None
    if needs_capture and not args.no_hand:
        # same limp-then-hold startup as ../perception/capture_pointclouds.py (shared function)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perception"))
        from capture_pointclouds import hold_hand_after_delay
        _hand = hold_hand_after_delay(args.hand_delay)  # keep a reference: torque stays on until exit

    if args.replay is not None:
        T_targets, _ = load_poses(args.replay)
        joints_path = os.path.join(args.replay, "replay_joints_elbow_up.npy")
        q_targets = None
        if os.path.exists(joints_path) and not args.cartesian:
            q_targets = np.load(joints_path)
            assert len(q_targets) == len(T_targets), f"{joints_path} doesn't match {len(T_targets)} poses"
            print(f"Using offline-solved elbow-up joint targets from {joints_path}")
        session_dir = new_session_dir()
        print(f"Replaying {len(T_targets)} poses from {args.replay}; saving captures to {session_dir}")
        T_base_ee_list, T_camw_board_list = replay_capture(
            session_dir, T_targets, acc=args.acc, vel=args.vel, settle_s=args.settle,
            confirm_each=args.confirm_each, q_targets=q_targets)
    elif args.add is not None:
        if args.session_dir is None:
            parser.error("--add requires a session_dir to append to")
        session_dir = args.session_dir
        T_base_ee_list, T_camw_board_list = load_poses(session_dir)
        start_index = len(T_base_ee_list) + 1
        print(f"Loaded {len(T_base_ee_list)} existing poses from {session_dir}; "
              f"capturing {args.add} more (poses {start_index}-{start_index + args.add - 1}).")
        T_base_ee_list, T_camw_board_list = capture_poses(
            session_dir, num_new_poses=args.add, start_index=start_index,
            T_base_ee_list=T_base_ee_list, T_camw_board_list=T_camw_board_list,
        )
    elif args.session_dir is not None:
        session_dir = args.session_dir
        T_base_ee_list, T_camw_board_list = load_poses(session_dir)
        print(f"Loaded {len(T_base_ee_list)} saved poses from {session_dir}")
    else:
        session_dir = new_session_dir()
        print(f"Saving captures to {session_dir}")
        T_base_ee_list, T_camw_board_list = capture_poses(session_dir)

    compute_and_save(session_dir, T_base_ee_list, T_camw_board_list)


if __name__ == "__main__":
    main()
