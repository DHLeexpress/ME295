"""
Manual multi-view point cloud capture with a wrist-mounted ZED Mini on a UR10e.

Companion to ../calibration/ur10e_hand_eye_calibration.py -- same UX contract
(live preview, no timer, press 'y' to capture, 'q' to stop) and the same
robot-read-only connection to urx. You manually move the arm (e.g. via
freedrive) between captures so the object is seen from several angles.
--replay SESSION_DIR instead drives the arm through a previous session's
viewpoints automatically (the only mode that commands motion).

Each capture saves the depth camera's XYZ point cloud IN THE CAMERA FRAME
(not yet transformed anywhere) plus the robot's T_base_ee at that instant.
Fusing those into one point cloud in the robot base frame -- using a hand-eye
calibration result from ../calibration/ -- and generating a grasp from the
fused cloud is a separate step: fuse_crop_and_grasp.py. That script needs
open3d/trimesh/torch/equidexflow, which live in equidexflow/.venv, not in
this (pyzed-only) conda env, so the two steps are deliberately split.

Run this with the SAME env as the calibration script (needs pyzed + urx):
python capture_pointclouds.py
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "leap_hand_api", "python"))
from leap_hand_utils.dynamixel_client import DynamixelClient  # noqa: E402

# ==============================================================================
# CONFIGURATION
# ==============================================================================
WRIST_SERIAL = 18982088  # same wrist ZED Mini as ur10e_hand_eye_calibration.py

NUM_POSES = 8  # far fewer views needed than for hand-eye calibration -- a
# handful of angles around the object is normally enough for a partial-surface
# grasp reconstruction.
MAX_RANGE_M = 1.2  # drop points farther than this from the camera (noisy long-range depth)
MIN_RANGE_M = 0.05

ROBOT_IP = "192.168.0.2"  # was 172.16.0.20 until 2026-10

CAPTURES_ROOT = "captures_pointcloud"  # each run gets its own captures_pointcloud/<timestamp>/
POSES_FILENAME = "poses.npz"


# ==============================================================================
# ROBOT POSE FEEDBACK (identical contract to ur10e_hand_eye_calibration.py)
# ==============================================================================
_arm = None


def _connect_robot():
    global _arm
    if _arm is None:
        _arm = urx.Robot(ROBOT_IP)
    return _arm


def get_ee_pose():
    """T_base_ee (4x4): p_base_homog = T_base_ee @ p_ee_homog."""
    arm = _connect_robot()
    trans = arm.get_pose()
    return np.array(trans.array, dtype=np.float64)


def get_joint_positions():
    """Actual UR10e joint angles (6,), radians -- saved with every capture so a session can be
    replayed later with plain joint moves (no IK)."""
    return np.asarray(_connect_robot().getj(), dtype=np.float64)


def save_rgb(session_dir, idx, img):
    """Save a view's color image (pose%02d_rgb.png) and refresh the numbered contact sheet views_rgb.png."""
    if img is None:
        return
    cv2.imwrite(os.path.join(session_dir, f"pose{idx:02d}_rgb.png"), img)
    tiles = []
    for f in sorted(p for p in os.listdir(session_dir) if p.startswith("pose") and p.endswith("_rgb.png")):
        t = cv2.resize(cv2.imread(os.path.join(session_dir, f)), (426, 240))
        cv2.putText(t, f"view {int(f[4:6])}", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(os.path.join(session_dir, "views_rgb.png"), np.vstack(rows))


def save_session_poses(session_dir, T_base_ee_list, q_list):
    np.savez(os.path.join(session_dir, POSES_FILENAME), T_base_ee=np.stack(T_base_ee_list), q=np.stack(q_list))


# ==============================================================================
# CAMERA (pyzed, depth enabled -- unlike the calibration script's RGB-only cam)
# ==============================================================================
class ZedCam:
    def __init__(self, serial_number):
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(serial_number)
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = MIN_RANGE_M
        init_params.depth_maximum_distance = MAX_RANGE_M

        self.cam = sl.Camera()
        status = self.cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED {serial_number}: {status}")

        self.runtime_params = sl.RuntimeParameters()
        self.image_mat = sl.Mat()
        self.xyz_mat = sl.Mat()

    def grab_bgr(self):
        if self.cam.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None
        self.cam.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        return cv2.cvtColor(self.image_mat.get_data(), cv2.COLOR_BGRA2BGR)

    def grab_points_cam_frame(self):
        """Grab + return (N, 3) float64 XYZ points in the camera's own frame,
        finite and within [MIN_RANGE_M, MAX_RANGE_M], or None on a failed grab."""
        if self.cam.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None
        self.cam.retrieve_measure(self.xyz_mat, sl.MEASURE.XYZ)
        xyz = self.xyz_mat.get_data()[:, :, :3].reshape(-1, 3).astype(np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[(r >= MIN_RANGE_M) & (r <= MAX_RANGE_M)]
        return xyz

    def close(self):
        self.cam.close()


# ==============================================================================
# LEAP HAND: hold the fingers out of the camera's way during the scan
# ==============================================================================
HAND_MOTORS = list(range(16))


def hold_hand_after_delay(delay_s):
    """Connect to the LEAP hand with torque OFF (fingers limp, so you can move them out of the
    camera's view), count down delay_s, then enable torque HOLDING wherever the joints are: the
    goal position is set to the measured position before torque comes on, so nothing moves.
    Uses the same position-current mode / gains as the LEAP API's LeapNode, without its
    snap-to-open on connect. Torque turns off again when this process exits (the Dynamixel
    client's own exit cleanup), so the fingers relax after the scan."""
    client = None
    for port in ("/dev/ttyUSB0", "/dev/ttyUSB1"):
        try:
            client = DynamixelClient(HAND_MOTORS, port, 4000000)
            client.connect()
            break
        except Exception:
            client = None
    if client is None:
        raise RuntimeError("LEAP hand not found on /dev/ttyUSB0 or /dev/ttyUSB1 (pass --no-hand to scan without it)")
    client.set_torque_enabled(HAND_MOTORS, False)
    client.sync_write(HAND_MOTORS, np.ones(16) * 5, 11, 1)  # operating mode: current-based position
    print(f"[hand] connected, torque OFF -- move the fingers out of the camera's view; holding them in {delay_s:.0f}s")
    t_end = time.monotonic() + delay_s
    while (left := t_end - time.monotonic()) > 0:
        print(f"[hand]   {left:.0f}s...", flush=True)
        time.sleep(min(1.0, left))
    present = np.asarray(client.read_pos(), dtype=float)
    client.write_desired_pos(HAND_MOTORS, present)  # goal = where the fingers are, BEFORE torque on
    side = [0, 4, 8]
    client.sync_write(HAND_MOTORS, np.ones(16) * 600, 84, 2)   # P gain (LeapNode defaults)
    client.sync_write(side, np.ones(3) * 450, 84, 2)
    client.sync_write(HAND_MOTORS, np.zeros(16), 82, 2)        # I gain
    client.sync_write(HAND_MOTORS, np.ones(16) * 200, 80, 2)   # D gain
    client.sync_write(side, np.ones(3) * 150, 80, 2)
    client.sync_write(HAND_MOTORS, np.ones(16) * 350, 102, 2)  # current limit: a gentle hold
    client.sync_write(HAND_MOTORS, np.ones(16) * 15, 108, 4)   # profile acceleration
    client.sync_write(HAND_MOTORS, np.ones(16) * 60, 112, 4)   # profile velocity
    client.set_torque_enabled(HAND_MOTORS, True)
    print("[hand] torque ON -- holding the fingers where they are for the scan")
    return client


# ==============================================================================
# CAPTURE LOOP
# ==============================================================================
def wait_for_capture_keypress(cam, pose_idx, total_poses):
    """Live preview, no timer. Returns (points_cam, T_base_ee, img_bgr) on 'y', or None on 'q'."""
    print(f"\n=== Pose {pose_idx}/{total_poses}: reposition the robot so a NEW part of the "
          f"object is visible, then press 'y' to capture (or 'q' to stop early). ===")
    last_img = None
    while True:
        img = cam.grab_bgr()
        if img is not None:
            last_img = img
            display = img.copy()
            cv2.putText(display, f"view {pose_idx}/{total_poses}: 'y'=capture, 'q'=stop", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Wrist Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            return None
        if key in (ord("y"), ord("Y")):
            points_cam = cam.grab_points_cam_frame()
            if points_cam is None or len(points_cam) == 0:
                print(" Camera grab failed or returned no valid depth just now, try again.")
                continue
            T_base_ee = get_ee_pose()
            return points_cam, T_base_ee, last_img


def replay_session(session_dir, source_dir, acc=0.3, vel=0.2, settle_s=1.5, confirm_each=False):
    """Automatic capture: drive the arm through SOURCE_DIR's viewpoints with plain joint movej's
    and capture at each. Joint targets: SOURCE_DIR/poses.npz's recorded joints ('q', saved since
    2026-10-09) if present, else SOURCE_DIR/replay_joints_elbow_up.npy (offline elbow-up IK from
    ../calibration/solve_replay_joints.py). The ACTUAL pose/joints after each move are recorded.
    The tool poses are only meaningful while the camera mount (hand-eye calibration) is unchanged.
    Ctrl+C stops the arm and keeps what was captured."""
    src = np.load(os.path.join(source_dir, POSES_FILENAME))
    if "q" in src.files:
        Q, how = src["q"], "recorded joints"
    else:
        path = os.path.join(source_dir, "replay_joints_elbow_up.npy")
        if not os.path.exists(path):
            raise SystemExit(f"{source_dir} has no recorded joints; run "
                             f"../calibration/solve_replay_joints.py {source_dir} first (leap_retarget env)")
        Q, how = np.load(path), "offline elbow-up IK joints"
    arm = _connect_robot()
    cam = ZedCam(WRIST_SERIAL)
    T_base_ee_list, q_list = [], []
    n = len(Q)
    try:
        dq0 = np.degrees(np.abs(Q[0] - get_joint_positions()).max())
        print(f"\nReplaying {n} viewpoints from {source_dir} ({how}) at acc={acc}, vel={vel}. "
              f"Largest joint move to the first one: {dq0:.0f} deg. Keep the e-stop in reach.")
        input("Press ENTER to start moving (Ctrl+C to abort). ")
        for i, q in enumerate(Q, start=1):
            if confirm_each:
                input(f"\nView {i}/{n}: ENTER to move to {np.round(np.degrees(q)).astype(int).tolist()} deg. ")
            else:
                print(f"\nView {i}/{n}: moving")
            arm.movej(list(np.asarray(q, float)), acc=acc, vel=vel, wait=True)
            t_end = time.monotonic() + settle_s
            last_img = None
            while time.monotonic() < t_end:  # settle, keeping the preview live
                img = cam.grab_bgr()
                if img is not None:
                    last_img = img
                    cv2.imshow("Wrist Camera", img)
                cv2.waitKey(1)
            points_cam = cam.grab_points_cam_frame()
            if points_cam is None or len(points_cam) == 0:
                print("  no valid depth here -- skipped")
                continue
            idx = len(T_base_ee_list) + 1
            np.save(os.path.join(session_dir, f"pose{idx:02d}_points_cam.npy"), points_cam.astype(np.float32))
            save_rgb(session_dir, idx, last_img)
            T_base_ee_list.append(get_ee_pose())
            q_list.append(get_joint_positions())
            save_session_poses(session_dir, T_base_ee_list, q_list)
            print(f"  captured {len(points_cam)} points -> pose {idx}")
    except KeyboardInterrupt:
        print(f"\nStopped after {len(T_base_ee_list)} captures (saved to {session_dir}).")
        arm.stopj(acc)
    finally:
        cv2.destroyAllWindows()
        cam.close()
    return session_dir, len(T_base_ee_list)


def new_session_dir():
    session_dir = os.path.join(CAPTURES_ROOT, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def capture_session(session_dir, num_poses):
    cam = ZedCam(WRIST_SERIAL)
    T_base_ee_list, q_list = [], []
    pose_idx = 0
    try:
        while pose_idx < num_poses:
            pose_idx += 1
            result = wait_for_capture_keypress(cam, pose_idx, num_poses)
            if result is None:
                print(f"\nStopped by 'q' after {len(T_base_ee_list)} total poses (saved to {session_dir}).")
                break
            points_cam, T_base_ee, img = result
            save_rgb(session_dir, pose_idx, img)
            np.save(os.path.join(session_dir, f"pose{pose_idx:02d}_points_cam.npy"),
                    points_cam.astype(np.float32))
            T_base_ee_list.append(T_base_ee)
            q_list.append(get_joint_positions())
            save_session_poses(session_dir, T_base_ee_list, q_list)
            print(f"  Captured pose {pose_idx}: {len(points_cam)} points, EE pose recorded "
                  f"and saved to {session_dir}.")
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(T_base_ee_list)} total poses (saved to {session_dir}).")
    finally:
        cv2.destroyAllWindows()
        cam.close()
    return session_dir, len(T_base_ee_list)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-poses", type=int, default=NUM_POSES,
                         help=f"number of viewpoints to capture (default {NUM_POSES})")
    parser.add_argument("--hand-delay", type=float, default=5.0,
                         help="seconds the LEAP hand stays limp at startup (move the fingers out of the "
                              "camera's view) before it holds them there for the scan (default 5s)")
    parser.add_argument("--no-hand", action="store_true", help="don't touch the LEAP hand")
    parser.add_argument("--replay", metavar="SESSION_DIR", default=None,
                         help="AUTOMATIC capture: drive the arm through SESSION_DIR's viewpoints and capture at each")
    parser.add_argument("--acc", type=float, default=0.3, help="--replay: joint acceleration (rad/s^2)")
    parser.add_argument("--vel", type=float, default=0.2, help="--replay: joint speed (rad/s)")
    parser.add_argument("--settle", type=float, default=1.5, help="--replay: seconds to settle before capturing")
    parser.add_argument("--confirm-each", action="store_true", help="--replay: press ENTER before every move")
    args = parser.parse_args()

    hand = None if args.no_hand else hold_hand_after_delay(args.hand_delay)

    session_dir = new_session_dir()
    print(f"Saving captures to {session_dir}")
    if args.replay:
        session_dir, n = replay_session(session_dir, args.replay, acc=args.acc, vel=args.vel,
                                        settle_s=args.settle, confirm_each=args.confirm_each)
    else:
        session_dir, n = capture_session(session_dir, args.num_poses)
    if n == 0:
        print("No poses captured -- nothing saved.")
        return
    print(f"\nDone: {n} pose(s) saved to {session_dir}. Next, run fuse_crop_and_grasp.py "
          f"(in the equidexflow venv) with --capture-dir {session_dir}")


if __name__ == "__main__":
    main()
