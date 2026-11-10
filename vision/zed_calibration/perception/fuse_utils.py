"""Multi-view point-cloud fusion + object crop, shared by every grasp pipeline.

Only numpy (+ open3d inside prepare_points_and_normals, imported there), so it runs in any env that has open3d (the equidexflow venv for
fuse_crop_and_grasp.py, the springgrasp conda env for ../springgrasp/fuse_for_springgrasp.py).
Moved here verbatim from fuse_crop_and_grasp.py (and clean_cropped_cloud.py's distance-from-median
filter) so both pipelines use the exact same fusion/crop instead of two copies drifting apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CALIB_ROOT = Path(__file__).resolve().parent.parent / "calibration" / "captures_ur10e"


ACTIVE_CALIBRATION_FILE = CALIB_ROOT.parent / "active_calibration.txt"


def newest_calibration() -> Path:
    """The default hand-eye result everywhere: the one pinned in ../calibration/active_calibration.txt
    (a path relative to the repo root, edit it to switch), else the newest under captures_ur10e/."""
    if ACTIVE_CALIBRATION_FILE.exists():
        rel = ACTIVE_CALIBRATION_FILE.read_text().strip()
        path = (CALIB_ROOT.parent.parent / rel).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{ACTIVE_CALIBRATION_FILE} points to {path}, which doesn't exist")
        return path
    candidates = sorted(CALIB_ROOT.glob("*/hand_eye_calibration_ur10e_result.npz"))
    if not candidates:
        raise FileNotFoundError(f"no calibration result found under {CALIB_ROOT}")
    return candidates[-1]


def load_calibration(calib_path: Path) -> np.ndarray:
    data = np.load(calib_path)
    return np.asarray(data["T_ee_cam_wrist"], dtype=np.float64)


def load_capture_session(capture_dir: Path, views=None):
    """Return list of (points_cam (N,3) float64, T_base_ee (4,4) float64).
    views: optional 1-based view numbers to keep (e.g. [2, 5, 7]); default all."""
    poses = np.load(capture_dir / "poses.npz")
    T_base_ee_list = list(poses["T_base_ee"])
    if views is not None:
        bad = [v for v in views if not 1 <= v <= len(T_base_ee_list)]
        if bad:
            raise ValueError(f"views {bad} out of range 1..{len(T_base_ee_list)}")
    out = []
    for i in (views if views is not None else range(1, len(T_base_ee_list) + 1)):
        pts = np.load(capture_dir / f"pose{i:02d}_points_cam.npy").astype(np.float64)
        out.append((pts, T_base_ee_list[i - 1]))
    return out


def transform_points(points_cam: np.ndarray, T: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    return (homog @ T.T)[:, :3]


def fuse_to_base_frame(views, T_ee_cam_wrist: np.ndarray) -> np.ndarray:
    fused = []
    for points_cam, T_base_ee in views:
        T_base_cam = T_base_ee @ T_ee_cam_wrist
        fused.append(transform_points(points_cam, T_base_cam))
    return np.concatenate(fused, axis=0)


# Fixed robot-base-frame workspace box -- NOT derived from the cloud itself.
# Centroid-based auto-centering pulled in whatever was above z-min (robot
# links, mount hardware, background clutter), not just the object, so the
# crop is hardcoded to the known object-placement region instead: Z above
# the table, |X| within reach in front of the base, and Y in one of the two
# bands to either side of the base (never the -25..25cm strip straddling the
# base centerline itself, which is where the robot/mount sit).
WORKSPACE_Z_MIN = 0.02
WORKSPACE_X_RANGE = (-0.20, 0.20)
WORKSPACE_Y_BANDS = [(0.25, 0.75), (-0.75, -0.25)]


def crop_object(points_base: np.ndarray, z_min: float = WORKSPACE_Z_MIN):
    """Keep only points inside the hardcoded workspace box (see above). z_min overrides the floor
    (default WORKSPACE_Z_MIN = 2cm; lower it to keep more of a low object's base, at the risk of table points)."""
    x, y, z = points_base[:, 0], points_base[:, 1], points_base[:, 2]
    keep = z >= z_min
    keep &= (x >= WORKSPACE_X_RANGE[0]) & (x <= WORKSPACE_X_RANGE[1])
    y_keep = np.zeros_like(keep)
    for lo, hi in WORKSPACE_Y_BANDS:
        y_keep |= (y >= lo) & (y <= hi)
    keep &= y_keep

    cropped = points_base[keep]
    if len(cropped) < 200:
        above_z = points_base[z >= z_min]
        bbox = (f"x[{above_z[:,0].min():.3f},{above_z[:,0].max():.3f}] "
                f"y[{above_z[:,1].min():.3f},{above_z[:,1].max():.3f}]" if len(above_z) else "(no points above z-min)")
        raise RuntimeError(
            f"Only {len(cropped)} points survive the hardcoded workspace box "
            f"(x in {WORKSPACE_X_RANGE}, y in {WORKSPACE_Y_BANDS}, z >= {z_min}) -- "
            f"the object isn't where WORKSPACE_X_RANGE/WORKSPACE_Y_BANDS expect it. "
            f"Post-Z-crop cloud bbox: {bbox}"
        )
    return cropped


def prepare_points_and_normals(points: np.ndarray, voxel_size: float, min_cluster_points: int = 10):
    """Denoise/downsample the cropped cloud and estimate outward normals.
    Returns (points (N,3) float32, normals (N,3) float32, the open3d cloud)."""
    import open3d as o3d  # only this step needs open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) < 50:
        raise RuntimeError(f"Only {len(pcd.points)} points left after downsampling to "
                            f"voxel_size={voxel_size} -- use a smaller --voxel-size.")

    # SOR flags points by their OWN neighbor-distance stats, so a tiny 2-3
    # point speckle (multi-view registration ghosting, mostly) reads as
    # locally dense among itself and slips through. Drop anything that isn't
    # part of a spatially-connected cluster of at least min_cluster_points,
    # using the same radius as the normal-estimation KNN below.
    labels = np.array(pcd.cluster_dbscan(eps=voxel_size * 3, min_points=min_cluster_points))
    keep = labels >= 0
    if keep.sum() < 50:
        raise RuntimeError(f"Only {keep.sum()} points left after cluster filtering "
                            f"(min_cluster_points={min_cluster_points}) -- lower --min-cluster-points.")
    pcd = pcd.select_by_index(np.where(keep)[0])

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(15)

    pts = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    return pts, nrm, pcd


def drop_far_points(points: np.ndarray, radius: float = 0.15):
    """clean_cropped_cloud.py's filter: drop points farther than radius (m) from the cloud's
    median point (stray background slivers that survive the workspace box). Returns (kept, mask)."""
    median = np.median(points, axis=0)
    keep = np.linalg.norm(points - median, axis=1) <= radius
    return points[keep], keep
