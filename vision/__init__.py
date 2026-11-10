"""Vision for ME295: the wrist ZED Mini (real or simulated) -> calibrated RGB-D in the robot base frame -> what to
grasp, and the controllers that grasp it. See vision/README.md.

    camera.py      Intrinsics, RGBDFrame (shared by the real and the simulated camera)
    wrist.py       WristCamera: hand-eye calibration (vision/zed_calibration), flange_pose(rob)
    zed.py         the real ZED Mini through the ZED SDK (pyzed); raw UVC fallback without the SDK
    sim_zed.py     the calibrated ZED Mini in MuJoCo (add_wrist_zed, SimZedCamera)
    stream.py      real-time calibrated stream: frames + the robot pose at each frame (RTDE / urx / sim)
    look.py        view poses (shelf look pose, top-down view)
    shelf.py, pipeline.py   bookshelf perception and the librarian task's look step
    tabletop.py    table plane, object segmentation, top-down parallel-jaw grasp
    safety.py      table / workspace safety filter (joint paths + velocity commands)
    controllers/   grasp controllers for the ablation (scripted, PBVS, MPC)
    sim/           tabletop scenes for the ablation
    zed_calibration/   the original ZED hand-eye calibration bundle (scripts + results)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from .camera import Intrinsics, RGBDFrame, depth_colormap, transform_points  # noqa: E402
from .wrist import WristCamera, flange_pose  # noqa: E402

__all__ = ["Intrinsics", "RGBDFrame", "WristCamera", "depth_colormap", "flange_pose", "transform_points"]
