"""Look at the shelf with the wrist ZED and decide which book to take: the step that replaces the task's hard-coded
book pose. Robot calls are python-urx (movel with the robot's TCP), so the same function runs on SimRobot and on the
real UR10e; the camera is any callable returning an RGBDFrame (SimZedCamera.grab, ZedCamera.grab).

    target, est, frame = look_and_pick(rob, grab, wrist, tcp, colour="dark blue")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .camera import RGBDFrame
from .look import look_pose
from .shelf import Book, Params, PerceptionError, ShelfEstimate, ShelfPrior, estimate_shelf
from .wrist import WristCamera, flange_pose


@dataclass
class BookTarget:
    """What the librarian task needs to know about the book (base frame, metres)."""
    x: float                 # centre of the spine across the row
    z: float                 # grasp height (mid-height of the spine)
    spine_y: float           # spine plane
    thickness: float
    gap_left: Optional[float] = None
    gap_right: Optional[float] = None
    colour: str = ""
    source: str = "constants"

    @classmethod
    def from_estimate(cls, book: Book, est: ShelfEstimate, max_yaw_deg: float = 3.0) -> "BookTarget":
        if abs(est.yaw_deg) > max_yaw_deg:
            raise PerceptionError(f"shelf turned {est.yaw_deg:+.1f} deg: the task approaches along +y only "
                                  f"(limit {max_yaw_deg} deg)")
        return cls(x=float(book.center[0]), z=float(book.center[2]), spine_y=float(book.center[1]),   # the plane at the book
                   thickness=float(book.thickness), gap_left=book.gap_left, gap_right=book.gap_right,
                   colour=book.colour, source="vision")

    def describe(self) -> str:
        g = lambda v: "?" if v is None else f"{1e3 * v:.1f}"
        return (f"{self.colour or 'book'} at x={self.x:+.4f} z={self.z:.4f}, spines y={self.spine_y:.4f}, "
                f"{1e3 * self.thickness:.1f} mm thick, gaps {g(self.gap_left)} | {g(self.gap_right)} mm ({self.source})")


def look_and_pick(rob, grab: Callable[[], RGBDFrame], wrist: WristCamera, tcp: Sequence[float],
                  colour: Optional[str] = None, aim=(0.0, 0.82, 0.53), distance: float = 0.45,
                  prior: ShelfPrior = ShelfPrior(), params: Params = Params(), acc: float = 0.5, vel: float = 0.15,
                  settle: float = 0.5, out: Optional[Path] = None, say=print):
    """Move to the look pose aimed at `aim`, grab one frame, estimate the shelf and pick the book ("the {colour}
    book", or the one with the widest gaps). Returns (BookTarget, ShelfEstimate, RGBDFrame); raises PerceptionError
    if no book can be taken. The frame's camera pose comes from the robot's flange pose and the hand-eye
    calibration (the simulator's own camera pose agrees to numerical precision)."""
    from shear_agent.robot import mat_to_pose, pose_to_mat
    T = look_pose(wrist, aim, into=prior.into, distance=distance)
    say(f"look at the shelf: camera {distance:.2f} m from {np.round(aim, 3).tolist()}")
    rob.movel([float(v) for v in mat_to_pose(T @ pose_to_mat(tcp))], acc=acc, vel=vel)
    rob.sleep(settle)
    frame = grab().with_pose(wrist.T_base_cam(flange_pose(rob, tcp)))
    est = estimate_shelf(frame, prior, params)
    say(est.describe())
    book = est.select(colour)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        frame.save(out / "look_frame.npz")
        try:
            import cv2
            from .camera import depth_colormap
            from .shelf import draw
            img = np.concatenate([draw(frame, est, book), depth_colormap(frame.depth, 0.2, 1.0)], 1)
            cv2.imwrite(str(out / "look.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        except ImportError:
            pass
    if book is None:
        raise PerceptionError(f"no book with air on both sides{f' matching {colour!r}' if colour else ''}")
    target = BookTarget.from_estimate(book, est)
    say(f"target: {target.describe()}")
    return target, est, frame
