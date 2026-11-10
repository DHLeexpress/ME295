"""The wrist ZED Mini: where it sits on the flange (hand-eye calibration) and its intrinsics.

Source of truth is the calibration bundle vision/zed_calibration (README_camera_extrinsic.md there; override with
$ME295_ZED_DIR): its calibration/active_calibration.txt names the npz whose `T_ee_cam_wrist` is the left camera's
pose in the UR tool0 (flange) frame, solved with urx get_pose() at TCP = flange. vision/zed_mini_wrist.json is a
snapshot of that result plus the camera's intrinsics (refresh: vision/scripts/zed_check.py --snapshot).

    wrist = WristCamera.active()                      # bundle if found (warns if the snapshot is stale), else snapshot
    T_base_cam = wrist.T_base_cam(flange_pose(rob))   # rob: SimRobot or urx.Robot (TCP accounted for)

Frames: base = UR controller base frame (= the MuJoCo world, checked against the bundle's recorded poses to
0.02 mm); flange = tool0; cam = ZED left camera, IMAGE coordinates (x right, y down, z forward).
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .camera import Intrinsics

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).resolve().parent / "zed_mini_wrist.json"


def default_zed_dir() -> Path:
    """The calibration bundle: $ME295_ZED_DIR, else vision/zed_calibration."""
    return Path(os.environ.get("ME295_ZED_DIR", Path(__file__).resolve().parent / "zed_calibration"))


def _rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


@dataclass
class WristCamera:
    T_flange_cam: np.ndarray
    intr: Intrinsics
    serial: int = 0
    model: str = "ZED-M"
    source: str = ""
    notes: dict = field(default_factory=dict)

    def T_base_cam(self, T_base_flange: np.ndarray) -> np.ndarray:
        return np.asarray(T_base_flange, float) @ self.T_flange_cam

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path | str = SNAPSHOT) -> "WristCamera":
        d = json.loads(Path(path).read_text())
        return cls(T_flange_cam=np.array(d["T_flange_cam"], float), intr=Intrinsics(**d["intrinsics"]),
                   serial=int(d.get("serial", 0)), model=d.get("model", "ZED-M"), source=d.get("source", str(path)),
                   notes=d.get("notes", {}))

    def save(self, path: Path | str = SNAPSHOT) -> None:
        d = {"model": self.model, "serial": self.serial, "source": self.source,
             "frames": "T_flange_cam: p_flange = T @ p_cam; flange = UR tool0; cam = ZED left, IMAGE coords "
                       "(x right, y down, z forward)",
             "T_flange_cam": np.round(self.T_flange_cam, 6).tolist(), "intrinsics": self.intr.to_dict(),
             "notes": self.notes}
        Path(path).write_text(json.dumps(d, indent=2) + "\n")

    @classmethod
    def from_zed_bundle(cls, zed_dir: Optional[Path | str] = None, intr: Optional[Intrinsics] = None) -> "WristCamera":
        """The active calibration of the ../ZED bundle (intrinsics: given, else the snapshot's)."""
        zed_dir = Path(zed_dir) if zed_dir else default_zed_dir()
        pointer = zed_dir / "calibration" / "active_calibration.txt"
        if pointer.exists():
            npz = zed_dir / pointer.read_text().strip()
        else:
            found = sorted((zed_dir / "calibration" / "captures_ur10e").glob("*/hand_eye_calibration_ur10e_result.npz"))
            if not found:
                raise FileNotFoundError(f"no hand-eye calibration under {zed_dir}")
            npz = found[-1]
        data = np.load(npz)
        snap = cls.load() if SNAPSHOT.exists() else None
        intr = intr or (snap.intr if snap else None)
        if intr is None:
            raise ValueError("no intrinsics: pass intr= or keep vision/zed_mini_wrist.json")
        return cls(T_flange_cam=np.asarray(data["T_ee_cam_wrist"], float), intr=intr,
                   serial=snap.serial if snap else 0, source=str(npz.relative_to(zed_dir.parent)
                                                                   if zed_dir.parent in npz.parents else npz),
                   notes={"num_poses": int(data["num_poses"]) if "num_poses" in data.files else None})

    @classmethod
    def active(cls, zed_dir: Optional[Path | str] = None, warn: bool = True) -> "WristCamera":
        """The bundle's active calibration if the bundle is there, else the snapshot. Warns when they differ by
        more than 1 mm / 0.1 deg (the snapshot is stale: re-run vision/scripts/zed_check.py --snapshot)."""
        try:
            live = cls.from_zed_bundle(zed_dir)
        except (FileNotFoundError, OSError):
            return cls.load()
        if warn and SNAPSHOT.exists():
            snap = cls.load()
            dp = 1e3 * np.linalg.norm(live.T_flange_cam[:3, 3] - snap.T_flange_cam[:3, 3])
            dr = _rot_deg(live.T_flange_cam[:3, :3], snap.T_flange_cam[:3, :3])
            if dp > 1.0 or dr > 0.1:
                warnings.warn(f"vision/zed_mini_wrist.json differs from the active calibration {live.source} by "
                              f"{dp:.1f} mm / {dr:.2f} deg; using the active one (update the snapshot)")
        return live


def flange_pose(rob, tcp=None) -> np.ndarray:
    """T_base_flange (4x4) of a SimRobot or a python-urx Robot. urx's get_pose() is the TCP's pose, so the TCP set
    on the robot is taken off (real_ur.connect records it as rob.tcp; pass tcp= otherwise)."""
    if hasattr(rob, "flange_pose"):
        return rob.flange_pose()
    from shear_agent.robot import pose_to_mat
    tcp = tcp if tcp is not None else getattr(rob, "tcp", None)
    if tcp is None:
        raise ValueError("unknown TCP on this robot: pass tcp= (the value given to set_tcp)")
    T_tcp = np.array(rob.get_pose().array, dtype=float)
    return T_tcp @ np.linalg.inv(pose_to_mat(tcp))
