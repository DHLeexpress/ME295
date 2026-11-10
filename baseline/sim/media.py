"""High-resolution views of an episode for figures and videos (never shown to the policy).

The policy sees 224 x 224 `ext_cam` / `wrist_cam` images; these extra views use MuJoCo free cameras aimed at the
designated book, so the scene model (and therefore the policy's input) is unchanged. Frames are streamed to one MP4
per view, so a 40 s episode does not sit in memory.

    rec = MediaRecorder(env, "baseline/data/media_runs/pi05_ft/random_005")   # -> random_005_{close,top,wide}.mp4
    ... rec.capture() after every control step ...
    rec.close()
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

# Free cameras relative to the designated book's centre (x right, y into the shelf, z up; azimuth/elevation in deg)
VIEWS = {
    "close": dict(offset=(0.0, -0.04, 0.05), distance=0.62, azimuth=120.0, elevation=-28.0),  # 3/4 front-left
    "top":   dict(offset=(0.0, -0.03, 0.10), distance=0.42, azimuth=90.0, elevation=-72.0),   # gaps from above
    "wide":  dict(offset=(0.0, -0.25, 0.12), distance=1.70, azimuth=135.0, elevation=-22.0),  # whole cell
}


def free_camera(center: np.ndarray, view: dict) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(center, float) + np.asarray(view["offset"], float)
    cam.distance, cam.azimuth, cam.elevation = view["distance"], view["azimuth"], view["elevation"]
    return cam


class MediaRecorder:
    def __init__(self, env, prefix: str | Path, views=("close", "top", "wide"), size=(960, 720), fps: float = 10.0):
        import imageio
        w, h = size
        env.m.vis.global_.offwidth = max(env.m.vis.global_.offwidth, w)
        env.m.vis.global_.offheight = max(env.m.vis.global_.offheight, h)
        self.env = env
        self.renderer = mujoco.Renderer(env.m, h, w)
        self.cams = {v: free_camera(env.info.center, VIEWS[v]) for v in views}
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.paths = {v: prefix.with_name(f"{prefix.name}_{v}.mp4") for v in views}
        self.writers = {v: imageio.get_writer(self.paths[v], fps=fps, quality=8, macro_block_size=8) for v in views}
        self.n = 0

    def capture(self):
        for v, cam in self.cams.items():
            self.renderer.update_scene(self.env.rob.d, camera=cam)
            self.writers[v].append_data(self.renderer.render())
        self.n += 1

    def close(self):
        for wr in self.writers.values():
            wr.close()
        self.renderer.close()
