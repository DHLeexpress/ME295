"""Offscreen rendering of a MuJoCo scene: Filament (physically based, MuJoCo >= 3.17) or classic OpenGL.

    r = SceneRenderer(model)                  # backend "auto": Filament via Vulkan, else classic
    rgb = r.render(data, "cam_main", (1280, 720), contacts=True)
    rgb = r.render(data, free_camera, (640, 480))   # mujoco.MjvCamera also works

Filament gives image-based lighting, soft shadows and MSAA; it needs Vulkan (Linux/Windows with a
GPU driver) and runs headless. Its decorations are not reachable from Python yet, so contact
markers and extra geoms (force arrows) come from a classic overlay pass drawn on top. The classic renderer works everywhere MuJoCo does (headless Linux:
MUJOCO_GL=egl, CPU-only: MUJOCO_GL=osmesa; macOS/Windows: default).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import mujoco
import numpy as np

Camera = Union[str, int, mujoco.MjvCamera]


def _as_mjv_camera(m: mujoco.MjModel, cam: Camera) -> mujoco.MjvCamera:
    if isinstance(cam, mujoco.MjvCamera):
        return cam
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FIXED
    c.fixedcamid = m.camera(cam).id if isinstance(cam, str) else int(cam)
    return c


def contact_option(contacts: bool) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    if contacts:
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    return opt


class SceneRenderer:
    def __init__(self, m: mujoco.MjModel, backend: str = "auto", msaa: int = 4):
        self.m = m
        self.backend = None
        self._targets: Dict[Tuple[int, int], np.ndarray] = {}
        if backend in ("auto", "filament"):
            try:
                self._init_filament(msaa)
                self.backend = "filament"
            except Exception as exc:  # no Vulkan / old MuJoCo: fall back
                if backend == "filament":
                    raise
                print(f"[render] Filament unavailable ({exc.__class__.__name__}: {exc}); using the classic renderer")
        if self.backend is None:
            self._classic: Dict[Tuple[int, int], mujoco.Renderer] = {}
            self.backend = "classic"

    # ------------------------------------------------------------ Filament
    def _init_filament(self, msaa: int):
        from mujoco import mjrf
        from mujoco.experimental.studio import window  # noqa: F401  (registers Filament resources)
        from mujoco.rendering.filament import model_scene, renderer

        self._mjrf, self._view = mjrf, renderer.View
        self._ctx = mjrf.Context(mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_VULKAN))
        self._fr = renderer.Renderer(self._ctx)
        self._scene = model_scene.ModelScene(self._ctx, self.m, self._fr.scene("world"))
        self._scene.scene.set_msaa(msaa)
        self._overlay: Dict[Tuple[int, int], mujoco.Renderer] = {}

    def _render_filament(self, d, cam, size, contacts, extra_geoms):
        w, h = size
        key = (w, h)
        if key not in self._targets:
            self._targets[key] = np.zeros((h, w, 3), np.uint8)
            self._fr.target(f"t{w}x{h}", self._targets[key])
        mcam = _as_mjv_camera(self.m, cam)
        self._scene.update(d)
        glcam = mujoco.mjv_camera2GLCamera(self.m, d, mcam)
        self._fr.render([self._view(scene="world", target=f"t{w}x{h}", camera=glcam)])
        img = self._targets[key].copy()
        if contacts or extra_geoms:
            over = self._overlay_pass(d, mcam, size, contacts, extra_geoms)
            mask = over.max(axis=2) > 8
            img[mask] = over[mask]
        return img

    def _overlay_pass(self, d, mcam, size, contacts, extra_geoms):
        """Decorations only (no geoms, no skybox) on black, for compositing over Filament."""
        w, h = size
        if (w, h) not in self._overlay:
            self._overlay[(w, h)] = mujoco.Renderer(self.m, h, w)
        r = self._overlay[(w, h)]
        opt = contact_option(contacts)
        opt.geomgroup[:] = 0
        opt.sitegroup[:] = 0
        r.update_scene(d, camera=mcam, scene_option=opt)
        for flag in (mujoco.mjtRndFlag.mjRND_SKYBOX, mujoco.mjtRndFlag.mjRND_HAZE, mujoco.mjtRndFlag.mjRND_FOG,
                     mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_SHADOW):
            r.scene.flags[flag] = 0
        _add_geoms(r.scene, extra_geoms)
        return r.render()

    # ------------------------------------------------------------ classic
    def _render_classic(self, d, cam, size, contacts, extra_geoms):
        w, h = size
        if (w, h) not in self._classic:
            self._classic[(w, h)] = mujoco.Renderer(self.m, h, w)
        r = self._classic[(w, h)]
        r.update_scene(d, camera=_as_mjv_camera(self.m, cam), scene_option=contact_option(contacts))
        _add_geoms(r.scene, extra_geoms)
        return r.render()

    def render(self, d: mujoco.MjData, cam: Camera, size: Tuple[int, int], contacts: bool = False,
               extra_geoms: Optional[list] = None) -> np.ndarray:
        if self.backend == "filament":
            return self._render_filament(d, cam, size, contacts, extra_geoms)
        return self._render_classic(d, cam, size, contacts, extra_geoms)

    def close(self):
        for r in list(getattr(self, "_classic", {}).values()) + list(getattr(self, "_overlay", {}).values()):
            r.close()


def arrow(start, vec, rgba, width: float = 0.004) -> tuple:
    """An arrow from `start` along `vec` (world, m), for SceneRenderer.render(extra_geoms=[...])."""
    start = np.asarray(start, float)
    return ("arrow", start, start + np.asarray(vec, float), np.asarray(rgba, np.float32), float(width))


def _add_geoms(scn: mujoco.MjvScene, specs) -> None:
    for kind, a, b, rgba, width in specs or []:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]  # initialise the scene's own slot in place
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, a, b)
        g.emission = 0.5
        scn.ngeom += 1
