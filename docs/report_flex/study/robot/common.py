"""Shared helpers for the robot-mount study (MuJoCo 3.17.0, python bindings).

Loads copies of the scratch assemblies in ./asm/<grid>/ (mesh paths resolve through ./asm/model -> repo model/),
applies option/material edits through MjSpec, and steps with divergence detection by the mjWARN_BADQACC counter
(MuJoCo silently resets the state after a NaN/huge qacc and keeps stepping).
"""
import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ASM = HERE / "asm"
REPO = Path("/home/dohyun/projects/ME295")
sys.path.insert(0, str(REPO / "scripts_sim"))
sys.path.insert(0, str(REPO / "src"))
W = mujoco.mjtWarning
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def f1_inputs(E, nu):
    """F1 workaround: (young, poisson) to pass to a dof="full" flex so that the effective Lame
    parameters are those of (E, nu): poisson_in = nu/(2(1-nu)), young_in = 2 G (1 + poisson_in)."""
    G = E / (2 * (1 + nu))
    p = nu / (2 * (1 - nu))
    return 2 * G * (1 + p), p


def scene_path(grid="11x15x3", scene="scene.xml"):
    return ASM / grid / scene


def make_spec(grid="11x15x3", o=None):
    """o: dict of edits. Keys:
      E, nu, f1 (bool: apply F1 correction), h, solver ('CG'/'Newton'), it, tol, ls_it, impratio, cone,
      noslip, memory (bytes), pdms_mass (parent body mass), flexdamp (Rayleigh), grav (bool),
      box_half, table (bool), lock (list of joint names to delete), noeq, noact, one_flex
    """
    o = o or {}
    spec = mujoco.MjSpec.from_file(str(scene_path(grid, o.get("scene", "scene.xml"))))
    E, nu = o.get("E", 1e6), o.get("nu", 0.45)
    young, pois = f1_inputs(E, nu) if o.get("f1") else (E, nu)
    for f in spec.flexes:
        f.young, f.poisson = young, pois
        if "flexdamp" in o:
            f.damping = o["flexdamp"]
    if "h" in o:
        spec.option.timestep = o["h"]
    if "solver" in o:
        spec.option.solver = {"PGS": 0, "CG": 1, "Newton": 2}[o["solver"]]
    if "it" in o:
        spec.option.iterations = o["it"]
    if "tol" in o:
        spec.option.tolerance = o["tol"]
    if "ls_it" in o:
        spec.option.ls_iterations = o["ls_it"]
    if "impratio" in o:
        spec.option.impratio = o["impratio"]
    if "cone" in o:
        spec.option.cone = {"pyramidal": 0, "elliptic": 1}[o["cone"]]
    if "noslip" in o:
        spec.option.noslip_iterations = o["noslip"]
    if "memory" in o:
        spec.memory = int(o["memory"])
    if o.get("grav") is False:
        spec.option.gravity = [0, 0, 0]
    if "pdms_mass" in o:
        for s in ("left", "right"):
            b = spec.body(f"{s}_pdms")
            b.mass = o["pdms_mass"]
            b.inertia = [o["pdms_mass"] * 1e-4] * 3
    if "box_half" in o:
        spec.geom("box").size = list(o["box_half"])
        spec.geom("table_top").size = [0.6 * o["box_half"][0], 0.6 * o["box_half"][1], 0.15]
    if o.get("one_flex"):
        for f in list(spec.flexes):
            if f.name.startswith("right"):
                spec.delete(f)
    return spec


def compile_spec(spec):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = spec.compile()
    return m, mujoco.MjData(m)


def load(grid="11x15x3", o=None):
    return compile_spec(make_spec(grid, o))


def reset_home(m, d):
    """sim_gripper.reset: keyframe 'home', arm servo targets = HOME, fingers open."""
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.ctrl[:6] = HOME
    d.ctrl[m.actuator("fingers").id] = 0
    mujoco.mj_forward(m, d)


def nbad(d):
    return int(d.warning[W.mjWARN_BADQACC].number)


def warns(d):
    return {mujoco.mjtWarning(i).name: int(d.warning[i].number) for i in range(len(d.warning)) if d.warning[i].number}


def flex_dofs(m, f=None):
    """dof indices of the flex vertex bodies (all flexes if f is None)."""
    out = []
    fl = range(m.nflex) if f is None else [f]
    for k in fl:
        va, vn = m.flex_vertadr[k], m.flex_vertnum[k]
        for b in sorted(set(m.flex_vertbodyid[va:va + vn])):
            if m.body_dofnum[b] and m.body_weldid[b] == b:
                a = m.body_dofadr[b]
                out.extend(range(a, a + m.body_dofnum[b]))
    return np.array(sorted(set(out)), int)


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
