"""Scene variations of the librarian cell (assembly/scene_bookshelf.xml) for the VLA baseline.

Only shelf A's upper tier changes; the robot, the desk, shelf B and the lower tier stay as in the simulator:

    designated book   any upper-tier book thick enough for the pads to squeeze (>= 35 mm; the pads close to 32.2 mm)
    row order         books re-packed between the bookends with 0.5-1 mm gaps (as in the original scene)
    gaps              air on each side of the designated book (the original: 10 mm each)
    colour            named colour of the designated book (used in the instruction) and of the other books
    mass              of the designated book (inertia scaled)
    distractor        optionally a second book with the same gaps, another colour (tests language grounding)
    start             perturbation of the robot's start pose

nominal() reproduces the simulator's scene exactly: the dark blue 40 mm book_target in the middle, 10 mm gaps, 0.8 kg.
An exterior camera `ext_cam` (third-person, front-right of the robot, sees the shelf and the gripper) is added to
every variant; `wrist_cam` is the robot's own wrist camera.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "assembly" / "scene_bookshelf.xml"

UPPER = ["book_L6", "book_L5", "book_L4", "book_L3", "book_L2", "book_L1", "book_target",
         "book_R1", "book_R2", "book_R3", "book_R4", "book_R5", "book_R6", "book_R7"]   # left -> right, original order
SPINE_Y = 0.82            # every spine face
SHELF_FRONT_Y = 0.80      # front edge of the boards / side panels
BOARD_TOP = 0.42          # upper tier stands on the middle board
ROW_LIMITS = (-0.333, 0.333)   # inner side panels at +-0.34 (bookend feet included)
BOOKEND_GAP = 0.004       # bookend plate 4 mm past the last book (original)
MIN_GRASP = 0.035         # thinnest designated book: the pads close to 32.2 mm; on 34 mm books the fingers stop within
                          # 3 register counts of "closed", the Robotiq reports "at position" instead of "object
                          # detected" and the fixed-force grasp never starts pulling (expert fails)

# Named colours for the designated book / distractors / other books (well separated hues, cloth-like values)
PALETTE: Dict[str, Tuple[float, float, float]] = {
    "red": (0.62, 0.07, 0.07),
    "blue": (0.06, 0.16, 0.52),
    "green": (0.07, 0.40, 0.15),
    "yellow": (0.86, 0.72, 0.10),
    "white": (0.90, 0.89, 0.85),
    "black": (0.05, 0.05, 0.06),
    "orange": (0.88, 0.40, 0.06),
    "purple": (0.36, 0.12, 0.46),
}
NOMINAL_COLOR = "dark blue"   # book_target's own colour (0.055, 0.115, 0.36)

# Exterior camera: front-right of the robot at shoulder height, looking at shelf A's upper tier
EXT_CAM = {"pos": (0.62, 0.20, 0.95), "lookat": (0.0, 0.86, 0.50), "fovy": 52.0}


def look_at_xyaxes(pos, lookat, up=(0.0, 0.0, 1.0)) -> List[float]:
    """MuJoCo camera xyaxes for a camera at pos looking at lookat (camera looks along its -z)."""
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return [*x, *y]


@dataclass
class Variation:
    designated: str = "book_target"
    color: str = NOMINAL_COLOR
    order: List[str] = field(default_factory=lambda: list(UPPER))
    gap_left: float = 0.010
    gap_right: float = 0.010
    mass: Optional[float] = None                 # kg, None = keep
    distractor: Optional[str] = None
    distractor_color: Optional[str] = None
    colors: Dict[str, str] = field(default_factory=dict)        # other books' palette names ({} = keep original)
    packed_gaps: List[float] = field(default_factory=list)      # gaps between consecutive books (len = n - 1)
    row_shift: float = 0.0                       # extra x shift of the whole row (m)
    start_offset: Tuple[float, ...] = (0.0,) * 6  # TCP start perturbation [dx, dy, dz, drx, dry, drz]
    keep_original: bool = True                   # True: the simulator's layout untouched (nominal)
    seed: int = -1
    split: str = "nominal"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Variation":
        d = json.loads(s)
        d["start_offset"] = tuple(d["start_offset"])
        return cls(**d)


def nominal() -> Variation:
    return Variation()


@dataclass
class SceneInfo:
    """What the expert and the metrics need to know about a built scene (sim ground truth)."""
    variation: Variation
    designated: str
    center: np.ndarray            # designated book centre at rest (m)
    half: np.ndarray              # its half extents (x thickness, y depth, z height)
    mass: float
    others: List[str]             # every other free book
    instruction: str


def _book_dims(spec: mujoco.MjSpec) -> Dict[str, dict]:
    out = {}
    for n in UPPER:
        g = spec.geom(n)
        out[n] = {"half": np.array(g.size, float), "pos": np.array(spec.body(n).pos, float), "mass": float(g.mass),
                  "rgba": np.array(g.rgba, float)}
    return out


def sample(rng: np.random.Generator, split: str = "train", gap_range=(0.008, 0.012), mass_range=(0.4, 1.2),
           p_distractor: float = 0.3, start_scale: float = 1.0) -> Variation:
    """Random layout. split is recorded only; the caller chooses ranges per split."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    dims = _book_dims(spec)
    graspable = [n for n in UPPER if 2 * dims[n]["half"][0] >= MIN_GRASP]
    for _ in range(200):
        designated = str(rng.choice(graspable))
        distractor = None
        if rng.random() < p_distractor:
            distractor = str(rng.choice([n for n in graspable if n != designated]))
        order = [str(n) for n in rng.permutation(UPPER)]
        gl, gr = (float(rng.uniform(*gap_range)) for _ in range(2))
        packed = [float(rng.uniform(0.0005, 0.001)) for _ in range(len(order) - 1)]
        special = {designated: (gl, gr)}
        if distractor:
            special[distractor] = (float(rng.uniform(*gap_range)), float(rng.uniform(*gap_range)))
        for i in range(len(order) - 1):  # widen the gaps beside the designated book (and the distractor)
            a, b = order[i], order[i + 1]
            if b in special:
                packed[i] = max(packed[i], special[b][0])
            if a in special:
                packed[i] = max(packed[i], special[a][1])
        # drop books from the ends until the row fits between the side panels
        while True:
            width = sum(2 * dims[n]["half"][0] for n in order) + sum(packed) + 2 * (BOOKEND_GAP + 0.034)
            if width <= ROW_LIMITS[1] - ROW_LIMITS[0]:
                break
            end = 0 if rng.random() < 0.5 else -1
            if order[end] in special:
                end = -1 - end
            if order[end] in special:
                break
            order.pop(end)
            packed.pop(0 if end == 0 else -1)
        idx = order.index(designated)
        if idx < 2 or idx > len(order) - 3:   # keep a neighbour pair on each side
            continue
        slack = (ROW_LIMITS[1] - ROW_LIMITS[0]) - width
        v = Variation(designated=designated, order=order, gap_left=gl, gap_right=gr, distractor=distractor,
                      packed_gaps=packed, row_shift=float(rng.uniform(-0.5, 0.5) * slack), keep_original=False,
                      split=split)
        x = _row_positions(v, dims)[designated]
        if abs(x) > 0.16:                     # stay inside the arm's comfortable horizontal-approach box
            continue
        names = list(PALETTE)
        v.color = str(rng.choice(names))
        rest = [c for c in names if c != v.color]
        if distractor:
            v.distractor_color = str(rng.choice(rest))
        v.colors = {n: str(rng.choice(rest)) for n in order if n not in (designated, distractor)}
        v.mass = float(rng.uniform(*mass_range))
        v.start_offset = tuple(float(s) for s in np.r_[rng.uniform(-0.03, 0.03, 3), rng.uniform(-0.06, 0.06, 3)]
                               * start_scale)
        return v
    raise RuntimeError("could not sample a feasible layout")


def _row_positions(v: Variation, dims: Dict[str, dict]) -> Dict[str, float]:
    """Book centre x for a re-packed row, centred between the side panels (+ row_shift)."""
    width = sum(2 * dims[n]["half"][0] for n in v.order) + sum(v.packed_gaps)
    x = -0.5 * width + v.row_shift
    out = {}
    for i, n in enumerate(v.order):
        h = dims[n]["half"][0]
        out[n] = x + h
        x += 2 * h + (v.packed_gaps[i] if i < len(v.packed_gaps) else 0.0)
    return out


def instruction_for(v: Variation, template: str = "pick up the {color} book") -> str:
    return template.format(color=v.color)


def build(v: Variation, ext_cam: dict = EXT_CAM) -> Tuple[mujoco.MjModel, SceneInfo]:
    spec = mujoco.MjSpec.from_file(str(SCENE))
    dims = _book_dims(spec)
    if not v.keep_original:
        xs = _row_positions(v, dims)
        for n in UPPER:
            if n in xs:
                spec.body(n).pos = [xs[n], dims[n]["pos"][1], dims[n]["pos"][2]]
            else:  # dropped to make room for the wider gaps
                spec.delete(spec.body(n))
        # bookends 4 mm past the first/last book; feet outward
        left = xs[v.order[0]] - dims[v.order[0]]["half"][0] - BOOKEND_GAP - 0.002
        right = xs[v.order[-1]] + dims[v.order[-1]]["half"][0] + BOOKEND_GAP + 0.002
        spec.geom("bookend_L").pos = [left, 0.905, 0.505]
        spec.geom("bookend_L_foot").pos = [left - 0.017, 0.905, 0.421]
        spec.geom("bookend_R").pos = [right, 0.905, 0.505]
        spec.geom("bookend_R_foot").pos = [right + 0.017, 0.905, 0.421]
        for n, cname in v.colors.items():
            spec.geom(n).rgba = [*PALETTE[cname], 1.0]
        spec.geom(v.designated).rgba = [*PALETTE[v.color], 1.0]
        if v.distractor:
            spec.geom(v.distractor).rgba = [*PALETTE[v.distractor_color], 1.0]
    if v.mass is not None:
        spec.geom(v.designated).mass = v.mass
    cam = spec.worldbody.add_camera()
    cam.name = "ext_cam"
    cam.pos = list(ext_cam["pos"])
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = look_at_xyaxes(ext_cam["pos"], ext_cam["lookat"])
    cam.fovy = ext_cam["fovy"]
    m = spec.compile()
    b = m.body(v.designated).id
    g = m.geom(v.designated).id
    others = [m.body(i).name for i in range(m.nbody)
              if m.body(i).name.startswith("book") and m.body(i).name != v.designated]
    info = SceneInfo(variation=v, designated=v.designated, center=m.body_pos[b].copy(), half=m.geom_size[g].copy(),
                     mass=float(m.body_mass[b]), others=others, instruction=instruction_for(v))
    return m, info
