"""One stability run of the robot scene with edits; prints one JSON line.

python run.py '<json options>' [tag]
options (besides common.make_spec ones): grid, T (s, default 0.2), lock: list of joint-name substrings to weld,
noeq, noact, nosens, warmstart (bool), island (bool), stop_on_bad (default True), armature0, damping0 (zero all joint armature/damping)
Divergence = mjWARN_BADQACC count > 0 (MuJoCo auto-resets after NaN/huge qacc).
"""
import sys
import json
import time
import numpy as np
import mujoco
from common import make_spec, compile_spec, nbad, warns, flex_dofs, HOME

o = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
tag = sys.argv[2] if len(sys.argv) > 2 else ""
spec = make_spec(o.get("grid", "11x15x3"), o)
if o.get("armature0") or o.get("damping0"):
    for j in spec.joints:
        if o.get("armature0"):
            j.armature = 0
        if o.get("damping0"):
            j.damping = 0

lock = o.get("lock", [])
if lock or o.get("noeq") or o.get("noact"):
    for k in list(spec.keys):
        spec.delete(k)
    for s in list(spec.sensors):
        spec.delete(s)
if lock:
    locked = [j.name for j in spec.joints if j.name and any(s in j.name for s in lock)]
    for e in list(spec.equalities):
        if e.name1 in locked or e.name2 in locked:
            spec.delete(e)
    for a in list(spec.actuators):
        if a.target in locked:
            spec.delete(a)
    for j in list(spec.joints):
        if j.name in locked:
            spec.delete(j)
if o.get("noeq"):
    for e in list(spec.equalities):
        spec.delete(e)
if o.get("noact"):
    for a in list(spec.actuators):
        spec.delete(a)
if o.get("warmstart") is False:
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
if o.get("island") is False:
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
m, d = compile_spec(spec)

# reset: keyframe home if present, else set the arm joints that still exist to HOME and hold them
if m.nkey:
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
else:
    mujoco.mj_resetData(m, d)
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    for n, q in zip(names, HOME):
        try:
            d.qpos[m.joint(n).qposadr[0]] = q
        except KeyError:
            pass
for i, n in enumerate(["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]):
    try:
        d.ctrl[m.actuator(n).id] = HOME[i]
    except KeyError:
        pass
try:
    d.ctrl[m.actuator("fingers").id] = o.get("ctrl_fingers", 0)
except KeyError:
    pass
mujoco.mj_forward(m, d)

fd = flex_dofs(m)
T = o.get("T", 0.2)
n = int(round(T / m.opt.timestep))
t0 = time.time()
first_bad = None
maxv = 0.0
niter = []
fatal = None
for k in range(n):
    try:
        mujoco.mj_step(m, d)
    except mujoco.FatalError as e:
        fatal = str(e).splitlines()[0][:120]
        first_bad = first_bad or k + 1
        break
    niter.append(int(d.solver_niter[:max(1, d.nisland)].sum()))
    if nbad(d) and first_bad is None:
        first_bad = k + 1
        if o.get("stop_on_bad", True):
            break
    if len(fd):
        maxv = max(maxv, float(np.abs(d.qvel[fd]).max()))
wall = time.time() - t0
res = {"tag": tag, "o": o, "mujoco_file": mujoco.__file__, "nv": m.nv, "steps": k + 1, "first_bad": first_bad, "bad": nbad(d),
       "max_flex_qvel": maxv, "ms_per_step": 1e3 * wall / (k + 1), "mean_niter": float(np.mean(niter)) if niter else 0,
       "max_niter": int(np.max(niter)) if niter else 0, "fatal": fatal, "warnings": warns(d), "njnt": m.njnt, "neq": m.neq, "nu": m.nu}
print(json.dumps(res), flush=True)
