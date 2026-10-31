"""Minimal repro of F4: ONE hinge (UR10e wrist parameters: armature 0.1, damping 2) carrying a 51 x 71 x 1.5 mm
dof="full" flex (11x15x3, back face pinned to the hinge's child body), robot option block.

python minimal.py '<json>'   writes repro/min_<tag>.xml and prints one JSON line.
options: con = none | eq (joint equality holding the hinge at 0) | limit (hinge starts on its limit) |
              floor (a box on the link rests on a plane) ;  island (default true) ; joint = hinge | slide ;
         grid "11 15 3"; young, poisson; h; T (s); solver; pdms_mass; fixed (true: no joint at all)
"""
import sys, json, os, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from pathlib import Path

o = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
tag = o.get("tag", "min")
con = o.get("con", "eq")
nx, ny, nz = map(int, o.get("grid", "11 15 3").split())
sx, sy, sz = 0.051 / (nx - 1), 0.071 / (ny - 1), 0.0015 / (nz - 1)
jt = o.get("joint", "hinge")
rng = 'range="0 1" limited="true"' if con == "limit" else ""
if o.get("fixed"):
    joint = ""
elif jt == "hinge":
    joint = f'<joint name="j" type="hinge" axis="0 1 0" armature="0.1" damping="2" {rng}/>'
else:
    joint = f'<joint name="j" type="slide" axis="0 0 1" armature="0.1" damping="2" {rng}/>'
eq = '<equality><joint joint1="j" polycoef="0 0 0 0 0" solref="0.001 1"/></equality>' if con == "eq" else ""
floor = '<geom name="floor" type="plane" size="0 0 1" pos="0 0 -0.2"/>' if con == "floor" else ""
foot = '<geom type="box" size="0.02 0.02 0.02" pos="0.15 0 -0.18" mass="0.1"/>' if con == "floor" else ""
isl = '' if o.get("island", True) else ' island="disable"'
xml = f"""<mujoco model="min_{tag}">
  <option integrator="discrete" solver="{o.get('solver', 'CG')}" timestep="{o.get('h', 0.0005)}" cone="elliptic" impratio="100">
    <flag refsafe="disable"{isl}/>
  </option>
  <worldbody>
    {floor}
    <body name="link" pos="0 0 0">
      {joint}
      <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.03" mass="1" contype="0" conaffinity="0"/>
      {foot}
      <body name="pdms" pos="0.15 0 0.04">
        <inertial pos="0 0 0" mass="{o.get('pdms_mass', 1e-6)}" diaginertia="1e-12 1e-12 1e-12"/>
        <flexcomp name="pdms_flex" type="grid" dim="3" count="{nx} {ny} {nz}" spacing="{sx:.7f} {sy:.7f} {sz:.7f}"
                  radius="0" mass="0.0052414">
          <elasticity young="{o.get('young', 1e6)}" poisson="{o.get('poisson', 0.45)}" damping="0"/>
          <contact contype="2" conaffinity="2" condim="3" priority="2" friction="1 0.005 0.0001"
                   solref="0.0001 1" solimp="0.99 0.99 0.001" selfcollide="none"/>
          <pin gridrange="0 0 {nz-1} {nx-1} {ny-1} {nz-1}"/>
        </flexcomp>
      </body>
    </body>
  </worldbody>
  {eq}
</mujoco>
"""
p = Path(__file__).resolve().parent / "repro" / f"min_{tag}.xml"
p.write_text(xml)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = mujoco.MjModel.from_xml_path(str(p))
d = mujoco.MjData(m)
if con == "limit" and not o.get("fixed"):
    d.qpos[m.joint("j").qposadr[0]] = -1e-4      # slightly beyond the lower limit -> active limit row
mujoco.mj_forward(m, d)
n = int(round(o.get("T", 0.1) / m.opt.timestep))
first_bad, fatal, t0 = None, None, time.time()
k = 0
for k in range(n):
    try:
        mujoco.mj_step(m, d)
    except mujoco.FatalError as e:
        fatal = str(e).splitlines()[0][:120]; first_bad = k + 1; break
    if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number and first_bad is None:
        first_bad = k + 1
        break
W = {mujoco.mjtWarning(i).name: int(d.warning[i].number) for i in range(len(d.warning)) if d.warning[i].number}
print(json.dumps({"tag": tag, "o": o, "xml": str(p), "mujoco_file": mujoco.__file__, "nv": m.nv, "nefc_last": int(d.nefc),
                  "nisland_last": int(d.nisland), "steps": k + 1, "first_bad": first_bad, "fatal": fatal, "warnings": W,
                  "ms_per_step": 1e3 * (time.time() - t0) / (k + 1)}), flush=True)
