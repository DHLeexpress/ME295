"""Compile check of recommended_pdms_flex.xml inside a minimal model with the robot options; 200 steps under gravity."""
import re
import mujoco
import numpy as np
s = open("recommended_pdms_flex.xml").read()
fc = re.search(r"<flexcomp name=.*?</flexcomp>", s, re.S).group(0)
xml = ('<mujoco><option timestep="0.0005" integrator="discrete" solver="CG" iterations="100" cone="elliptic" '
       'impratio="100"><flag island="disable" refsafe="disable"/></option><worldbody>'
       '<body name="left_pdms">' + fc + '</body></worldbody></mujoco>')
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
for _ in range(200):
    mujoco.mj_step(m, d)
print("flex verts", m.flex_vertnum[0], "nv", m.nv, "free-vertex mass kg", float(m.body_mass[m.body_dofnum > 0][0]),
      "finite", bool(np.isfinite(d.qpos).all()), "max |u| um", 1e6 * float(np.abs(d.qpos).max()))
