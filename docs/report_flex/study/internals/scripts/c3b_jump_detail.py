"""Detail for c3: position of the contact of one surface tet under the static box platen, step by step,
and its inverse-distance weights (which vertex gets the force)."""
import sys
import numpy as np, mujoco
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import load, dump
from c1_contact_census import layer_xml, elem_vert_weight
ZF = 0.1 - 0.00075
m, _ = load(layer_xml('<geom type="box" size="0.03 0.04 0.005" contype="2" conaffinity="2"/>', ZF + 5e-6 - 0.005))
d = mujoco.MjData(m)
for k in range(400):
    mujoco.mj_step(m, d)
target = None
rows = []
for k in range(8):
    mujoco.mj_step(m, d)
    for i in range(d.ncon):
        e = d.contact[i].elem[1]
        if target is None:
            target = e
        if e == target:
            ed, w = elem_vert_weight(m, d, 0, e, d.contact[i].pos)
            X = d.flexvert_xpos[ed]
            rows.append(dict(step=k, pos_mm=((d.contact[i].pos - np.array([0, 0, 0.1])) * 1e3).round(4).tolist(),
                             dist_um=round(d.contact[i].dist * 1e6, 4), weights=w.round(3).tolist(),
                             vert_z_local_mm=((X[:, 2] - 0.1) * 1e3).round(4).tolist()))
            print(rows[-1])
dump("c3b_jump_detail.json", rows)
