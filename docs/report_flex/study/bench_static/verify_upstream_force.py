# Momentum check of the flex elastic force when the pinned pad's parent body has a joint (as on the robot).
# Internal elastic forces sum to zero, so the generalized spring force on an upstream slide joint along z must be
# Q_base = sum_i f_i,z = 0 for ANY deformation. Displace free points (random small field), read qfrc_spring.
# Also: free-point displacement -> world force sum, and the energy-consistency (symmetry) of the generalized K.
import sys, os, json
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from upstream import xml_up

out = []
for c in [dict(variant="full_corr", nt=1, nz=3, nu=0.45), dict(variant="trilinear", nt=1, cz=2, nu=0.45),
          dict(variant="quadratic", nt=1, cz=1, nu=0.45)]:
    for joint in ("slide", "hinge"):
        m = mujoco.MjModel.from_xml_string(xml_up(c, joint, 0.1)); d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        P = points(m, d)
        rng = np.random.default_rng(0)
        q = d.qpos.copy(); q[1:] += 1e-6 * rng.standard_normal(m.nq - 1)     # deform: move free points only
        f = passive_force(m, d, q)
        # world-frame elastic force on every free point = its 3 slide-dof forces (parent axes = world here)
        free = P["dofadr"] >= 0
        fw = np.array([f[a:a + 3] for a in P["dofadr"][free]])
        rec = dict(id=cfg_id(c), joint=joint,
                   Q_base=float(f[0]),
                   sum_free_point_forces_z=float(fw[:, 2].sum()),
                   max_point_force=float(np.abs(fw).max()))
        if joint == "slide":
            rec["expected_Q_base"] = 0.0
            rec["Q_base_over_max_point_force"] = abs(rec["Q_base"]) / rec["max_point_force"]
        out.append(rec)
        print(json.dumps(rec))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_upstream_force.json"), "w"), indent=1)
