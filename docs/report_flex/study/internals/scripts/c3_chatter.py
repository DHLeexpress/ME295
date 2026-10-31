"""Item 3c: does the contact with a rigid platen come to rest? (element contacts vs vertex contacts)

Static mocap platen held at delta = 5 um into the free face of the 11x15x3 layer (fixed parent), discrete
integrator, our contact settings. We record every step: total normal contact force, number of contacts,
mean free-face u_z, max |qvel|, and how far the contact points of the same element move between steps.
Outputs: out/c3_chatter.json
"""
import sys

import numpy as np
import mujoco

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import load, dump
from c1_contact_census import layer_xml

ZF = 0.1 - 0.00075
TOOLS = {
    "box": ('<geom type="box" size="0.03 0.04 0.005" contype="2" conaffinity="2"/>', 0.005),
    "plane": ('<geom type="plane" size="0.1 0.1 0.01" contype="2" conaffinity="2"/>', 0.0),
    "sphere_R25": ('<geom type="sphere" size="0.025" contype="2" conaffinity="2"/>', 0.025),
}


def run(tool, delta=5e-6, nstep=2000, **kw):
    geom, top = TOOLS[tool]
    m, _ = load(layer_xml(geom, ZF + delta - top, **kw))
    d = mujoco.MjData(m)
    va, n = m.flex_vertadr[0], m.flex_vertnum[0]
    face = np.isclose(d.flexvert_xpos[va:va + n, 2], ZF) if False else None
    mujoco.mj_forward(m, d)
    face = np.isclose(d.flexvert_xpos[va:va + n, 2], ZF)
    F, NC, UZ, QV, JUMP = [], [], [], [], []
    prev = {}
    c6 = np.zeros(6)
    for k in range(nstep):
        mujoco.mj_step(m, d)
        tot = 0.0
        cur = {}
        for i in range(d.ncon):
            mujoco.mj_contactForce(m, d, i, c6)
            tot += c6[0]
            cur[(d.contact[i].elem[1], d.contact[i].vert[1])] = d.contact[i].pos.copy()
        jumps = [np.linalg.norm(cur[key] - prev[key]) for key in cur if key in prev]
        prev = cur
        F.append(tot)
        NC.append(d.ncon)
        UZ.append(float((d.flexvert_xpos[va:va + n][face, 2] - ZF).mean()))
        QV.append(float(np.abs(d.qvel).max()))
        JUMP.append(float(np.max(jumps)) if jumps else 0.0)
    F, NC, UZ, QV, JUMP = map(np.array, (F, NC, UZ, QV, JUMP))
    tail = slice(nstep // 2, None)
    out = dict(tool=tool, delta=delta, kw={k: str(v) for k, v in kw.items()},
               F_mean=float(F[tail].mean()), F_cv=float(F[tail].std() / max(abs(F[tail].mean()), 1e-300)),
               F_minmax=[float(F[tail].min()), float(F[tail].max())],
               ncon_minmax=[int(NC[tail].min()), int(NC[tail].max())],
               uz_mean_over_delta=float(UZ[tail].mean() / delta), uz_cv=float(UZ[tail].std() / abs(UZ[tail].mean())),
               qvel_max_tail_median=float(np.median(QV[tail])),
               contact_point_jump_um_median=float(np.median(JUMP[tail]) * 1e6),
               contact_point_jump_um_max=float(JUMP[tail].max() * 1e6),
               F_trace=F[::10].tolist())
    return out


def main():
    res = []
    for tool, kw in (("plane", {}), ("box", {}), ("sphere_R25", {}), ("box", dict(radius=1e-5)),
                     ("box", dict(solref="0.001 1")), ("box", dict(refsafe=True))):
        r = run(tool, **kw)
        res.append(r)
        print({k: v for k, v in r.items() if k != "F_trace"})
    p = dump("c3_chatter.json", res)
    print("wrote", p)


if __name__ == "__main__":
    main()
