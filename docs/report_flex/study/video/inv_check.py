"""Count inverted / strongly compressed tetrahedra in saved frames (signed volume ratio vs rest).
Usage: python inv_check.py <seg> [frame indices...]   (default: every 10th frame)"""
import sys, json
import numpy as np, mujoco
import sim
k = int(sys.argv[1])
z = np.load(f"runs/seg{k}.npz", allow_pickle=True)
m = mujoco.MjModel.from_xml_path(f"runs/seg{k}.xml")
d = mujoco.MjData(m)
adr, num = m.flex_vertadr[0], m.flex_vertnum[0]
el = m.flex_elem[m.flex_elemdataadr[0]:m.flex_elemdataadr[0] + 4 * m.flex_elemnum[0]].reshape(-1, 4)
def vols(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_flex(m, d)
    p = d.flexvert_xpos[adr:adr + num]
    a, b, c, e = (p[el[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), e - a) / 6
v0 = vols(z["f_qpos"][0] * 0 + m.qpos0) if False else None
mujoco.mj_resetData(m, d); v0 = vols(d.qpos.copy())
idx = [int(x) for x in sys.argv[2:]] or list(range(0, len(z["f_step"]), 10)) + [len(z["f_step"]) - 1]
out = []
for i in idx:
    r = vols(z["f_qpos"][i]) / v0
    s = int(z["f_step"][i])
    out.append(dict(frame=i, step=s, phase=int(z["phase"][s]), n_inverted=int((r < 0).sum()), min_J=float(r.min()),
                    n_J_below_0p5=int((r < 0.5).sum()), uz_max_um=float(z["f_ufront"][i][:, 2].max() * 1e6),
                    Fz_N=float(z["F"][s, 2])))
    print(out[-1])
