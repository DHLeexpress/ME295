"""Is the dof="full" flex elasticity resistant to inversion? Fixed (no joint) 11x15x3 layer, back face pinned.
Push all free-face vertices (z = -t/2) through the thickness by w and record |qfrc_spring| (normal force on the free
face) and the elastic energy proxy. w = 1.5 mm puts the free face onto the back face: the free-middle tetrahedra are
then the mirror images of their rest shape (all edge lengths at rest). python mirror_test.py -> out/mirror_test.json"""
import json, numpy as np, mujoco
from pathlib import Path
H = Path(__file__).resolve().parent
xml = (H / "repro" / "min_m_fixed.xml").read_text()
res = []
for young, pois, lab in ((1e6, 0.45, "raw nu .45"), (971787.0, 0.409091, "F1-corrected nu .45")):
    x = xml.replace('young="1000000.0"', f'young="{young}"').replace('poisson="0.45"', f'poisson="{pois}"')
    m = mujoco.MjModel.from_xml_string(x); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # vertex bodies: the free ones have 3 slide dofs; rest positions in the parent frame
    f = 0; va, vn = m.flex_vertadr[f], m.flex_vertnum[f]
    X0 = d.flexvert_xpos[va:va + vn].copy()
    zfree = X0[:, 2].min()
    rows = []
    for w in np.r_[np.linspace(0, 1.5e-3, 31), 1.6e-3, 1.75e-3, 2.0e-3]:
        mujoco.mj_resetData(m, d)
        for v in range(vn):
            b = m.flex_vertbodyid[va + v]
            if m.body_dofnum[b] == 3 and abs(X0[v, 2] - zfree) < 1e-9:
                a = m.body_jntadr[b]
                # slide joints along x,y,z of the parent frame (parent axes = world here: pdms body unrotated)
                ax = [m.jnt_axis[a + k] for k in range(3)]
                kz = int(np.argmax([abs(q[2]) for q in ax]))
                d.qpos[m.jnt_qposadr[a + kz]] = w * np.sign(ax[kz][2])
        mujoco.mj_forward(m, d)
        Fz = 0.0
        for v in range(vn):
            b = m.flex_vertbodyid[va + v]
            if m.body_dofnum[b] == 3 and abs(X0[v, 2] - zfree) < 1e-9:
                a = m.body_dofadr[b]; ja = m.body_jntadr[b]
                for k in range(3):
                    if abs(m.jnt_axis[ja + k][2]) > 0.5:
                        Fz += d.qfrc_spring[a + k] * np.sign(m.jnt_axis[ja + k][2])
        rows.append({"w_mm": 1e3 * w, "Fz_free_face_N": float(Fz), "qfrc_spring_norm": float(np.linalg.norm(d.qfrc_spring))})
    res.append({"case": lab, "young": young, "poisson": pois, "rows": rows})
    print(lab)
    for r in rows[::3] + rows[-4:]:
        print(f"  w {r['w_mm']:.3f} mm  free-face force {r['Fz_free_face_N']:10.2f} N  |qfrc_spring| {r['qfrc_spring_norm']:.3e}")
json.dump(res, open(H / "out" / "mirror_test.json", "w"), indent=1)
