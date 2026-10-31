"""Diagnostic for c2: per-contact static relation for the box platen (element contacts) vs the plane."""
import sys
import numpy as np, mujoco
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from c2_platen_static import run
from common import dump
res = {}
for tool, cond in (("plane", 3), ("box", 3), ("box", 1)):
    r, m, d = run(tool=tool, condim=cond, mu=1.0 if cond == 3 else 0.0, nstep=3000, iterations=1000, tolerance=1e-12)
    h = m.opt.timestep
    st = {}
    rel = []
    for i in range(d.ncon):
        a = d.contact[i].efc_address
        if a < 0:
            continue
        K, B, I, P = d.efc_KBIP.reshape(-1, 4)[a]
        f = 1 + h * B + h * h * K * I
        pred = -K * I * d.efc_pos[a] / (f * d.efc_R[a])
        st[int(d.efc_state[a])] = st.get(int(d.efc_state[a]), 0) + 1
        if d.efc_force[a] > 0:
            rel.append(d.efc_force[a] / pred)
    rel = np.array(rel)
    out = dict(K_n=r["K_n_N_per_um"], conv=r["converged_rel"], qvel_max=float(np.abs(d.qvel).max()),
               states=st, force_over_static_pred_median=float(np.median(rel)), p10=float(np.percentile(rel, 10)),
               p90=float(np.percentile(rel, 90)), face_uz=r["face_uz_mean_over_delta"], niter=r["solver_niter"])
    res[f"{tool} condim={cond}"] = out
    print(tool, cond, out)
dump("c2b_box_diag.json", res)
