# Why the robot's PCG warning appears only when an upstream joint is free (F4): the same pinned pad, but its
# parent body now has one joint (slide z or hinge x). Checks in MuJoCo whether the flex stiffness is still
# assembled into the effective-metric CSR (d.nefmK) and preconditioned by 3x3 blocks (d.nefmdof), measures
# MuJoCo's own mj_effSolve iteration count (smallest opt.iterations with no mjWARN_INERTIA), and reproduces
# it with an independent PCG on M_gen + h^2 K_gen preconditioned by M_gen (the "backbone" solve).
# Usage: python upstream.py out.json [maxnv]
import sys, os, json, time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from run_model import pcg_mujoco


def xml_up(c, joint, mbase):
    x = make_xml(c, pinned=True, gravity=True)
    if joint == "none":
        return x
    if joint == "slide":
        j = '<joint name="base" type="slide" axis="0 0 1"/>'
    else:   # hinge about x through a pivot 50 mm behind the pad (finger-like)
        j = '<joint name="base" type="hinge" axis="1 0 0" pos="0 -0.05 0"/>'
    ine = f'<inertial pos="0 0 0" mass="{mbase}" diaginertia="1e-4 1e-4 1e-4"/>'
    return x.replace('<body name="pad">', f'<body name="pad">\n      {j}\n      {ine}')


def mj_iters(m, N):
    m.opt.iterations = int(N)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return int(d.warning[mujoco.mjtWarning.mjWARN_INERTIA].number) == 0


def mj_min_iters(m, hi=20000):
    """Smallest opt.iterations with which mj_forward's mj_effSolve converges (no warning); bisection."""
    if not mj_iters(m, hi):
        return None
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mj_iters(m, mid):
            hi = mid
        else:
            lo = mid
    return hi


def run(c, joint, mbase=0.1):
    m = mujoco.MjModel.from_xml_string(xml_up(c, joint, mbase)); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    nv = m.nv
    rec = dict(id=cfg_id(c), joint=joint, base_mass_kg=mbase, nv=nv, nefmK=int(d.nefmK), nefmdof=int(d.nefmdof),
               efm_active=int(d.efm_active))
    # generalized M (dense) and K_gen by central FD of qfrc_spring over all qpos (nv small here)
    M = np.zeros((nv, nv)); mujoco.mj_fullM(m, d, M)
    q0 = d.qpos.copy(); h = 1e-8
    K = np.zeros((nv, nv))
    for j in range(nv):
        qp = q0.copy(); qp[j] += h; fp = passive_force(m, d, qp)
        qm = q0.copy(); qm[j] -= h; fm = passive_force(m, d, qm)
        K[:, j] = -(fp - fm) / (2 * h)
    passive_force(m, d, q0); mujoco.mj_forward(m, d)
    rec["Kgen_sym_rel"] = float(np.abs(K - K.T).max() / np.abs(K).max())
    rec["Kgen_base_row_rel"] = float(np.abs(K[0]).max() / np.abs(K).max()) if joint != "none" else None
    A = M + H_STEP ** 2 * K
    A = 0.5 * (A + A.T)
    b = np.array(d.qfrc_smooth)
    # preconditioner actually used by MuJoCo: 3x3 blocks if the flex is assembled, else M^{-1} (backbone)
    if d.nefmdof:
        n = nv; Pinv = np.zeros((n, n))
        dofid = np.array(d.efm_dofid[:d.nefmdof])
        covered = np.zeros(n, bool)
        for k in dofid:
            Pinv[k:k + 3, k:k + 3] = np.linalg.inv(A[k:k + 3, k:k + 3]); covered[k:k + 3] = True
        unc = np.where(~covered)[0]
        if unc.size:   # backbone solve on the uncovered dofs (M with covered rhs entries zeroed)
            Minv = np.linalg.inv(M)
            Pinv[np.ix_(unc, unc)] = Minv[np.ix_(unc, unc)]
        prec = "3x3 blocks of M+h^2K"
    else:
        Pinv = np.linalg.inv(M); prec = "M^-1 (backbone only, no stiffness blocks)"
    Ps = sp.csr_matrix(Pinv); As = sp.csr_matrix(A)
    rec["preconditioner"] = prec
    rec["pcg_emulated"] = pcg_mujoco(As, Ps, b, maxit=50000, tmax=600)
    ev = sla.eigh(A, M, eigvals_only=True)
    rec["cond_MinvA"] = float(ev.max() / ev.min())
    rec["mu_min_max_MinvA"] = [float(ev.min()), float(ev.max())]
    t0 = time.time()
    rec["mujoco_min_iterations"] = mj_min_iters(m)
    rec["mujoco_time_s"] = time.time() - t0
    rec["match"] = rec["mujoco_min_iterations"] == rec["pcg_emulated"].get("iterations")
    # one mj_step with the default 100 iterations: warning?
    m.opt.iterations = 100
    d2 = mujoco.MjData(m); mujoco.mj_step(m, d2)
    rec["warns_at_default_100"] = int(d2.warning[mujoco.mjtWarning.mjWARN_INERTIA].number) > 0
    return rec


if __name__ == "__main__":
    out = sys.argv[1]
    cfgs = [dict(variant="full_raw", nt=2, nz=3, nu=0.45), dict(variant="full_corr", nt=2, nz=3, nu=0.45),
            dict(variant="full_corr", nt=2, nz=3, nu=0.49), dict(variant="full_corr", nt=1, nz=2, nu=0.45),
            dict(variant="full_corr", nt=2, nz=5, nu=0.45), dict(variant="trilinear", nt=2, cz=2, nu=0.45),
            dict(variant="quadratic", nt=1, cz=1, nu=0.45)]
    res = []
    for c in cfgs:
        for joint in ("none", "slide", "hinge"):
            t0 = time.time()
            try:
                r = run(c, joint)
            except Exception as e:
                r = dict(id=cfg_id(c), joint=joint, error=repr(e))
            r["wall_s"] = time.time() - t0
            print(json.dumps({k: v for k, v in r.items() if k != "pcg_emulated"}), r.get("pcg_emulated"), flush=True)
            res.append(r)
            json.dump(to_json(res), open(out, "w"), indent=1)
