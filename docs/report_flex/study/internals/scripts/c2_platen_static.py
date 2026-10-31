"""Item 3b: static normal stiffness of the layer under a rigid platen, and what the contact model adds.

Static rig (flex pinned to a world-fixed body = MuJoCo's "simple" path), mocap box platen covering the whole
free face pushed in by delta (case C1/C1f geometry), discrete integrator, stepped to rest. Compared with
reference solves using MuJoCo's own assembled stiffness K (= efm_K / h^2 at rest):
  K_vertex_noslip : free-face vertices prescribed u = (0,0,delta)            (C1 with the flex's K)
  K_vertex_frictionless : only u_z prescribed on the free face             (C1f with the flex's K)
  K_bound = M_eff A / t with M_eff from (2 lambda, mu) and from (lambda, mu).
Sweeps: contact time constant (refsafe off), refsafe on, plane (vertex contacts) vs box (element contacts),
friction 0/1.  Outputs: out/c2_platen_static.json
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import mujoco

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import load, dump, lame
from c1_contact_census import layer_xml

A, B, T = 0.051, 0.071, 0.0015
ZF = 0.1 - T / 2


def efmK(m, d):
    """Assembled flex stiffness (nv x nv, sparse) from the discrete integrator's CSR (scaled by h^2)."""
    h = m.opt.timestep
    rows, cols, vals = [], [], []
    for i in range(m.nv):
        n = d.efm_K_rownnz[i]
        a = d.efm_K_rowadr[i]
        rows += [i] * n
        cols += list(d.efm_K_colind[a:a + n])
        vals += list(d.efm_K_val[a:a + n])
    return sp.csr_matrix((np.array(vals) / h ** 2, (rows, cols)), shape=(m.nv, m.nv))


def reference(m, d, delta, noslip=True):
    """Static FE solve with MuJoCo's K: prescribe the free-face vertex displacement, return the force."""
    K = efmK(m, d)
    f = 0
    va, n = m.flex_vertadr[f], m.flex_vertnum[f]
    X = d.flexvert_xpos[va:va + n]
    vb = m.flex_vertbodyid[va:va + n]
    face = np.isclose(X[:, 2], ZF)
    pres = []
    for v in np.where(face)[0]:
        a = m.body_dofadr[vb[v]]
        pres += [a, a + 1, a + 2] if noslip else [a + 2]
    pres = np.array(pres)
    free = np.setdiff1d(np.arange(m.nv), pres)
    free = free[np.asarray(K[free][:, free].diagonal()).ravel() > 0]
    up = np.zeros(len(pres))
    up[(np.arange(len(pres)) % 3 == 2) if noslip else slice(None)] = delta
    Kff = K[free][:, free].tocsc()
    uf = spla.spsolve(Kff, -K[free][:, pres] @ up)
    u = np.zeros(m.nv)
    u[pres] = up
    u[free] = uf
    react = K @ u
    Fz = react[pres][(np.arange(len(pres)) % 3 == 2) if noslip else slice(None)].sum()
    return float(Fz / delta), K


def run(tool="box", delta=5e-6, solref="0.0001 1", refsafe=False, mu=1.0, condim=3, nstep=600, iterations=100,
        tolerance=1e-8, solimp=None):
    if tool == "box":
        geom = '<geom type="box" size="0.03 0.04 0.005" contype="2" conaffinity="2"/>'
        top = 0.005
    else:
        geom = '<geom type="plane" size="0.1 0.1 0.01" contype="2" conaffinity="2"/>'
        top = 0.0
    xml = layer_xml(geom, ZF + delta - top, solref=solref, refsafe=refsafe, iterations=iterations,
                    tolerance=tolerance)
    xml = xml.replace('friction="1 0.005 0.0001"', f'friction="{mu} 0.005 0.0001"').replace(
        'condim="3"', f'condim="{condim}"')
    if solimp:
        xml = xml.replace('solimp="0.99 0.99 0.001"', f'solimp="{solimp}"')
    m, _ = load(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    va, n = m.flex_vertadr[0], m.flex_vertnum[0]
    face = np.isclose(d.flexvert_xpos[va:va + n, 2], ZF)
    hist = []
    t0 = time.time()
    for k in range(nstep):
        mujoco.mj_step(m, d)
        if k % 50 == 49 or k == nstep - 1:
            F = 0.0
            for i in range(d.ncon):
                c6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, c6)
                F += c6[0]
            hist.append(F)
    wall = (time.time() - t0) / nstep
    X = d.flexvert_xpos[va:va + n]
    uz_face = X[face, 2] - ZF
    dist = np.array([d.contact[i].dist for i in range(d.ncon)])
    fn = np.array([0.0] * d.ncon)
    for i in range(d.ncon):
        c6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, c6)
        fn[i] = c6[0]
    act = fn > 0
    W = mujoco.mjtWarning
    out = dict(tool=tool, delta=delta, solref=solref, refsafe=refsafe, mu=mu, condim=condim, solimp=solimp,
               K_n_N_per_um=hist[-1] / delta * 1e-6, F_hist=hist, converged_rel=(abs(hist[-1] - hist[-2]) / max(abs(hist[-1]), 1e-300)) if len(hist) > 1 else None,
               face_uz_mean_over_delta=float(uz_face.mean() / delta), face_uz_min=float(uz_face.min() / delta),
               face_uz_max=float(uz_face.max() / delta), ncon=int(d.ncon), nactive=int(act.sum()),
               pen_mean_over_delta=float(-dist[act].mean() / delta) if act.any() else None,
               k_contact_measured=float(np.median(fn[act] / (-dist[act]))) if act.any() else None,
               solver_niter=int(d.solver_niter[0]), warn_inertia=int(d.warning[W.mjWARN_INERTIA].number),
               warn_badqacc=int(d.warning[W.mjWARN_BADQACC].number), ms_per_step=wall * 1e3)
    i0 = d.contact[0].efc_address if d.ncon else -1
    if i0 >= 0:
        K_, B_, I_, P_ = d.efc_KBIP.reshape(-1, 4)[i0]
        f_impl = 1 + m.opt.timestep * B_ + m.opt.timestep ** 2 * K_ * I_
        diagA_raw = d.efc_diagA[i0] * f_impl
        out.update(K=float(K_), B=float(B_), imp=float(I_), implicit_factor=float(f_impl),
                   diagA_raw=float(diagA_raw), k_contact_pred=float(K_ * I_ * I_ / ((1 - I_) * diagA_raw)))
    return out, m, d


def main():
    lam, mu_ = lame(1e6, 0.45)
    bound_doubled = (2 * lam + 2 * mu_) * A * B / T * 1e-6
    bound_true = (lam + 2 * mu_) * A * B / T * 1e-6
    res = dict(bound_true_N_per_um=bound_true, bound_doubled_N_per_um=bound_doubled, runs=[])
    # reference solves with MuJoCo's own stiffness
    r0, m0, d0 = run(nstep=1)
    Kns, _ = reference(m0, d0, 5e-6, noslip=True)
    Kfl, _ = reference(m0, d0, 5e-6, noslip=False)
    res["K_vertex_noslip_N_per_um"] = Kns * 1e-6
    res["K_vertex_frictionless_N_per_um"] = Kfl * 1e-6
    print(f"bound (lambda): {bound_true:.3f}  bound (2 lambda): {bound_doubled:.3f}  "
          f"vertex-prescribed no-slip: {Kns*1e-6:.3f}  frictionless: {Kfl*1e-6:.3f} N/um")
    cases = [
        dict(tool="box"), dict(tool="plane"),
        dict(tool="box", mu=0.0, condim=1), dict(tool="plane", mu=0.0, condim=1),
        dict(tool="box", refsafe=True), dict(tool="plane", refsafe=True),
        dict(tool="box", solref="0.002 1"), dict(tool="box", solref="0.00001 1"), dict(tool="box", solref="0.000001 1"),
        dict(tool="plane", solref="0.00001 1"), dict(tool="plane", solref="0.000001 1"),
        dict(tool="box", iterations=1000, tolerance=1e-12),
        dict(tool="box", solimp="0.9 0.95 0.001"),
    ]
    for c in cases:
        r, _, _ = run(**c)
        res["runs"].append(r)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "F_hist"})
    p = dump("c2_platen_static.json", res)
    print("wrote", p)


if __name__ == "__main__":
    main()
