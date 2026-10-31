"""Check (2)+(3): build the effective metric explicitly and compare with the operators the engine applies.

PYTHONPATH=./pymj python metric.py '<json>'   (needs the instrumented build: me295_effApply exported)
json: model = "min" (repro/min_<tag>.xml path given by xml) | "robot" ; xml ; f1 ; island_fix (env set by caller)
Outputs out/metric_<tag>.json (+ .npz with the matrices).
 M        = mj_fullM
 K_fd     = -d(qfrc_spring + qfrc_actuator)/dq  by central differences (mj_integratePos per dof), all dofs incl. arm
 D_fd     = -d(qfrc_passive + qfrc_actuator)/dqdot by central differences
 A_ref    = M + h D_fd + h^2 K_fd   (the discrete-integrator effective metric, symmetrised)
 S_mono   = engine stiffness/damping operator of the monolithic metric (me295_effApply island=-1, with_M=0)
 S_isl    = engine operator of the island metric (island 0), scattered to global dofs
"""
import sys, os, json, glob, ctypes, time
import numpy as np
import mujoco
from pathlib import Path
HERE = Path(__file__).resolve().parent
o = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
tag = o.get("tag", "metric")
lib = ctypes.CDLL(glob.glob(str(Path(mujoco.__file__).parent / "libmujoco.so*"))[0])
lib.me295_effApply.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

if o.get("model", "min") == "min":
    m = mujoco.MjModel.from_xml_path(o["xml"]); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
else:
    sys.path.insert(0, str(HERE))
    from common import load, reset_home
    m, d = load(o.get("grid", "11x15x3"), o); reset_home(m, d)
nsteps = o.get("presteps", 1)
for _ in range(nsteps):
    mujoco.mj_step(m, d)
mujoco.mj_forward(m, d)
nv, h = m.nv, m.opt.timestep
print("nv", nv, "nisland", d.nisland, "efm_active", d.efm_active, "nefc", d.nefc, flush=True)

def apply_engine(island):
    """columns of the engine operator S (without M)."""
    if island < 0:
        n, idx = nv, np.arange(nv)
    else:
        n = int(d.island_nv[island]); a = int(d.island_idofadr[island]); idx = np.array(d.map_idof2dof[a:a + n])
    S = np.zeros((nv, nv))
    e = np.zeros(n); r = np.zeros(n)
    for k in range(n):
        e[:] = 0; e[k] = 1
        lib.me295_effApply(m._address, d._address, r.ctypes.data, e.ctypes.data, island, 0)
        S[idx, idx[k]] = r
    return S, idx

S_mono, _ = apply_engine(-1)
S_isl, isl_idx = apply_engine(0) if d.nisland > 0 else (np.zeros((nv, nv)), np.arange(0))
M = np.zeros((nv, nv)); mujoco.mj_fullM(m, d, M)

# finite differences (state saved/restored), forces evaluated by mj_forward
q0, v0 = d.qpos.copy(), d.qvel.copy()
def forces():
    mujoco.mj_forward(m, d)
    return d.qfrc_spring.copy(), d.qfrc_damper.copy(), d.qfrc_actuator.copy()
eps_q, eps_v = o.get("eps_q", 1e-7), o.get("eps_v", 1e-6)
Kf = np.zeros((nv, nv)); Ka = np.zeros((nv, nv)); Dd = np.zeros((nv, nv)); Da = np.zeros((nv, nv))
dq = np.zeros(nv)
t0 = time.time()
for k in range(nv):
    out = []
    for s in (+1, -1):
        d.qpos[:] = q0; d.qvel[:] = v0
        dq[:] = 0; dq[k] = s * eps_q
        mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
        out.append(forces())
    Kf[:, k] = -(out[0][0] - out[1][0]) / (2 * eps_q)
    Ka[:, k] = -(out[0][2] - out[1][2]) / (2 * eps_q)
    out = []
    for s in (+1, -1):
        d.qpos[:] = q0; d.qvel[:] = v0; d.qvel[k] += s * eps_v
        out.append(forces())
    Dd[:, k] = -(out[0][1] - out[1][1]) / (2 * eps_v)
    Da[:, k] = -(out[0][2] - out[1][2]) / (2 * eps_v)
d.qpos[:] = q0; d.qvel[:] = v0; mujoco.mj_forward(m, d)
print("fd wall", time.time() - t0, flush=True)

fd = np.array(sorted({i for f in range(m.nflex) for b in set(m.flex_vertbodyid[m.flex_vertadr[f]:m.flex_vertadr[f] + m.flex_vertnum[f]])
                      if m.body_dofnum[b] and m.body_weldid[b] == b for i in range(m.body_dofadr[b], m.body_dofadr[b] + m.body_dofnum[b])}))
other = np.setdiff1d(np.arange(nv), fd)
sym = lambda A: 0.5 * (A + A.T)
S_flex_ref = h * h * Kf                         # flex elasticity part (springs)
S_full_ref = h * h * (Kf + Ka) + h * (Dd + Da)  # everything implicit
A_ref = sym(M + S_full_ref)
res = {"tag": tag, "o": o, "nv": nv, "n_flexdof": len(fd), "n_other": len(other), "nisland": int(d.nisland),
       "island_nv": int(len(isl_idx)), "h": h}
def rel(A, B, rows, cols):
    nb = np.linalg.norm(B[np.ix_(rows, cols)])
    return float(np.linalg.norm((A - B)[np.ix_(rows, cols)]) / nb) if nb > 0 else float(np.linalg.norm(A[np.ix_(rows, cols)]))
for name, S in (("mono", S_mono), ("isl", S_isl)):
    for ref_name, R in (("h2Kflex", S_flex_ref), ("full", S_full_ref)):
        res[f"relerr_{name}_vs_{ref_name}"] = {"flex-flex": rel(S, R, fd, fd), "flex-other": rel(S, R, fd, other) if len(other) else None,
                                               "other-other": rel(S, R, other, other) if len(other) else None,
                                               "all": rel(S, R, np.arange(nv), np.arange(nv))}
    res[f"norm_{name}_flexflex"] = float(np.linalg.norm(S[np.ix_(fd, fd)]))
for name, S in (("S_mono", S_mono), ("S_isl", S_isl), ("ref_full", S_full_ref)):
    res[f"absnorm_{name}"] = {"flex-flex": float(np.linalg.norm(S[np.ix_(fd, fd)])), "flex-other": float(np.linalg.norm(S[np.ix_(fd, other)])),
                              "other-other": float(np.linalg.norm(S[np.ix_(other, other)]))}
res["absnorm_mono_minus_ref"] = {"flex-other": float(np.linalg.norm((S_mono - S_full_ref)[np.ix_(fd, other)]))}
res["norm_h2Kfd_flexflex"] = float(np.linalg.norm(S_flex_ref[np.ix_(fd, fd)]))
res["norm_M_flexflex"] = float(np.linalg.norm(M[np.ix_(fd, fd)]))
res["Kfd_asym"] = float(np.linalg.norm(Kf - Kf.T) / np.linalg.norm(Kf))
for name, A in (("M", sym(M)), ("A_ref", A_ref), ("M+S_mono", sym(M + S_mono)), ("M+S_isl", sym(M + S_isl)),
                ("M+h2Kfd", sym(M + S_flex_ref))):
    w = np.linalg.eigvalsh(A)
    res[f"eig_{name}"] = {"min": float(w[0]), "max": float(w[-1]), "n_neg": int((w <= 0).sum())}
res["eig_Kfd_flex_min_max"] = [float(x) for x in np.linalg.eigvalsh(sym(Kf[np.ix_(fd, fd)]))[[0, -1]]]
# explicit-stability indicator of the island metric: generalized eigenvalues of (M + S_mono) wrt (M + S_isl)
w = np.linalg.eigvals(np.linalg.solve(sym(M + S_isl), sym(M + S_mono)))
res["max_eig_(M+S_isl)^-1(M+S_mono)"] = float(np.max(w.real))
out = HERE / "out" / f"metric_{tag}.json"
json.dump(res, open(out, "w"), indent=1)
np.savez_compressed(HERE / "out" / f"metric_{tag}.npz", M=M, Kf=Kf, Ka=Ka, Dd=Dd, Da=Da, S_mono=S_mono, S_isl=S_isl, fd=fd)
print(json.dumps(res), flush=True)
