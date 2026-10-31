"""Grasp sequence (as grasp.py) with geometric diagnostics at each phase; saves the full state per phase.
python grasp_diag2.py '<json>'  -> out/diag2_<tag>_<phase>.npz (mjSTATE_INTEGRATION) and printed report.
Per phase: contacts on the box per part with positions/normals in the left_pdms / right_pdms frames (x across blade,
y along finger, z into the layer; free face at z = -0.75 mm), box orientation in the left_pdms frame, flex
displacement extremes (vertex position in the pdms frame minus reference) and the smallest tetrahedron volume ratio."""
import sys, json, numpy as np, mujoco
from common import make_spec, compile_spec, nbad, warns
import sim_gripper as sg
o = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
tag = o.get("tag", "d2")
o.setdefault("box_half", [0.025, 0.04, 0.03])
spec = make_spec(o.get("grid", "11x15x3"), o)
if o.get("island", True) is False:
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
m, d = compile_spec(spec)
bg, bb = m.geom("box").id, m.body("box").id
F = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, f"{s}_pdms_flex") for s in ("left", "right")}
P = {s: m.body(f"{s}_pdms").id for s in ("left", "right")}

def frame(s):
    return d.xpos[P[s]].copy(), d.xmat[P[s]].reshape(3, 3).copy()

def flexinfo(s, ref=None):
    f = F[s]; va, vn = m.flex_vertadr[f], m.flex_vertnum[f]
    p0, R = frame(s)
    X = (d.flexvert_xpos[va:va + vn] - p0) @ R            # vertex positions in the pdms frame
    ea, en = m.flex_elemdataadr[f], m.flex_elemnum[f]
    el = m.flex_elem[ea:ea + 4 * en].reshape(en, 4)
    def vol(Y):
        a, b, c, e = (Y[el[:, k]] for k in range(4))
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), e - a) / 6
    return X, vol(X), vol

mujoco.mj_forward(m, d)
ref = {s: flexinfo(s) for s in ("left", "right")}            # reference at load (undeformed, after mj_forward)

def report(label):
    out = {}
    f6 = np.zeros(6)
    print(f"--- {label}: t={d.time:.3f} ncon={d.ncon} bad={nbad(d)} box_z={d.xpos[bb][2]:.4f}")
    for i in range(d.ncon):
        c = d.contact[i]
        if bg not in (c.geom[0], c.geom[1]):
            continue
        fl = [x for x in c.flex if x >= 0]
        if fl:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, fl[0])
        else:
            other = c.geom[1] if c.geom[0] == bg else c.geom[0]
            name = m.geom(other).name
        side = "left" if name.startswith("left") else "right" if name.startswith("right") else "left"
        p0, R = frame(side)
        mujoco.mj_contactForce(m, d, i, f6)
        q = out.setdefault(name, {"n": 0, "fn": 0.0, "pos": [], "nrm": [], "dist": []})
        q["n"] += 1; q["fn"] += f6[0]; q["pos"].append((c.pos - p0) @ R * 1e3); q["nrm"].append(c.frame[:3] @ R); q["dist"].append(c.dist * 1e3)
    for name, q in sorted(out.items()):
        P_ = np.array(q["pos"]); N_ = np.array(q["nrm"])
        print(f"   {name:16s} n={q['n']:3d} sum_fn={q['fn']:8.2f} N  pos(mm, pdms frame) x[{P_[:,0].min():6.1f},{P_[:,0].max():6.1f}] "
              f"y[{P_[:,1].min():6.1f},{P_[:,1].max():6.1f}] z[{P_[:,2].min():6.2f},{P_[:,2].max():6.2f}]  mean normal {np.round(N_.mean(0),3)}  dist[{min(q['dist']):.3f},{max(q['dist']):.3f}]mm")
    p0, R = frame("left")
    Rb = R.T @ d.xmat[bb].reshape(3, 3)
    print(f"   box centre in left_pdms frame (mm) {np.round((d.xpos[bb]-p0)@R*1e3,2)}; box axes in left frame (cols) {np.round(Rb,3).tolist()}")
    for s in ("left", "right"):
        X, v, vol = flexinfo(s)
        u = (X - ref[s][0]) * 1e3
        print(f"   {s} flex: |u| max {np.abs(u).max(0).round(3)} mm (x,y,z); u_z min/max {u[:,2].min():.3f}/{u[:,2].max():.3f}; "
              f"min vol ratio {np.min(v/ref[s][1]):.3f}  n_inverted {(v/ref[s][1] <= 0).sum()}")
    st = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
    tb = m.body("table").id
    np.savez(f"out/diag2_{tag}_{label}.npz", state=st, opts=json.dumps(o), table_pos=m.body_pos[tb].copy(), table_quat=m.body_quat[tb].copy())

sg.place_box_between_pdms(m, d)
if o.get("align"):   # same as grasp.py align
    nrm = d.xmat[m.body("left_pdms").id].reshape(3, 3)[:, 2].copy(); nrm[2] = 0; nrm /= np.linalg.norm(nrm)
    yaw = np.arctan2(nrm[1], nrm[0]); qb = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    a = m.jnt_qposadr[m.joint("box").id]; d.qpos[a + 3:a + 7] = qb; m.body_quat[m.body("table").id] = qb
    mujoco.mj_forward(m, d)
report("placed")
sg.step(m, d, 0.2); report("settled")
for k in range(4):
    sg.gripper_ramp(m, d, 255 * (k + 1) / 4, o.get("close_s", 1.2) / 4); report(f"closing{k+1}")
sg.ik_lift(m, d, 0.05, 1.0); report("lifted")
sg.step(m, d, 0.3); report("held")
print("warnings", warns(d))
