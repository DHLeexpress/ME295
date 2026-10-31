"""Replay the grasp from a saved diag2 state and trace the PDMS/box contact per step.
python contact_trace.py '<json>'  json: state (npz from grasp_diag2), tag, T (s), every, island, f1, ctrl_target (255), pymj via env
Per logged step: flex/blade/other normal force on the box, n contacts, PDMS compression (free-face vertex z in the pdms
frame minus rest, max), n inverted tets, max flex vertex speed, solver iterations."""
import sys, json, numpy as np, mujoco
from common import make_spec, compile_spec, nbad, reset_home
o = json.loads(sys.argv[1])
st0 = np.load(o["state"]); so = json.loads(str(st0["opts"])); so.update({k: v for k, v in o.items() if k in ("f1", "island", "E", "nu", "h", "it", "tol")})
spec = make_spec(so.get("grid", "11x15x3"), so)
if so.get("island", True) is False:
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
m, d = compile_spec(spec)
reset_home(m, d)
bg = m.geom("box").id
F = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, f"{s}_pdms_flex") for s in ("left", "right")}
P = {s: m.body(f"{s}_pdms").id for s in ("left", "right")}
def local(s):
    f = F[s]; va, vn = m.flex_vertadr[f], m.flex_vertnum[f]
    return (d.flexvert_xpos[va:va + vn] - d.xpos[P[s]]) @ d.xmat[P[s]].reshape(3, 3)
def tetvol(s, X):
    f = F[s]; ea, en = m.flex_elemdataadr[f], m.flex_elemnum[f]
    el = m.flex_elem[ea:ea + 4 * en].reshape(en, 4); a, b, c, e = (X[el[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), e - a)
ref = {s: local(s) for s in F}; vref = {s: tetvol(s, ref[s]) for s in F}
free = {s: np.isclose(ref[s][:, 2], ref[s][:, 2].min()) for s in F}     # free-face vertices (z = -t/2)
# table/box geometry was moved by place_box_between_pdms: restore from opts not possible -> state carries box pose only.
if "table_pos" in st0:
    m.body_pos[m.body("table").id] = st0["table_pos"]; m.body_quat[m.body("table").id] = st0["table_quat"]
mujoco.mj_setState(m, d, st0["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m, d)
a = m.actuator("fingers").id
rate = 255.0 * 150.0 / 140.0
n = int(o.get("T", 0.3) / m.opt.timestep); f6 = np.zeros(6)
for k in range(n):
    d.ctrl[a] += np.clip(o.get("ctrl_target", 255) - d.ctrl[a], -rate * m.opt.timestep, rate * m.opt.timestep)
    mujoco.mj_step(m, d)
    if k % o.get("every", 10) and k != n - 1:
        continue
    acc = {}
    for i in range(d.ncon):
        c = d.contact[i]
        if bg not in (c.geom[0], c.geom[1]): continue
        fl = [x for x in c.flex if x >= 0]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, fl[0]) if fl else m.geom(c.geom[1] if c.geom[0] == bg else c.geom[0]).name
        mujoco.mj_contactForce(m, d, i, f6); q = acc.setdefault(name, [0, 0.0]); q[0] += 1; q[1] += f6[0]
    line = f"t {d.time:.4f} bad {nbad(d)} it {int(d.solver_niter[:max(1,d.nisland)].sum()):3d} ctrl {d.ctrl[a]:6.1f} "
    for s in F:
        X = local(s); dz = (X - ref[s])[:, 2]
        line += f"| {s[0]} comp max {1e3*dz[free[s]].max():6.3f}mm inv {(tetvol(s, X)/vref[s] <= 0).sum():4d} "
    line += "| " + " ".join(f"{k}:{v[0]}/{v[1]:.1f}N" for k, v in sorted(acc.items()))
    print(line, flush=True)
