"""Torsional grip capacity: squeeze the test box, lift it, ramp a torque about the jaw axis, report the torque at which
it turns 1 and 5 deg relative to the pad, and the contacts carrying the grip. argv: repo_root force_scale tag"""
import json, sys, time
root, scale, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3]
sys.path.insert(0, root + "/src"); sys.path.insert(0, root + "/scripts_sim")
import mujoco, numpy as np
import sim_gripper as sg
spec = mujoco.MjSpec.from_file(str(sg.SCENE))
half = (0.025, 0.04, 0.03)
spec.geom("box").size = list(half); spec.geom("table_top").size = [0.6 * half[0], 0.6 * half[1], 0.15]
m = spec.compile(); d = mujoco.MjData(m)
fa = m.actuator("fingers").id
m.actuator_forcerange[fa] *= scale
sg.place_box_between_pdms(m, d)
sg.gripper_ramp(m, d, 255, 1.2)
sg.ik_lift(m, d, 0.05, 1.0)
sg.step(m, d, 0.3)
box = m.jnt_bodyid[m.joint("box").id]; pad = m.body("left_pdms").id
jaw = d.site("right_pdms_surface").xpos - d.site("left_pdms_surface").xpos; jaw /= np.linalg.norm(jaw)
R0 = d.xmat[box].reshape(3, 3) @ d.xmat[pad].reshape(3, 3).T
def contacts():
    out = {}
    for side in ("left", "right"):
        pts, fn = [], []
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1] if c.geom1 >= 0 else -1
            b2 = m.geom_bodyid[c.geom2] if c.geom2 >= 0 else -1
            if box not in (b1, b2): continue
            other = c.geom2 if b1 == box else c.geom1
            flex = c.flex[1] if b1 == box else c.flex[0]
            name = m.geom(other).name if other >= 0 else (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, flex) or "")
            if not name.startswith(side + "_pdms"): continue
            f6 = np.zeros(6); mujoco.mj_contactForce(m, d, i, f6)
            pts.append(c.pos.copy()); fn.append(f6[0])
        pts, fn = np.array(pts), np.array(fn)
        if len(fn) and fn.sum() > 0:
            cop = (pts * fn[:, None]).sum(0) / fn.sum()
            r = np.linalg.norm(np.cross(pts - cop, jaw), axis=1)      # distance from the axis through the CoP
            out[side] = dict(n=int(len(fn)), N=float(fn.sum()), n_active=int((fn > 1e-3 * fn.sum()).sum()),
                             r_mean_mm=float(1e3 * (fn * r).sum() / fn.sum()), r_max_mm=float(1e3 * r.max()))
        else:
            out[side] = dict(n=int(len(fn)), N=0.0)
    return out
c0 = contacts()
res = dict(tag=tag, scale=scale, contacts_before=c0, nflex=int(m.nflex), solver=int(m.opt.solver))
t1 = t5 = None; rate = 2.0; tmax = 8.0; t0 = time.time(); nsteps = int(tmax / rate / m.opt.timestep)
for k in range(nsteps):
    tau = rate * k * m.opt.timestep
    d.xfrc_applied[box, 3:] = tau * jaw
    mujoco.mj_step(m, d)
    R = d.xmat[box].reshape(3, 3) @ d.xmat[pad].reshape(3, 3).T @ R0.T
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if t1 is None and ang > 1: t1 = tau
    if t5 is None and ang > 5: t5 = tau; break
    if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: res["diverged"] = True; break
res.update(torque_at_1deg=t1, torque_at_5deg=t5, wall_s=round(time.time() - t0, 1),
           note=f"capacity estimate mu*sum(N_i r_i) = {sum(1.0 * c0[s].get('N', 0) * c0[s].get('r_mean_mm', 0) * 1e-3 for s in c0):.3f} N m")
print(json.dumps(res))
