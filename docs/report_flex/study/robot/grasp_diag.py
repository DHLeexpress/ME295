"""Run the grasp sequence (workaround: islands disabled, wheel MuJoCo) and dump the contact set on the box at the end:
which parts touch it, net world force per part, equilibrium check.  python grasp_diag.py '<json>'"""
import sys, json, numpy as np, mujoco
from common import make_spec, compile_spec, nbad, warns
import sim_gripper as sg
o = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
o.setdefault("box_half", [0.025, 0.04, 0.03])
spec = make_spec(o.get("grid", "11x15x3"), o)
if o.get("island", True) is False:
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
m, d = compile_spec(spec)
bg, bb = m.geom("box").id, m.body("box").id
def report(label):
    parts = {}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if bg not in (c.geom[0], c.geom[1]):
            continue
        if c.flex[0] >= 0 or c.flex[1] >= 0:
            fl = c.flex[0] if c.flex[0] >= 0 else c.flex[1]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, fl)
        else:
            other = c.geom[1] if c.geom[0] == bg else c.geom[0]
            name = m.geom(other).name or f"<{m.body(m.geom_bodyid[other]).name}>"
        mujoco.mj_contactForce(m, d, i, f6)
        fw = c.frame.reshape(3, 3).T @ f6[:3]          # force on the second object, world frame
        box_is_2 = (c.geom[1] == bg)
        fbox = fw if box_is_2 else -fw
        p = parts.setdefault(name, {"n": 0, "F_on_box": np.zeros(3), "sum_normal": 0.0, "dist_min": 1e9})
        p["n"] += 1; p["F_on_box"] += fbox; p["sum_normal"] += f6[0]; p["dist_min"] = min(p["dist_min"], c.dist)
    tot = sum((p["F_on_box"] for p in parts.values()), np.zeros(3))
    print(f"--- {label}: t={d.time:.3f} ncon={d.ncon} bad={nbad(d)} box_z={d.xpos[bb][2]:.4f} q_drv={d.joint('left_driver_joint').qpos[0]:.4f} "
          f"act_force={d.actuator_force[m.actuator('fingers').id]:.2f}")
    for k, p in sorted(parts.items()):
        print(f"   {k:28s} n={p['n']:4d} sum_normal={p['sum_normal']:8.2f} N  F_on_box(world)={np.round(p['F_on_box'],2)}  min dist={1e3*p['dist_min']:.3f} mm")
    print(f"   total contact force on box {np.round(tot,3)}  weight {m.body_mass[bb]*9.81:.3f} N  (sum should be ~ [0,0,+w] when static)")
    jaw = d.site("right_pdms_surface").xpos - d.site("left_pdms_surface").xpos
    print(f"   PDMS surface sites: left {np.round(d.site('left_pdms_surface').xpos,4)} right {np.round(d.site('right_pdms_surface').xpos,4)} gap {1e3*np.linalg.norm(jaw):.2f} mm; box at {np.round(d.xpos[bb],4)}")
sg.place_box_between_pdms(m, d); report("placed")
sg.step(m, d, 0.2); report("settled")
sg.gripper_ramp(m, d, 255, o.get("close_s", 1.2)); report("closed")
sg.ik_lift(m, d, 0.05, 1.0); report("lifted")
sg.step(m, d, 0.3); report("held")
print("warnings", warns(d))
