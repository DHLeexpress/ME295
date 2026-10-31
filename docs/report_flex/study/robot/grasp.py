"""Proof run: the UR10e + 2F-140 with the dof="full" PDMS flex on both blades closes on a 300 g box and lifts it.

python grasp.py '<json options>'      (run against the instrumented build: PYTHONPATH=./pymj, MJ_FIX_ISLANDFLEX=1)
options: grid (11x15x3), island (default true; false = <flag island="disable"/>, the no-patch workaround),
         f1 (F1-corrected material inputs), nu, E, it, tol, h, memory, close_s (1.2), lift_dz (0.05), lift_s (1.0),
         hold_s (0.3), settle_s (0.2), save (npz path for the time series), stop_on_bad (true),
         align (true: yaw the box onto the PDMS normal instead of the site-to-site vector), flex_solref [tc, dr], flex_solimp [5],
         slow_from (ctrl at which to switch to speed_slow mm/s, default 20; fast_s, speed_fast 150)
Sequence (scripts_sim/sim_gripper.py helpers): place_box_between_pdms (home pose, 0.5 s arm settle, box placed
between the PDMS faces on a pedestal) -> settle -> gripper_ramp to 255 (150 mm/s) -> ik_lift (flange straight
up) -> hold. Divergence = mjWARN_BADQACC count (MuJoCo auto-resets after NaN).
"""
import sys
import json
import time
import numpy as np
import mujoco
from common import make_spec, compile_spec, nbad, warns, flex_dofs
import sim_gripper as sg
from shear_agent.robot import PDMSSkin

o = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
o.setdefault("box_half", [0.025, 0.04, 0.03])
spec = make_spec(o.get("grid", "11x15x3"), o)
if o.get("island", True) is False:
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
for f_ in spec.flexes:
    if "flex_solref" in o:
        f_.solref = o["flex_solref"]
    if "flex_solimp" in o:
        f_.solimp = o["flex_solimp"]
m, d = compile_spec(spec)
mujoco.mj_forward(m, d)


def flex_geom(s_):
    """vertex positions in the pdms frame and tetrahedron (signed) volumes."""
    f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, f"{s_}_pdms_flex"); ea, en = m.flex_elemdataadr[f], m.flex_elemnum[f]
    el = m.flex_elem[ea:ea + 4 * en].reshape(en, 4); va = m.flex_vertadr[f]; b = m.body(f"{s_}_pdms").id
    X = (d.flexvert_xpos[va:va + m.flex_vertnum[f]] - d.xpos[b]) @ d.xmat[b].reshape(3, 3)
    A, B, C, E_ = (X[el[:, k]] for k in range(4))
    return X, np.einsum("ij,ij->i", np.cross(B - A, C - A), E_ - A)


GREF = {s_: flex_geom(s_) for s_ in ("left", "right")}       # rest state (qpos0)
FREE = {s_: np.isclose(GREF[s_][0][:, 2], GREF[s_][0][:, 2].min()) for s_ in GREF}
box_g, box_b = m.geom("box").id, m.body("box").id
fdofs = flex_dofs(m)
flex_side = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, f"{s}_pdms_flex"): s for s in ("left", "right")}


def box_forces():
    """Contact force on the box from each PDMS flex: normal (N, >0 squeezing), tangential (N), count."""
    out = {s: np.zeros(3) for s in ("left", "right", "other")}
    out["rigid"] = 0.0
    out["pen"] = 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if box_g not in (c.geom[0], c.geom[1]):
            continue
        fl = [f for f in c.flex if f >= 0]
        side = flex_side.get(fl[0], "other") if fl else "other"
        mujoco.mj_contactForce(m, d, i, f6)
        if fl:
            out["pen"] = max(out["pen"], -c.dist)
        if not fl and "blade" in (m.geom(c.geom[0]).name + m.geom(c.geom[1]).name):
            out["rigid"] += f6[0]
        out[side] += [f6[0], np.linalg.norm(f6[1:3]), 1]
    return out


rec = {k: [] for k in ("t", "box_z", "q_drv", "ctrl", "nL", "tL", "cL", "nR", "tR", "cR", "nO", "ncon", "nefc",
                       "niter", "bad", "wall_ms", "vflex", "rigid_n", "compL", "compR", "invL", "invR", "pen")}
phase_t = {}
t_wall0 = time.time()
stop = {"bad": None, "fatal": None}


def logstep(wall):
    f = box_forces()
    rec["t"].append(d.time); rec["box_z"].append(d.xpos[box_b][2])
    rec["q_drv"].append(d.joint("left_driver_joint").qpos[0]); rec["ctrl"].append(d.ctrl[m.actuator("fingers").id])
    for k, s in (("L", "left"), ("R", "right")):
        rec["n" + k].append(f[s][0]); rec["t" + k].append(f[s][1]); rec["c" + k].append(f[s][2])
    rec["nO"].append(f["other"][2]); rec["ncon"].append(d.ncon); rec["nefc"].append(d.nefc)
    rec["niter"].append(int(d.solver_niter[:max(1, d.nisland)].sum())); rec["bad"].append(nbad(d))
    rec["wall_ms"].append(wall); rec["vflex"].append(float(np.abs(d.qvel[fdofs]).max()))
    rec["rigid_n"].append(f["rigid"]); rec["pen"].append(f["pen"])
    for k, s_ in (("L", "left"), ("R", "right")):
        X, v = flex_geom(s_)
        rec["comp" + k].append(float((X - GREF[s_][0])[FREE[s_], 2].max()))
        rec["inv" + k].append(int((v / GREF[s_][1] <= 0).sum()))


orig_step = mujoco.mj_step
nstep = [0]


def step_logged(m_, d_):
    t = time.time()
    orig_step(m_, d_)
    wall = 1e3 * (time.time() - t)
    nstep[0] += 1
    if nbad(d_) and stop["bad"] is None:
        stop["bad"] = d_.time
        if o.get("stop_on_bad", True):
            raise RuntimeError("BADQACC")
    if nstep[0] % o.get("every", 10) == 0:
        logstep(wall)


mujoco.mj_step = step_logged   # sim_gripper's helpers call mujoco.mj_step
try:
    sg.place_box_between_pdms(m, d)            # includes a 0.5 s arm settle (box parked)
    if o.get("align"):
        # sim_gripper yaws the box along the site-to-site vector, which is 0.48 deg off the PDMS normal
        # (out/diag2_noisl_f1.txt / NOTES.md): over the 51 mm pad width that is a 0.43 mm wedge, enough to push
        # the StVK layer through its inversion point at the leading edge. Align the box faces with the PDMS.
        nrm = d.xmat[m.body("left_pdms").id].reshape(3, 3)[:, 2].copy(); nrm[2] = 0; nrm /= np.linalg.norm(nrm)
        yaw = np.arctan2(nrm[1], nrm[0]); qb = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        a = m.jnt_qposadr[m.joint("box").id]; d.qpos[a + 3:a + 7] = qb; m.body_quat[m.body("table").id] = qb
        mujoco.mj_forward(m, d)
    phase_t["box_yaw_deg"] = float(np.degrees(2 * np.arctan2(d.qpos[m.jnt_qposadr[m.joint("box").id] + 6], d.qpos[m.jnt_qposadr[m.joint("box").id] + 3])))
    phase_t["placed"] = d.time
    sg.step(m, d, o.get("settle_s", 0.2)); phase_t["settled"] = d.time
    if "slow_from" in o:   # fast approach, then slow closing through contact
        sg.gripper_ramp(m, d, o["slow_from"], o.get("fast_s", 1.0), speed=o.get("speed_fast", 150.0)); phase_t["approached"] = d.time
        sg.gripper_ramp(m, d, 255, o.get("close_s", 1.2), speed=o.get("speed_slow", 20.0))
    else:
        sg.gripper_ramp(m, d, 255, o.get("close_s", 1.2), speed=o.get("speed_fast", 150.0))
    phase_t["closed"] = d.time
    z_closed = float(d.xpos[box_b][2])
    sg.ik_lift(m, d, o.get("lift_dz", 0.05), o.get("lift_s", 1.0)); phase_t["lifted"] = d.time
    sg.step(m, d, o.get("hold_s", 0.3)); phase_t["held"] = d.time
except RuntimeError as e:
    stop["fatal"] = str(e)
except mujoco.FatalError as e:
    stop["fatal"] = str(e).splitlines()[0][:150]
finally:
    mujoco.mj_step = orig_step

wall = time.time() - t_wall0
R = {k: np.array(v) for k, v in rec.items()}
res = {"o": o, "mujoco_file": mujoco.__file__, "nv": m.nv, "steps": nstep[0], "sim_time": d.time, "wall_s": wall,
       "phase_t": phase_t, "bad": nbad(d), "first_bad_t": stop["bad"], "fatal": stop["fatal"], "warnings": warns(d)}
if len(R["t"]):
    t0 = phase_t.get("settled", 0.0)
    sl = R["t"] >= t0
    res.update({
        "ms_per_step_all": float(R["wall_ms"].mean()),
        "ms_per_step_after_place": float(R["wall_ms"][sl].mean()) if sl.any() else None,
        "ms_per_step_contact": float(R["wall_ms"][R["cL"] + R["cR"] > 0].mean()) if (R["cL"] + R["cR"] > 0).any() else None,
        "max_ncon": int(R["ncon"].max()), "max_nefc": int(R["nefc"].max()), "mean_niter": float(R["niter"].mean()),
        "max_niter": int(R["niter"].max()), "max_vflex": float(R["vflex"].max())})
    if "held" in phase_t:
        last = R["t"] >= phase_t["held"] - 0.1
        box_lift = float(d.xpos[box_b][2] - z_closed)
        res["held_window"] = {k: [float(R[k][last].mean()), float(R[k][last].std())] for k in
                              ("nL", "nR", "rigid_n", "compL", "compR", "invL", "invR", "niter", "pen")}
        res["max_over_run"] = {k: float(R[k].max()) for k in ("nL", "nR", "rigid_n", "compL", "compR", "invL", "invR")}
        res.update({"final": {k: float(R[k][last].mean()) for k in ("nL", "tL", "cL", "nR", "tR", "cR", "nO")},
                    "box_lift_mm": 1e3 * box_lift, "lift_cmd_mm": 1e3 * o.get("lift_dz", 0.05),
                    "box_weight_N": float(m.body_mass[box_b] * 9.81)})
        frL = PDMSSkin(m, "left").read(d)
        res["skin_left"] = {"total_normal": frL.total_normal, "total_shear": frL.total_shear.tolist(),
                            "normal_taxels": np.round(frL.normal, 2).tolist(),
                            "max_compression_um": float(1e6 * frL.deformation[..., 2].max()),
                            "max_shear_um": float(1e6 * np.linalg.norm(frL.deformation[..., :2], axis=-1).max())}
# rigid-blade share of the box contact force at the end (should be ~0 if the PDMS carries the grasp)
f6 = np.zeros(6); rigid = 0.0; pdms = 0.0
for i in range(d.ncon):
    c = d.contact[i]
    if box_g not in (c.geom[0], c.geom[1]):
        continue
    mujoco.mj_contactForce(m, d, i, f6)
    if c.flex[0] >= 0 or c.flex[1] >= 0:
        pdms += f6[0]
    elif "blade" in (m.geom(c.geom[0]).name + m.geom(c.geom[1]).name):
        rigid += f6[0]
res["end_normal_N"] = {"pdms": pdms, "rigid_blade": rigid}
inv = {s_: int((flex_geom(s_)[1] / GREF[s_][1] <= 0).sum()) for s_ in ("left", "right")}
res["inverted_tets_end"] = inv
if o.get("save"):
    np.savez(o["save"], **R, phase=json.dumps(phase_t))
print(json.dumps(res, default=float), flush=True)
