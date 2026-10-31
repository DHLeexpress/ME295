"""contact-video simulation: the 51 x 71 x 1.5 mm PDMS pad (MuJoCo 3.17 dof="full" flex, F1-corrected inputs for
E = 1 MPa, nu = 0.45, back face pinned, no material damping) touched by one rigid object per segment:
approach -> press (until a target normal force or an indentation cap) -> hold -> slide (shear) -> hold -> lift off.

Pad frame (as in bench_dynamic/rig.py and the robot): origin at the centre of the FREE face, x across (51 mm),
y along (71 mm), +z into the layer; the layer is 0 <= z <= T. For rendering, the rig is turned upside down in the
world (body "rig", 180 deg about x) so that the free face faces world +z and objects come down from above:
p_world = Rw p_pad with Rw = diag(1, -1, -1).

Usage: python sim.py <segment index> [--cells 40 56 3] [--out runs] [--steps N (test: stop early)]
Writes runs/seg<k>.npz (frames + per-step series) and runs/seg<k>.json (settings, metrics, checks).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
A, B, T = 0.051, 0.071, 0.0015
RHO = 965.0
NXT, NYT = 5, 7
DXT, DYT = A / NXT, B / NYT
MASS = RHO * A * B * T
RW = np.diag([1.0, -1.0, -1.0])            # pad frame -> world
E_T, NU_T = 1e6, 0.45                      # target material
SEED = 20261030

# contact: 'stiffest' preset of bench_dynamic (solref 1e-5 1, solimp 0.9999 ...), refsafe disabled; mu = 1
SOLREF = (1e-5, 1.0)
SOLVER = "CG"
SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2)
MU = 1.0
MASS_SCALE = 1.0   # x10 was tried (pen lower in benchmark C1f) but went unstable here (test/m10): keep the true mass

# timeline (s)
H = 5e-4
T_APP, T_PRESS, T_HOLD1, T_SLIDE, T_HOLD2, T_LIFT = 0.08, 0.40, 0.08, 0.40, 0.08, 0.12
GAP0 = 1.5e-3                 # start gap (m) between the object's lowest point and the free face
GAP1 = 0.02e-3                # gap at the end of the approach
DEPTH_CAP = 0.10e-3           # max travel past first touch (6.7 % of T); 0.25 mm inverted tets while sliding (runs_cap250/)
FRAME_EVERY = 18              # steps per video frame (9 ms of simulated time)


def remap_f1(E, nu):
    G = E / (2 * (1 + nu))
    nu_in = nu / (2 * (1 - nu))
    return 2 * G * (1 + nu_in), nu_in


def quat_from_R(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def R_align(a, b):
    """Rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def objects():
    """Deterministic pseudo-random objects. Orientation R maps object-local -> pad frame; the pad's free face is
    z = 0 and the object sits at z < 0 (the pad is at z > 0)."""
    rng = np.random.default_rng(SEED)
    out = []

    def place(margin_x=14e-3, margin_y=22e-3):
        return np.array([rng.uniform(-margin_x, margin_x), rng.uniform(-margin_y, margin_y)])

    def common(o):
        o["xy0"] = place()
        ang = rng.uniform(0, 2 * np.pi)
        o["slide_m"] = float(rng.uniform(2e-3, 4e-3))
        o["slide_dir"] = [float(np.cos(ang)), float(np.sin(ang))]
        o["F_target_N"] = float(rng.uniform(5, 20))
        o["xy0"] = o["xy0"].tolist()
        o["R"] = np.asarray(o["R"]).tolist()
        out.append(o)

    # 1 sphere
    r = rng.uniform(4e-3, 12e-3)
    common(dict(name="sphere", label=f"sphere R = {r*1e3:.1f} mm", type="sphere", size=[r], R=np.eye(3)))
    # 2 rod lying across (cylinder, axis in the pad plane, random yaw)
    r = rng.uniform(2e-3, 4e-3)
    yaw = rng.uniform(0, np.pi)
    common(dict(name="rod", label=f"rod lying across, R = {r*1e3:.1f} mm, yaw {np.degrees(yaw):.0f} deg",
                type="cylinder", size=[r, 0.045], R=Rz(yaw) @ Ry(np.pi / 2)))
    # 3 box edge: box rotated 45 deg about its x axis (edge along local x points to the pad), random yaw
    yaw = rng.uniform(0, np.pi)
    hx = rng.uniform(6e-3, 12e-3)
    common(dict(name="box_edge", label=f"box edge, length {2*hx*1e3:.0f} mm, yaw {np.degrees(yaw):.0f} deg",
                type="box", size=[hx, 8e-3, 8e-3], R=Rz(yaw) @ Rx(np.pi / 4)))
    # 4 book corner: cuboid 40 x 30 x 12 mm whose corner (+,+,+) direction points to the pad (+z), random yaw
    yaw = rng.uniform(0, 2 * np.pi)
    hs = np.array([20e-3, 15e-3, 6e-3])
    diag = hs / np.linalg.norm(hs)
    Rc = R_align(diag, np.array([0, 0, 1.0]))
    common(dict(name="book_corner", label=f"book corner (40 x 30 x 12 mm), yaw {np.degrees(yaw):.0f} deg",
                type="box", size=hs.tolist(), R=Rz(yaw) @ Rc))
    # 5 capsule tip: capsule axis tilted from the vertical by 0-30 deg
    r = rng.uniform(3e-3, 5e-3)
    tilt = rng.uniform(0, np.radians(30))
    yaw = rng.uniform(0, 2 * np.pi)
    common(dict(name="capsule_tip", label=f"capsule tip R = {r*1e3:.1f} mm, tilt {np.degrees(tilt):.0f} deg",
                type="capsule", size=[r, 10e-3], R=Rz(yaw) @ Rx(tilt)))
    # 6 ellipsoid, random orientation
    ax = np.sort([rng.uniform(8e-3, 14e-3), rng.uniform(4e-3, 8e-3), rng.uniform(3e-3, 6e-3)])[::-1]
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    Re = np.zeros(9)
    mujoco.mju_quat2Mat(Re, q)
    common(dict(name="ellipsoid", label=f"ellipsoid {2*ax[0]*1e3:.0f} x {2*ax[1]*1e3:.0f} x {2*ax[2]*1e3:.0f} mm",
                type="ellipsoid", size=ax.tolist(), R=Re.reshape(3, 3)))
    # 7 coin-like disk: thin cylinder nearly flat, tilted 1-4 deg
    r = rng.uniform(9e-3, 12e-3)
    tilt = np.radians(rng.uniform(1, 4))
    yaw = rng.uniform(0, 2 * np.pi)
    common(dict(name="coin", label=f"coin R = {r*1e3:.1f} mm, 1.8 mm thick, tilt {np.degrees(tilt):.1f} deg",
                type="cylinder", size=[r, 0.9e-3], R=Rz(yaw) @ Rx(tilt)))
    # 8 small cube: 8 mm, face down, random yaw, tilt 0-3 deg
    yaw = rng.uniform(0, np.pi / 2)
    tilt = np.radians(rng.uniform(0, 3))
    common(dict(name="cube", label=f"8 mm cube, yaw {np.degrees(yaw):.0f} deg, tilt {np.degrees(tilt):.1f} deg",
                type="box", size=[4e-3] * 3, R=Rz(yaw) @ Ry(tilt)))
    return out


def lowest_extent(o):
    """Support of the object in pad +z (toward the pad), from its centre: max over the body of z."""
    R = np.asarray(o["R"])
    n = R.T @ np.array([0, 0, 1.0])     # pad +z in object coordinates
    s = o["size"]
    t = o["type"]
    if t == "sphere":
        return s[0]
    if t == "box":
        return float(np.sum(np.abs(n) * np.asarray(s)))
    if t == "capsule":
        return abs(n[2]) * s[1] + s[0]
    if t == "cylinder":
        return abs(n[2]) * s[1] + s[0] * np.sqrt(max(0.0, 1 - n[2] ** 2))
    if t == "ellipsoid":
        return float(np.sqrt(np.sum((np.asarray(s) * n) ** 2)))
    raise ValueError(t)


def build_xml(o, cells, iterations, mass_scale=MASS_SCALE):
    cx, cy, cz = cells
    vx, vy, vz = cx + 1, cy + 1, cz + 1
    young, poisson = remap_f1(E_T, NU_T)
    sp = f"{A/cx:.10g} {B/cy:.10g} {T/cz:.10g}"
    si = " ".join(f"{v:g}" for v in SOLIMP)
    sr = f"{SOLREF[0]:g} {SOLREF[1]:g}"
    con = f'condim="3" friction="{MU:g} 0.005 0.0001" solref="{sr}" solimp="{si}"'
    q = quat_from_R(o["R"])
    size = " ".join(f"{v:.9g}" for v in o["size"])
    z0 = -(lowest_extent(o) + GAP0)
    x0, y0 = o["xy0"]
    rgba = "0.80 0.80 0.84 1" if o["type"] in ("box", "cylinder") else "0.75 0.78 0.85 1"
    xml = f"""<mujoco model="contact_video_{o['name']}">
<option timestep="{H}" integrator="discrete" solver="{SOLVER}" cone="elliptic" impratio="100" gravity="0 0 0"
        iterations="{iterations}" tolerance="1e-8"><flag refsafe="disable"/></option>
<size memory="2G"/>
<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/>
  <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/></visual>
<asset>
  <material name="blade" rgba="0.30 0.32 0.36 1" specular="0.3" shininess="0.4"/>
  <material name="obj" rgba="{rgba}" specular="0.6" shininess="0.6" reflectance="0.0"/>
</asset>
<worldbody>
  <light pos="0.05 -0.08 0.15" dir="-0.3 0.5 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
  <body name="rig" quat="0 1 0 0">
    <geom name="blade" type="box" size="0.0285 0.0385 0.002" pos="0 0 {T + 0.002 + 1e-5:.9g}" material="blade"
          contype="0" conaffinity="0"/>
    <body name="pad" pos="0 0 {T/2:.10g}">
      <flexcomp name="pdms" type="grid" dim="3" count="{vx} {vy} {vz}" spacing="{sp}" radius="0"
                mass="{MASS * mass_scale:.8g}" rgba="0.95 0.55 0.15 1">
        <elasticity young="{young:.10g}" poisson="{poisson:.10g}" damping="0"/>
        <contact contype="2" conaffinity="2" priority="2" {con} selfcollide="none"/>
        <pin gridrange="0 0 {vz-1} {vx-1} {vy-1} {vz-1}"/>
      </flexcomp>
    </body>
    <body name="obj" pos="{x0:.9g} {y0:.9g} {z0:.9g}">
      <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
      <joint name="ox" type="slide" axis="1 0 0"/><joint name="oy" type="slide" axis="0 1 0"/>
      <joint name="oz" type="slide" axis="0 0 1"/>
      <geom name="obj" type="{o['type']}" size="{size}" quat="{q[0]:.9g} {q[1]:.9g} {q[2]:.9g} {q[3]:.9g}"
            material="obj" contype="2" conaffinity="2" {con}/>
    </body>
  </body>
</worldbody>
<actuator>
  <position name="ax" joint="ox" kp="1e10" dampratio="1"/><position name="ay" joint="oy" kp="1e10" dampratio="1"/>
  <position name="az" joint="oz" kp="1e10" dampratio="1"/>
</actuator>
</mujoco>"""
    return xml


def ease(s):
    s = min(1.0, max(0.0, s))
    return 0.5 - 0.5 * np.cos(np.pi * s)


def contact_forces_pad(m, d, flexid=0):
    """Per-contact force ON THE LAYER and contact positions, both in the pad frame."""
    n = d.ncon
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    con = d.contact
    adr = np.asarray(con.efc_address[:n])
    dim = np.asarray(con.dim[:n])
    f = np.zeros((n, 3))
    ok = adr >= 0
    for dm in np.unique(dim[ok]):
        sel = ok & (dim == dm)
        k = min(dm, 3)
        f[sel, :k] = d.efc_force[adr[sel][:, None] + np.arange(k)]
    frames = np.asarray(con.frame[:n]).reshape(-1, 3, 3)
    fw = np.einsum("kij,ki->kj", frames, f)          # force on the second object, world
    flex = np.asarray(con.flex[:n])
    sign = np.where(flex[:, 1] >= 0, 1.0, -1.0)      # the layer is the second object -> keep, else flip
    fw *= sign[:, None]
    return fw @ RW, np.asarray(con.pos[:n]) @ RW, np.asarray(con.dist[:n]).copy()


def taxel_bin(fp, pp):
    i = np.clip(np.floor((pp[:, 0] + A / 2) / DXT).astype(int), 0, NXT - 1)
    j = np.clip(np.floor((pp[:, 1] + B / 2) / DYT).astype(int), 0, NYT - 1)
    tax = np.zeros((NYT, NXT, 3))
    np.add.at(tax, (j, i), fp)
    return tax


def run(k, cells, iterations, outdir, max_steps=None, mass_scale=MASS_SCALE):
    objs = objects()
    o = objs[k]
    xml = build_xml(o, cells, iterations, mass_scale)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"seg{k}.xml"), "w") as f:
        f.write(xml)
    t0 = time.perf_counter()
    m = mujoco.MjModel.from_xml_string(xml)
    compile_s = time.perf_counter() - t0
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    adr0, num0 = m.flex_vertadr[0], m.flex_vertnum[0]
    v0 = d.flexvert_xpos[adr0:adr0 + num0] @ RW            # rest positions, pad frame
    vx, vy, vz = cells[0] + 1, cells[1] + 1, cells[2] + 1
    # grid index (i*vy + j)*vz + kk; free face kk = 0 (pad z = 0)
    front = np.array([(i * vy + j) * vz for i in range(vx) for j in range(vy)])
    assert np.allclose(v0[front, 2], 0, atol=1e-9), v0[front, 2][:5]
    qadr = np.array([m.jnt_qposadr[m.joint(f"o{c}").id] for c in "xyz"])
    aid = np.array([m.actuator(f"a{c}").id for c in "xyz"])
    obj_geom = m.geom("obj").id
    el = m.flex_elem[m.flex_elemdataadr[0]:m.flex_elemdataadr[0] + 4 * m.flex_elemnum[0]].reshape(-1, 4)

    def tetvol(p):
        a, b, c, e = (p[el[:, ii]] for ii in range(4))
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), e - a) / 6.0

    vol0 = tetvol(d.flexvert_xpos[adr0:adr0 + num0])

    nst = [int(round(x / H)) for x in (T_APP, T_PRESS, T_HOLD1, T_SLIDE, T_HOLD2, T_LIFT)]
    names = ["approach", "press", "hold", "slide", "hold2", "lift"]
    nsteps = sum(nst)
    if max_steps:
        nsteps = min(nsteps, max_steps)
    press_rate = DEPTH_CAP / (0.85 * T_PRESS)              # m/s of setpoint travel during the press
    sdir = np.array(o["slide_dir"])
    ts = dict(t=np.zeros(nsteps), F=np.zeros((nsteps, 3)), Fact=np.zeros((nsteps, 3)), ncon=np.zeros(nsteps, int),
              niter=np.zeros(nsteps, int), wall=np.zeros(nsteps), setp=np.zeros((nsteps, 3)),
              obj=np.zeros((nsteps, 3)), pen_max=np.zeros(nsteps), pen_fw=np.zeros(nsteps), uz_max=np.zeros(nsteps), phase=np.zeros(nsteps, int),
              taxsum_err=np.zeros(nsteps))
    frames = dict(step=[], qpos=[], ufront=[], tax=[], cpos=[], cforce=[], minJ=[], ninv=[])
    gap1_travel = GAP0 - GAP1
    z_touch_setp = None            # setpoint (pad +z travel from start) at first contact
    press_z = None                 # frozen press setpoint
    sp = np.zeros(3)
    ph_bounds = np.cumsum([0] + nst)
    stop_reason = None
    nan_step = None
    s_tot = 0
    warn0 = {w: 0 for w in ("BADQACC", "CNSTRFULL", "CONTACTFULL")}
    for s in range(nsteps):
        ph = int(np.searchsorted(ph_bounds, s, side="right") - 1)
        ls = s - ph_bounds[ph]
        Fz_now = ts["F"][s - 1, 2] if s else 0.0
        if ph == 0:
            sp = np.array([0, 0, gap1_travel * ease((ls + 1) / nst[0])])
        elif ph == 1:
            if press_z is None:
                znew = sp[2] + press_rate * H
                if z_touch_setp is not None and (Fz_now >= o["F_target_N"] or znew - z_touch_setp >= DEPTH_CAP):
                    press_z = sp[2]
                    stop_reason = "force" if Fz_now >= o["F_target_N"] else "depth_cap"
                else:
                    sp = np.array([0, 0, znew])
            if press_z is not None:
                sp = np.array([0, 0, press_z])
        elif ph == 2:
            if press_z is None:
                press_z = sp[2]
                stop_reason = "time"
        elif ph == 3:
            e = ease((ls + 1) / nst[3])
            sp = np.array([sdir[0] * o["slide_m"] * e, sdir[1] * o["slide_m"] * e, press_z])
        elif ph == 4:
            pass
        elif ph == 5:
            e = ease((ls + 1) / nst[5])
            sp = np.array([sp[0], sp[1], press_z - e * (press_z + 0.6e-3)])
        d.ctrl[aid] = sp
        tw = time.perf_counter()
        mujoco.mj_step(m, d)
        ts["wall"][s] = time.perf_counter() - tw
        if (not np.isfinite(d.qpos).all()) or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            nan_step = s
            print(f"seg{k}: NaN/BADQACC at step {s} ({names[ph]})", flush=True)
            break
        fp, pp, dist = contact_forces_pad(m, d)
        Fsum = fp.sum(0)
        tax = taxel_bin(fp, pp)
        ts["taxsum_err"][s] = float(np.abs(tax.sum((0, 1)) - Fsum).max())
        ts["t"][s] = d.time
        ts["F"][s] = Fsum
        ts["Fact"][s] = d.actuator_force[aid]          # force of the servo on the object (pad-frame axes)
        ts["ncon"][s] = d.ncon
        ts["niter"][s] = int(np.sum(d.solver_niter[:max(1, d.nisland)]))
        ts["setp"][s] = sp
        ts["obj"][s] = d.qpos[qadr]
        ts["phase"][s] = ph
        ts["pen_max"][s] = -dist.min() if dist.size else 0.0
        fn = np.maximum(fp[:, 2], 0.0)
        ts["pen_fw"][s] = float((-dist * fn).sum() / fn.sum()) if fn.sum() > 0 else 0.0
        u = d.flexvert_xpos[adr0:adr0 + num0][front] @ RW - v0[front]
        ts["uz_max"][s] = u[:, 2].max()
        if z_touch_setp is None and d.ncon > 0:
            z_touch_setp = float(d.qpos[qadr][2])
        if s % FRAME_EVERY == 0 or s == nsteps - 1:
            frames["step"].append(s)
            frames["qpos"].append(d.qpos.copy())
            frames["ufront"].append(u.astype(np.float32))
            frames["tax"].append(tax)
            frames["cpos"].append(pp.astype(np.float32))
            frames["cforce"].append(fp.astype(np.float32))
            J = tetvol(d.flexvert_xpos[adr0:adr0 + num0]) / vol0
            frames["minJ"].append(float(J.min()))
            frames["ninv"].append(int((J <= 0).sum()))
        s_tot = s + 1
        if s % 200 == 0:
            print(f"seg{k} step {s} {names[ph]} Fz={Fsum[2]:.3f} ncon={d.ncon} iter={ts['niter'][s]} "
                  f"uzmax={u[:,2].max()*1e6:.1f}um pen={ts['pen_max'][s]*1e6:.2f}um wall={ts['wall'][s]*1e3:.0f}ms",
                  flush=True)
    for key in ts:
        ts[key] = ts[key][:s_tot]
    for w in warn0:
        warn0[w] = int(d.warning[getattr(mujoco.mjtWarning, "mjWARN_" + w)].number)
    # metrics
    F = ts["F"]
    sel_press = ts["phase"] == 2
    hold_idx = np.where(sel_press)[0]
    sel_hold2 = np.zeros(s_tot, bool)
    sel_hold2[hold_idx[len(hold_idx) // 2:]] = True        # second half of the hold (transient decayed)
    sel_slide = (ts["phase"] == 3) | (ts["phase"] == 4)
    iz = int(np.argmax(F[:, 2])) if s_tot else 0
    depth = (ts["obj"][:, 2] - z_touch_setp) if z_touch_setp is not None else np.zeros(s_tot)
    rec = dict(seg=k, object=o, mass_scale=mass_scale, solver=SOLVER, depth_cap_m=DEPTH_CAP, cells=list(cells), verts=[vx, vy, vz], iterations=iterations,
               young_in_Pa=remap_f1(E_T, NU_T)[0], poisson_in=remap_f1(E_T, NU_T)[1],
               solref=SOLREF, solimp=SOLIMP, mu=MU, timestep_s=H, nsteps=s_tot, nan_step=nan_step,
               compile_s=compile_s, nv=int(m.nv), nflexvert=int(num0), nelem=int(m.flex_elemnum[0]),
               press_stop=stop_reason, warnings=warn0,
               wall_total_s=float(ts["wall"].sum()), wall_median_ms=float(np.median(ts["wall"]) * 1e3) if s_tot else None,
               Fz_max_N=float(F[:, 2].max()) if s_tot else 0.0,
               Ft_max_N=float(np.hypot(F[:, 0], F[:, 1]).max()) if s_tot else 0.0,
               Fz_hold_N=float(F[sel_press, 2].mean()) if sel_press.any() else None,
               depth_max_m=float(depth.max()) if s_tot else 0.0,
               uz_max_m=float(ts["uz_max"].max()) if s_tot else 0.0,
               pen_at_Fzmax_m=float(ts["pen_max"][iz]) if s_tot else 0.0,
               uz_at_Fzmax_m=float(ts["uz_max"][iz]) if s_tot else 0.0,
               ncon_max=int(ts["ncon"].max()) if s_tot else 0,
               niter_max=int(ts["niter"].max()) if s_tot else 0,
               frac_steps_iter_cap=float(np.mean(ts["niter"] >= iterations)) if s_tot else 0.0,
               taxsum_err_max_N=float(ts["taxsum_err"].max()) if s_tot else 0.0,
               # quasi-static check: servo force on the object vs contact force on the layer (Newton 3 + m a)
               act_vs_contact_err_hold_N=float(np.abs(ts["Fact"][sel_hold2] - F[sel_hold2]).max())
               if sel_hold2.any() else None,
               pen_fw_at_Fzmax_m=float(ts["pen_fw"][iz]) if s_tot else 0.0,
               depth_at_Fzmax_m=float(depth[iz]) if s_tot else 0.0,
               slide_Ft_over_Fz_max=float((np.hypot(F[sel_slide, 0], F[sel_slide, 1]) /
                                           np.maximum(F[sel_slide, 2], 1e-9)).max()) if sel_slide.any() else None,
               z_touch_setp_m=z_touch_setp, n_frames=len(frames["step"]),
               minJ_min=float(min(frames["minJ"])) if frames["minJ"] else None,
               ninv_max=int(max(frames["ninv"])) if frames["ninv"] else None)
    np.savez_compressed(os.path.join(outdir, f"seg{k}.npz"), **ts,
                        f_step=np.array(frames["step"]), f_qpos=np.array(frames["qpos"]),
                        f_ufront=np.array(frames["ufront"]), f_tax=np.array(frames["tax"]),
                        f_cpos=np.array(frames["cpos"], dtype=object), f_cforce=np.array(frames["cforce"], dtype=object),
                        f_minJ=np.array(frames["minJ"]), f_ninv=np.array(frames["ninv"]),
                        v0front=v0[front], depth=depth)
    with open(os.path.join(outdir, f"seg{k}.json"), "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(json.dumps({kk: rec[kk] for kk in ("seg", "nsteps", "nan_step", "press_stop", "Fz_max_N", "Ft_max_N",
                                              "depth_max_m", "uz_max_m", "pen_at_Fzmax_m", "pen_fw_at_Fzmax_m", "depth_at_Fzmax_m", "uz_at_Fzmax_m",
                                              "wall_median_ms", "ncon_max", "frac_steps_iter_cap",
                                              "taxsum_err_max_N", "act_vs_contact_err_hold_N", "minJ_min", "ninv_max", "warnings")},
                     default=float), flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seg", type=int)
    ap.add_argument("--cells", type=int, nargs=3, default=[40, 56, 3])
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--mass_scale", type=float, default=MASS_SCALE)
    ap.add_argument("--solver", default=SOLVER)
    ap.add_argument("--solref0", type=float, default=SOLREF[0])
    ap.add_argument("--cap_um", type=float, default=DEPTH_CAP * 1e6)
    a = ap.parse_args()
    DEPTH_CAP = a.cap_um * 1e-6
    SOLVER = a.solver
    SOLREF = (a.solref0, 1.0)
    run(a.seg, a.cells, a.iterations, a.out, a.steps, a.mass_scale)
