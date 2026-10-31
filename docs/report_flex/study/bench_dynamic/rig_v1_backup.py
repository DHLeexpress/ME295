"""bench-dynamic rig: the 51 x 71 x 1.5 mm PDMS pad as a MuJoCo flexcomp in a static body (back face pinned),
pressed by a rigid indenter under displacement control (mocap body), with real MuJoCo contacts.

Frame (= PDMS frame of the robot): origin at the centre of the FREE face, x across (51 mm), y along (71 mm),
+z into the layer. The layer occupies 0 <= z <= T; the back face z = T is pinned (bonded); the indenter
approaches from z < 0 and its displacement delta > 0 is +z (into the layer).

One call of run(cfg) = one simulation; it returns a JSON-able record (config + metrics, SI units, unit in each key)
and writes the per-step time series to runs/<id>.npz.

Usage:  python rig.py '<json cfg>'      (prints the record as one JSON line, also writes runs/<id>.json)
"""
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
A, B, T = 0.051, 0.071, 0.0015          # pad size (m): x, y, thickness
RHO = 965.0
NXT, NYT = 5, 7                         # taxels
DXT, DYT = A / NXT, B / NYT             # 10.2 mm, 10.142857 mm
MASS = RHO * A * B * T                  # 5.2414e-3 kg

# contact settings (applied to both the flex and the indenter geoms; the flex has priority 2)
CONTACT = {
    # the robot's setting
    "stiff":    dict(solref=(1e-4, 1.0), solimp=(0.99, 0.99, 0.001, 0.5, 2), refsafe=False, margin=0.0, radius=0.0),
    # MuJoCo defaults (solref, solimp, refsafe enabled)
    "default":  dict(solref=(0.02, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2), refsafe=True, margin=0.0, radius=0.0),
    # stiff + flex radius 0.1 mm (inflated elements); indenter start offset by the radius
    "radius":   dict(solref=(1e-4, 1.0), solimp=(0.99, 0.99, 0.001, 0.5, 2), refsafe=False, margin=0.0, radius=1e-4),
    # stiff + contact margin 0.1 mm (contacts created 0.1 mm before touching); start offset by the margin
    "margin":   dict(solref=(1e-4, 1.0), solimp=(0.99, 0.99, 0.001, 0.5, 2), refsafe=False, margin=1e-4, radius=0.0),
    # the stiffest the discrete integrator allows: impedance at mjMAXIMP, timeconst 10 us
    "stiffest": dict(solref=(1e-5, 1.0), solimp=(0.9999, 0.9999, 0.001, 0.5, 2), refsafe=False, margin=0.0,
                     radius=0.0),
}


def remap_f1(E, nu):
    """F1 workaround for dof='full': inputs (young, poisson) that make MuJoCo's doubled lambda equal the target."""
    G = E / (2 * (1 + nu))
    nu_in = nu / (2 * (1 - nu))
    return 2 * G * (1 + nu_in), nu_in


def taxel_index(x, y):
    i = np.clip(np.floor((np.asarray(x) + A / 2) / DXT).astype(int), 0, NXT - 1)
    j = np.clip(np.floor((np.asarray(y) + B / 2) / DYT).astype(int), 0, NYT - 1)
    return i, j


def default_cfg():
    return dict(
        variant="full_corr",        # full_corr | full_raw | trilinear | quadratic
        cells=[10, 14, 2],          # elements (full) or interpolation cells (trilinear/quadratic): x, y, z
        verts=None,                 # interpolated only: collision vertex grid (default: one vertex per node)
        E_Pa=1e6, nu=0.45,
        case="C1",                  # C1 C1f C2 C4 C5 C6 C6f
        contact="stiff",
        mu=2.0,                     # friction for the no-slip cases
        solver="CG", iterations=100, tolerance=1e-8,
        h_s=5e-4,
        mass_scale=1.0,
        split=1,                    # indenter split into split x split geoms (contact-cap workaround)
        delta_um=None,              # main displacement (case default if None)
        ramp_s=0.01, hold_s=0.25,   # each phase: linear ramp then hold
        max_step_s=1.0,             # abort (too slow) if the median of the first probe steps exceeds this
        memory="2G",
        driver="servo",             # servo: indenter on 3 slide joints + stiff position actuators; mocap: kinematic
        servo_kp=1e10, ind_mass_kg=0.1,
        tag="",
    )


CASE_DEFAULTS = {
    # delta: normal displacement (m); for C2 the tangential displacement is in 'shear'
    "C1": dict(delta=2e-6, friction=True),
    "C1f": dict(delta=2e-6, friction=False),
    "C2": dict(delta=3e-6, shear=1e-6, friction=True),
    "C4": dict(delta=10e-6, friction=False, a=5e-3),
    "C5": dict(delta=30e-6, friction=False, R=10e-3, levels=6),
    "C6": dict(delta=5e-6, friction=True),
    "C6f": dict(delta=5e-6, friction=False),
}


def indenter_geoms(case, split, cfg, cp):
    """Geoms of the mocap indenter; body origin = centre of the indenter's contact face (top, +z)."""
    cd = CASE_DEFAULTS[case]
    fr = cd["friction"]
    condim = 3 if fr else 1
    mu = cfg["mu"] if fr else 0.0
    con = (f'contype="2" conaffinity="2" condim="{condim}" friction="{max(mu, 1e-6) if fr else 1} 0.005 0.0001" '
           f'solref="{cp["solref"][0]:g} {cp["solref"][1]:g}" solimp="{" ".join(f"{v:g}" for v in cp["solimp"])}" '
           f'margin="{cp["margin"]:g}"')
    hz = 0.005
    out = []
    if case in ("C1", "C1f", "C2", "C6", "C6f"):
        hx, hy = (0.035, 0.045) if case in ("C1", "C1f", "C2") else (DXT / 2, DYT / 2)
        n = split
        for a in range(n):
            for b in range(n):
                cx = -hx + (2 * a + 1) * hx / n
                cy = -hy + (2 * b + 1) * hy / n
                out.append(f'<geom name="ind_{a}_{b}" type="box" size="{hx/n:.9g} {hy/n:.9g} {hz}" '
                           f'pos="{cx:.9g} {cy:.9g} {-hz}" {con}/>')
    elif case == "C4":
        a = cfg.get("punch_a_m") or cd["a"]
        if split != 1:
            raise ValueError("split is implemented for box indenters only")
        out.append(f'<geom name="ind" type="cylinder" size="{a:.9g} {hz}" pos="0 0 {-hz}" {con}/>')
    elif case == "C5":
        if split != 1:
            raise ValueError("split is implemented for box indenters only")
        R = cfg.get("sphere_R_m") or cd["R"]
        out.append(f'<geom name="ind" type="sphere" size="{R:.9g}" pos="0 0 {-R:.9g}" {con}/>')
    else:
        raise ValueError(case)
    return out, condim


def build_xml(cfg):
    cp = CONTACT[cfg["contact"]]
    case = cfg["case"]
    cx, cy, cz = cfg["cells"]
    var = cfg["variant"]
    E, nu = cfg["E_Pa"], cfg["nu"]
    if var == "full_corr":
        young, poisson = remap_f1(E, nu)
    else:
        young, poisson = E, nu
    fr = CASE_DEFAULTS[case]["friction"]
    condim = 3 if fr else 1
    mu = cfg["mu"] if fr else 1.0
    fcon = (f'contype="2" conaffinity="2" condim="{condim}" priority="2" friction="{mu:g} 0.005 0.0001" '
            f'solref="{cp["solref"][0]:g} {cp["solref"][1]:g}" solimp="{" ".join(f"{v:g}" for v in cp["solimp"])}" '
            f'margin="{cp["margin"]:g}" selfcollide="none"')
    mass = MASS * cfg["mass_scale"]
    if var in ("full_corr", "full_raw"):
        vx, vy, vz = cx + 1, cy + 1, cz + 1
        pin = f'<pin gridrange="0 0 {vz-1} {vx-1} {vy-1} {vz-1}"/>'
        extra = ""
    else:
        k = 1 if var == "trilinear" else 2
        vx, vy, vz = cfg["verts"] or (k * cx + 1, k * cy + 1, k * cz + 1)
        nxn, nyn, nzn = k * cx + 1, k * cy + 1, k * cz + 1
        # flexcomp grid point index (i*ny + j)*nz + k; the grid's +z face (k = nz-1) is the back face here
        ids = [(i * nyn + j) * nzn + (nzn - 1) for i in range(nxn) for j in range(nyn)]
        pin = f'<pin id="{" ".join(map(str, ids))}"/>'
        extra = f'dof="{var}" cellcount="{cx} {cy} {cz}"'
    sp = f"{A/(vx-1):.10g} {B/(vy-1):.10g} {T/(vz-1):.10g}"
    geoms, _ = indenter_geoms(case, cfg["split"], cfg, cp)
    opt_flag = "" if cp["refsafe"] else '<flag refsafe="disable"/>'
    off = cp["radius"] + cp["margin"]          # start the indenter at the effective surface
    kp = cfg["servo_kp"]
    ind_dofs = (f'<inertial pos="0 0 -0.005" mass="{cfg["ind_mass_kg"]}" diaginertia="1e-4 1e-4 1e-4"/>'
                '<joint name="ix" type="slide" axis="1 0 0"/><joint name="iy" type="slide" axis="0 1 0"/>'
                '<joint name="iz" type="slide" axis="0 0 1"/>')
    ind_act = ('<actuator>' + ''.join(f'<position name="a{c}" joint="i{c}" kp="{kp:g}" dampratio="1"/>' for c in "xyz")
               + '</actuator>')
    xml = f"""<mujoco model="bench_dynamic">
<option timestep="{cfg['h_s']}" integrator="discrete" solver="{cfg['solver']}" cone="elliptic" impratio="100"
        gravity="0 0 0" iterations="{cfg['iterations']}" tolerance="{cfg['tolerance']}">{opt_flag}</option>
<size memory="{cfg['memory']}"/>
<worldbody>
  <body name="pad" pos="0 0 {T/2:.10g}">
    <flexcomp name="pdms" type="grid" dim="3" {extra} count="{vx} {vy} {vz}" spacing="{sp}"
              radius="{cp['radius']:g}" mass="{mass:.8g}">
      <elasticity young="{young:.10g}" poisson="{poisson:.10g}" damping="0"/>
      <contact {fcon}/>
      {pin}
    </flexcomp>
  </body>
  <body name="ind" {'mocap="true"' if cfg['driver'] == 'mocap' else ''} pos="0 0 {-off:.10g}">
    {'' if cfg['driver'] == 'mocap' else ind_dofs}
    {chr(10).join('    ' + g for g in geoms)}
  </body>
</worldbody>
{'' if cfg['driver'] == 'mocap' else ind_act}
</mujoco>"""
    return xml, dict(young_in_Pa=young, poisson_in=poisson, verts=[vx, vy, vz])


def cfg_id(cfg):
    s = json.dumps({k: cfg[k] for k in sorted(cfg) if k not in ("tag",)}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def phases_for(cfg):
    """List of (target displacement (3,), ramp_steps, hold_steps, label)."""
    case = cfg["case"]
    cd = CASE_DEFAULTS[case]
    h = cfg["h_s"]
    nr = max(1, int(round(cfg["ramp_s"] / h)))
    nh = max(1, int(round(cfg["hold_s"] / h)))
    delta = cfg["delta_um"] * 1e-6 if cfg["delta_um"] is not None else cd["delta"]
    if case == "C2":
        return [(np.array([0, 0, delta]), nr, nh // 2, "press"),
                (np.array([0, cd["shear"], delta]), nr, nh, "shear")]
    if case == "C5":
        L = cd["levels"]
        ph = []
        for k in range(1, L + 1):
            ph.append((np.array([0, 0, delta * k / L]), nr, nh if k == L else max(nh // 4, 120), f"lvl{k}"))
        return ph
    return [(np.array([0, 0, delta]), nr, nh, "press")]


def contact_forces(m, d):
    """Per-contact force on the LAYER (world frame), positions, dist. Uses efc_force (elliptic/frictionless)."""
    n = d.ncon
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)
    con = d.contact
    adr = np.asarray(con.efc_address[:n])
    dim = np.asarray(con.dim[:n])
    ok = adr >= 0
    f = np.zeros((n, 3))
    for dm in np.unique(dim[ok]):
        sel = ok & (dim == dm)
        k = min(dm, 3)
        f[sel, :k] = d.efc_force[adr[sel][:, None] + np.arange(k)]
    frames = np.asarray(con.frame[:n]).reshape(-1, 3, 3)
    fw = np.einsum("kij,ki->kj", frames, f)
    # the solver's force acts on the second object: flip if the flex is the first object
    flex = np.asarray(con.flex[:n])
    sign = np.where(flex[:, 1] >= 0, 1.0, -1.0)
    fw *= sign[:, None]
    return fw, np.asarray(con.pos[:n]).copy(), np.asarray(con.dist[:n]).copy(), ok


def settle_time(t, y, t0, tol=0.01, ref=None):
    """Time after t0 at which y enters and stays in a band |y - ref| <= tol |ref| (ref = mean of last 200)."""
    sel = t >= t0 - 1e-12
    tt, yy = t[sel], y[sel]
    if ref is None:
        ref = yy[-200:].mean()
    if abs(ref) < 1e-12:
        return None
    out = np.abs(yy - ref) > tol * abs(ref)
    if not out.any():
        return 0.0
    last = np.where(out)[0][-1]
    if last >= len(tt) - 1:
        return None            # never settled within the window
    return float(tt[last + 1] - t0)


def run(cfg_in):
    cfg = default_cfg()
    cfg.update(cfg_in)
    rid = cfg_id(cfg)
    rec = dict(id=rid, cfg=cfg, status="ok")
    xml, info = build_xml(cfg)
    rec.update(info)
    t0 = time.perf_counter()
    try:
        m = mujoco.MjModel.from_xml_string(xml)
    except Exception as e:  # noqa
        rec.update(status="compile_error", error=str(e)[-2000:])
        return rec, None
    rec["compile_s"] = time.perf_counter() - t0
    d = mujoco.MjData(m)
    rec["nv"] = int(m.nv)
    rec["nflexvert"] = int(m.flex_vertnum[0])
    rec["nelem"] = int(m.flex_elemnum[0])
    lay = m.flex_elemlayer[m.flex_elemadr[0]:m.flex_elemadr[0] + m.flex_elemnum[0]]
    rec["nelem_active"] = int((lay < m.flex_activelayers[0]).sum())
    rec["vertex_mass_kg"] = float(np.median(m.body_mass[m.body_dofnum > 0])) if (m.body_dofnum > 0).any() else None
    rec["narena_bytes"] = int(d.narena)
    rec["nbuffer_bytes"] = int(d.nbuffer)
    mujoco.mj_forward(m, d)
    cp = CONTACT[cfg["contact"]]
    off = cp["radius"] + cp["margin"]          # the indenter starts at the effective surface (body pos in XML)
    base = np.array([0.0, 0.0, -off])
    servo = cfg["driver"] != "mocap"
    if servo:
        qadr = np.array([m.jnt_qposadr[m.joint(f"i{c}").id] for c in "xyz"])
        aid = np.array([m.actuator(f"a{c}").id for c in "xyz"])
    else:
        mid = m.body("ind").mocapid[0]
        d.mocap_pos[mid] = base
    mujoco.mj_forward(m, d)
    adr0, num0 = m.flex_vertadr[0], m.flex_vertnum[0]
    v0 = d.flexvert_xpos[adr0:adr0 + num0].copy()
    front = np.where(np.abs(v0[:, 2]) < 1e-9)[0]
    xy = v0[front, :2]
    # taxel weights for the free-face vertices (boundary vertices shared, as in tactile.PDMSSkin)
    W = np.zeros((NYT * NXT, front.size))
    tol = 1e-9
    for k, (x, y) in enumerate(xy):
        ii = [i for i in range(NXT) if -A / 2 + i * DXT - tol <= x <= -A / 2 + (i + 1) * DXT + tol]
        jj = [j for j in range(NYT) if -B / 2 + j * DYT - tol <= y <= -B / 2 + (j + 1) * DYT + tol]
        for i in ii:
            for j in jj:
                W[j * NXT + i, k] = 1.0
    W /= np.maximum(W.sum(1, keepdims=True), 1)
    case = cfg["case"]
    cd = CASE_DEFAULTS[case]
    # footprint of the indenter on the free face (for the 'under the indenter' displacement)
    if case in ("C1", "C1f", "C2"):
        foot = np.ones(front.size, bool)
    elif case in ("C6", "C6f"):
        foot = (np.abs(xy[:, 0]) <= DXT / 2 + 1e-9) & (np.abs(xy[:, 1]) <= DYT / 2 + 1e-9)
    elif case == "C4":
        a = cfg.get("punch_a_m") or cd["a"]
        foot = np.hypot(xy[:, 0], xy[:, 1]) <= a + 1e-9
    else:
        foot = np.hypot(xy[:, 0], xy[:, 1]) <= 1e-9 + 0.5 * min(A / (rec["verts"][0] - 1), B / (rec["verts"][1] - 1))
    rr = np.hypot(xy[:, 0], xy[:, 1])
    centre = int(np.argmin(rr))
    if not foot.any():                      # no free-face vertex inside the footprint: use the nearest one(s)
        foot = rr <= rr.min() + 1e-9
        rec["foot_fallback_nearest"] = True
    rec["foot_nverts"] = int(foot.sum())
    rec["centre_vertex_offset_m"] = float(rr[centre])
    # element -> vertex table and pinned flags (for the contact-weight diagnostic)
    fdim = int(m.flex_dim[0])
    elem = m.flex_elem[m.flex_elemdataadr[0]:m.flex_elemdataadr[0] + m.flex_elemnum[0] * (fdim + 1)].reshape(-1, fdim + 1)
    pinned_v = m.flex_vertbodyid[adr0:adr0 + num0] == m.body("pad").id
    interp = bool(m.flex_interp[0])
    ind_body = m.body("ind").id
    flexdofs = np.array([i for i in range(m.nv) if m.dof_bodyid[i] != ind_body])
    flexaxes = np.array([m.jnt_axis[m.dof_jntid[i]] for i in flexdofs]) if flexdofs.size else np.zeros((0, 3))

    phases = phases_for(cfg)
    nsteps = sum(p[1] + p[2] for p in phases)
    ts = dict(t=np.zeros(nsteps), F=np.zeros((nsteps, 3)), ncon=np.zeros(nsteps, int), nefc=np.zeros(nsteps, int),
              niter=np.zeros(nsteps, int), wall=np.zeros(nsteps), uz_foot=np.zeros(nsteps), uz_centre=np.zeros(nsteps),
              ind=np.zeros((nsteps, 3)), ind_target=np.zeros((nsteps, 3)), pen_max=np.zeros(nsteps),
              uy_foot=np.zeros(nsteps), nnz=np.zeros(nsteps, int))
    phase_of = np.zeros(nsteps, int)
    k = 0
    prev = np.zeros(3)
    rec["phases"] = []
    diverged = None
    t_start = time.perf_counter()
    too_slow = False
    for ip, (target, nr, nh, label) in enumerate(phases):
        for s in range(nr + nh):
            frac = min(1.0, (s + 1) / nr)
            disp = prev + frac * (target - prev)
            if servo:
                d.ctrl[aid] = disp
            else:
                d.mocap_pos[mid] = base + disp
            tw = time.perf_counter()
            mujoco.mj_step(m, d)
            dt = time.perf_counter() - tw
            ts["wall"][k] = dt
            ts["t"][k] = d.time
            ts["ind_target"][k] = disp
            ts["ind"][k] = d.qpos[qadr] if servo else disp
            phase_of[k] = ip
            bad = (not np.isfinite(d.qpos).all()) or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
            if bad:
                diverged = dict(step=k, phase=label, time_s=float(k * cfg["h_s"]))
                break
            fw, pos, dist, ok = contact_forces(m, d)
            ts["F"][k] = fw.sum(0)
            ts["ncon"][k] = d.ncon
            ts["nefc"][k] = d.nefc
            ts["niter"][k] = int(np.sum(d.solver_niter[:max(1, d.nisland)]))
            ts["nnz"][k] = int(np.sum(d.solver_nnz[:max(1, d.nisland)]))
            u = d.flexvert_xpos[adr0:adr0 + num0][front] - v0[front]
            ts["uz_foot"][k] = u[foot, 2].mean()
            ts["uy_foot"][k] = u[foot, 1].mean()
            ts["uz_centre"][k] = u[centre, 2]
            ts["pen_max"][k] = -dist.min() if dist.size else 0.0
            k += 1
            # cost probe: abort if the first steps are too slow
            if k == 12:
                med = float(np.median(ts["wall"][2:12]))
                if med > cfg["max_step_s"]:
                    too_slow = True
                    rec.update(status="too_slow", probe_step_s=med)
                    break
        if diverged or too_slow:
            break
        # end-of-phase snapshot (fields at the final step; force stats over the last 200 steps)
        fw, pos, dist, ok = contact_forces(m, d)
        i, j = taxel_index(pos[:, 0], pos[:, 1]) if len(pos) else (np.zeros(0, int), np.zeros(0, int))
        tax = np.zeros((NYT, NXT, 3))
        np.add.at(tax, (j, i), fw)
        ntax = np.zeros((NYT, NXT), int)
        np.add.at(ntax, (j, i), 1)
        u = d.flexvert_xpos[adr0:adr0 + num0][front] - v0[front]
        utax = (W @ u).reshape(NYT, NXT, 3)
        sl = slice(max(0, k - min(200, nh)), k)
        Fm = ts["F"][sl].mean(0)
        Fs = ts["F"][sl].std(0)
        loaded = fw[:, 2] > 1e-6 * max(1e-12, fw[:, 2].sum())
        r = np.hypot(pos[:, 0], pos[:, 1])
        pen = -dist
        wpen = float((pen[loaded] * fw[loaded, 2]).sum() / max(1e-30, fw[loaded, 2].sum())) if loaded.any() else 0.0
        # share of each contact's force that MuJoCo's inverse-distance weights put on pinned (back-face)
        # vertices, i.e. straight into the support (full flexes only; interpolated flexes map to nodes)
        pin_share = None
        if not interp and d.ncon:
            el = np.asarray(d.contact.elem[:d.ncon])[:, 1]
            vtx = elem[el]                                            # (ncon, 4)
            dv = np.linalg.norm(d.flexvert_xpos[adr0 + vtx] - pos[:, None, :], axis=2)
            w = 1.0 / np.maximum(dv, 1e-15)
            w /= w.sum(1, keepdims=True)
            sh = (w * pinned_v[vtx]).sum(1)
            fn = np.maximum(fw[:, 2], 0)
            pin_share = float((sh * fn).sum() / max(1e-30, fn.sum()))
        # force the contacts deliver to the free (moving) vertices/nodes, from MuJoCo's qfrc_constraint
        F_free = (d.qfrc_constraint[flexdofs][:, None] * flexaxes).sum(0) if flexdofs.size else np.zeros(3)
        ph = dict(label=label, target_m=target.tolist(), end_step=k, t_end_s=float(d.time),
                  pinned_weight_share=pin_share, F_to_free_dofs_N=F_free.tolist(),
                  F_final_step_N=fw.sum(0).tolist(),
                  ind_disp_m=ts["ind"][sl].mean(0).tolist(),
                  F_mean_N=Fm.tolist(), F_std_N=Fs.tolist(), F_final_N=ts["F"][k - 1].tolist(),
                  ncon=int(d.ncon), nefc=int(d.nefc), ncon_loaded=int(loaded.sum()),
                  taxel_force_N=tax.tolist(), taxel_ncon=ntax.tolist(), taxel_disp_m=utax.tolist(),
                  uz_foot_mean_m=float(ts["uz_foot"][sl].mean()), uz_centre_m=float(ts["uz_centre"][sl].mean()),
                  uy_foot_mean_m=float(ts["uy_foot"][sl].mean()),
                  uz_free_face_max_m=float(u[:, 2].max()), uz_free_face_min_m=float(u[:, 2].min()),
                  pen_max_m=float(pen.max()) if pen.size else 0.0,
                  pen_forceweighted_m=wpen,
                  pen_eff_max_m=float((pen + cp["margin"]).max()) if pen.size else 0.0,
                  contact_radius_max_m=float(r[loaded].max()) if loaded.any() else 0.0,
                  contact_radius_rms_fw_m=float(np.sqrt((r[loaded] ** 2 * fw[loaded, 2]).sum()
                                                       / max(1e-30, fw[loaded, 2].sum()))) if loaded.any() else 0.0,
                  niter_median=float(np.median(ts["niter"][sl])), niter_max=int(ts["niter"][sl].max()),
                  solver_nnz_median=float(np.median(ts["nnz"][sl])),
                  F_min_N=ts["F"][sl].min(0).tolist(), F_max_N=ts["F"][sl].max(0).tolist(),
                  ncon_min=int(ts["ncon"][sl].min()), ncon_max=int(ts["ncon"][sl].max()),
                  wall_median_s=float(np.median(ts["wall"][sl])))
        # settling of the main component after the ramp
        comp = 1 if label == "shear" else 2
        t0p = ts["t"][k - nh - 1] if k - nh - 1 >= 0 else 0.0
        seg = slice(k - nh - nr, k)
        y = ts["F"][seg, comp]
        tt = ts["t"][seg]
        tr = ts["t"][k - nh - 1]
        ph["settle_1pct_s"] = settle_time(tt, y, tr, 0.01)
        ph["settle_5pct_s"] = settle_time(tt, y, tr, 0.05)
        ph["chatter_rel"] = float(Fs[comp] / max(1e-12, abs(Fm[comp])))
        # drift: relative change of the mean between the two halves of the last 200 steps
        half = ts["F"][sl, comp]
        if len(half) >= 20:
            ph["drift_rel"] = float((half[len(half) // 2:].mean() - half[:len(half) // 2].mean())
                                    / max(1e-12, abs(Fm[comp])))
        rec["phases"].append(ph)
        prev = target
    rec["wall_total_s"] = time.perf_counter() - t_start
    rec["nsteps_done"] = int(k)
    if diverged:
        rec.update(status="diverged", diverged=diverged)
    nw = int(mujoco.mjtWarning.mjNWARNING)
    rec["warnings"] = {mujoco.mjtWarning(i).name: int(d.warning[i].number) for i in range(nw) if d.warning[i].number}
    rec["maxuse_arena_bytes"] = int(d.maxuse_arena)
    rec["maxuse_con"] = int(d.maxuse_con)
    rec["maxuse_efc"] = int(d.maxuse_efc)
    rec["maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    # cost over the steps in contact (all steps after the first ramp started, contact present)
    inc = ts["ncon"][:k] > 0
    if inc.sum() >= 1:
        rec["us_per_step_median"] = float(1e6 * np.median(ts["wall"][:k][inc]))
        rec["us_per_step_p90"] = float(1e6 * np.percentile(ts["wall"][:k][inc], 90))
        rec["n_steps_in_contact"] = int(inc.sum())
        rec["niter_median_all"] = float(np.median(ts["niter"][:k][inc]))
        rec["frac_steps_at_iter_cap"] = float(np.mean(ts["niter"][:k][inc] >= cfg["iterations"]))
    for key in ts:
        ts[key] = ts[key][:k]
    return rec, ts


def summarize(rec):
    if rec["status"] not in ("ok",) or not rec.get("phases"):
        return f"{rec['id']} {rec['status']}"
    p = rec["phases"][-1]
    return (f"{rec['id']} {rec['cfg']['variant']} {rec['cfg']['cells']} {rec['cfg']['case']} {rec['cfg']['contact']} "
            f"F={np.round(p['F_mean_N'], 4).tolist()} uz={1e6*p['uz_foot_mean_m']:.4f}um pen={1e6*p['pen_max_m']:.3f}um "
            f"ncon={p['ncon']} it={p['niter_median']:.0f} chat={p['chatter_rel']:.2e} set1={p['settle_1pct_s']} "
            f"us={rec.get('us_per_step_median', 0):.0f}")


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    outdir = cfg.pop("outdir", os.path.join(HERE, "runs"))
    os.makedirs(outdir, exist_ok=True)
    rec, ts = run(cfg)
    if ts is not None:
        np.savez_compressed(os.path.join(outdir, rec["id"] + ".npz"), **ts)
    with open(os.path.join(outdir, rec["id"] + ".json"), "w") as f:
        json.dump(rec, f)
    print("SUMMARY " + summarize(rec), flush=True)
