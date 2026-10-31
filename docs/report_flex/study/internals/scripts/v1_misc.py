"""Misc. confirmations for NOTES.md items 1, 4, 5.
(a) vertex bodies/joints inherit the parent body's childclass (joint armature/damping/stiffness) -> user_objects.cc:2460,2517
(b) pinned-vertex mass is dropped: total flex body mass = mass * nfree/npnt (user_flexcomp.cc:476)
(c) island-metric omission (engine_metric.c:2485 flexMatrixFree + engine_derivative.c:1905): ONE hinge carrying a pinned
    3D flex + one active constraint row (joint limit) diverges with islands on, not with mjDSBL_ISLAND.
    (finding of the flexstudy/robot agent, reproduced independently here)
(d) contact-stiffness ceiling under the discrete integrator (engine_core_constraint.c:2119-2121, R >= Rmin):
    per-contact static stiffness k = K*imp/(f*R) <= mjMAXIMP/(1-mjMAXIMP) * m_v / h^2 ; read from efc arrays.
Output: out/v1_misc.{json,txt}"""
import sys
import numpy as np
import mujoco
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import load, dump

A, B, T = 0.051, 0.071, 0.0015
res = {}

def flex(count, extra_el="", mass=0.0052414):
    nx, ny, nz = count
    return f'''<flexcomp name="f" type="grid" dim="3" count="{nx} {ny} {nz}" spacing="{A/(nx-1)} {B/(ny-1)} {T/(nz-1)}"
       radius="0" mass="{mass}"><elasticity young="1e6" poisson="0.45" damping="0"/>
       <contact contype="2" conaffinity="2" priority="2" friction="1" selfcollide="none" solref="0.0001 1" solimp="0.99 0.99 0.001"/>
       <pin gridrange="0 0 {nz-1} {nx-1} {ny-1} {nz-1}"/>{extra_el}</flexcomp>'''

# (a) + (b)
xml = f'''<mujoco><option integrator="discrete" solver="CG"/>
<default><default class="fing"><joint armature="0.01" stiffness="5" frictionloss="0.001"/></default></default>
<worldbody><body name="pd" childclass="fing" pos="0 0 0.1">{flex((11, 15, 3))}</body></worldbody></mujoco>'''
m, w = load(xml)
vb = [b for b in range(m.nbody) if m.body_dofnum[b] == 3]
dofs = np.concatenate([np.arange(m.body_dofadr[b], m.body_dofadr[b] + 3) for b in vb])
res["a_childclass"] = dict(nvertbody=len(vb), armature=sorted(set(np.round(m.dof_armature[dofs], 6))),
                           damping=sorted(set(np.round(m.dof_damping[dofs], 6))),
                           jnt_stiffness=sorted(set(np.round(m.jnt_stiffness, 6))), frictionloss=sorted(set(np.round(m.dof_frictionloss[dofs], 6))))
res["b_mass"] = dict(mass_attr=0.0052414, npnt=11 * 15 * 3, nfree=len(vb),
                     body_mass_each=float(m.body_mass[vb[0]]), total_flex_body_mass=float(m.body_mass[vb].sum()),
                     ratio=float(m.body_mass[vb].sum() / 0.0052414), warnings=w)

# (c) minimal island test
def hinge_model(count, row):
    lim = 'limited="true" range="-0.0001 0.0001"' if row == "limit" else ""
    eq = '<equality><joint joint1="hj" polycoef="0 0 0 0 0"/></equality>' if row == "eq" else ""
    return f'''<mujoco><option integrator="discrete" solver="CG" timestep="0.0005" gravity="0 0 -9.81"/>
<worldbody><body name="link" pos="0 0 0.5"><joint name="hj" type="hinge" axis="1 0 0" {lim}/>
<geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.01" mass="0.2" contype="0" conaffinity="0"/>
<body name="pd" pos="0 0 -0.06">{flex(count, mass=0.0052414)}</body></body></worldbody>{eq}</mujoco>'''

def run_stab(count, row, island, nstep=400):
    m, _ = load(hinge_model(count, row))
    if not island:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    first_bad = None
    vmax = 0.0
    for k in range(nstep):
        mujoco.mj_step(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or not np.isfinite(d.qvel).all():
            first_bad = k + 1
            break
        vmax = max(vmax, float(np.abs(d.qvel).max()))
    return dict(count=count, row=row, island=island, nv=int(m.nv), first_bad_step=first_bad,
                qvel_max=vmax, nisland_last=int(d.nisland), nefc_last=int(d.nefc))

res["c_island"] = [run_stab(c, r, isl) for c in ((3, 3, 2), (11, 15, 3)) for r in ("none", "limit", "eq")
                   for isl in (True, False)]

# (d) contact ceiling on a plane (vertex contacts)
def plane_contact(tc, delta=5e-6, h=0.0005, nstep=1500):
    xml = f'''<mujoco><option integrator="discrete" solver="CG" timestep="{h}" gravity="0 0 0" iterations="200"><flag refsafe="disable"/></option>
<worldbody><body name="pd" pos="0 0 0.1">{flex((11, 15, 3)).replace('solref="0.0001 1"', f'solref="{tc} 1"')}</body>
<body mocap="true" pos="0 0 {0.1 - T/2 + delta}"><geom type="plane" size="0.1 0.1 0.01" contype="2" conaffinity="2"/></body>
</worldbody></mujoco>'''
    m, _ = load(xml)
    d = mujoco.MjData(m)
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    con = [i for i in range(d.ncon)]
    pen = np.array([-d.contact[i].dist for i in con])
    fn = []
    for i in con:
        f6 = np.zeros(6); mujoco.mj_contactForce(m, d, i, f6); fn.append(f6[0])
    fn = np.array(fn)
    ok = pen > 1e-12
    k_meas = float(np.median(fn[ok] / pen[ok])) if ok.any() else float("nan")
    i0 = d.contact[0].efc_address
    K, Bc, I, _ = d.efc_KBIP.reshape(-1, 4)[i0]
    f = 1 + h * Bc + h * h * K * I
    R = d.efc_R[i0]
    mv = float(m.body_mass[m.flex_vertbodyid[m.flex_vertadr[0] + 0]] if False else m.body_mass[[b for b in range(m.nbody) if m.body_dofnum[b] == 3][0]])
    ceiling = 0.9999 / (1 - 0.9999) * mv / h ** 2
    return dict(timeconst=tc, ncon=int(d.ncon), k_contact_measured=k_meas, k_pred_K_imp_over_fR=float(K * I / (f * R)),
                k_unfloored=float(K * I * I / ((1 - I) / mv)), ceiling_maximp=ceiling, m_vertex=mv,
                total_face_contact_stiffness_N_per_um=float(k_meas * d.ncon * 1e-6))

res["d_contact_ceiling"] = [plane_contact(tc) for tc in (1e-3, 1e-4, 3e-5, 1e-5, 1e-6)]
p = dump("v1_misc.json", res)
with open(p.replace(".json", ".txt"), "w") as fo:
    for k, v in res.items():
        if isinstance(v, list):
            for x in v:
                fo.write(f"{k} {x}\n")
        else:
            fo.write(f"{k} {v}\n")
print(open(p.replace(".json", ".txt")).read())
