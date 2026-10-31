"""P0 extended: which elastic material does each MuJoCo 3.17 flex type actually simulate?

For every configuration we impose displacements on the dof bodies, read the elastic nodal forces and
identify the material:
  (a) small homogeneous strains (1e-6): 6x6 effective stiffness C from the energy u_a' K u_b / V,
      -> lambda_eff, mu_eff and an anisotropy residual (exact patch test: C = iso_C(lam, mu));
  (b) for non-interpolated flexes, a least-squares fit of the nodal forces to the forces of a reference
      element implementation (written here, independent of MuJoCo) with unknown (mu, lam):
      small random inhomogeneous fields and FINITE homogeneous strain (1 %), residual reported.
Outputs: out/p0_patch_ext.json and a printed table.
"""
import sys
import numpy as np
import mujoco

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import (Rig, load, compile_spec, lame, remap_full, iso_C, voigt_strain, tet_force_basis,
                    tri_force_basis, fit_two, dump)

E = 1e6
rng = np.random.default_rng(0)


def xml3(count=(5, 5, 5), L=(0.01, 0.01, 0.01), young=E, poisson=0.45, dof="full", cells=None,
         pose="", joint="", equality="", integrator="discrete", extra_opt=""):
    sp = [L[i] / (count[i] - 1) for i in range(3)]
    cellattr = f'cellcount="{cells[0]} {cells[1]} {cells[2]}"' if cells else ""
    eq = f'<edge equality="{equality}"/>' if equality else ""
    if joint:  # a moving parent needs its own mass
        joint = joint + '<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'
    return f"""<mujoco><option gravity="0 0 0" integrator="{integrator}" solver="CG" timestep="1e-4" {extra_opt}/>
<worldbody><body name="b" {pose}>{joint}<flexcomp name="f" type="grid" dim="3" count="{count[0]} {count[1]} {count[2]}"
 spacing="{sp[0]} {sp[1]} {sp[2]}" dof="{dof}" {cellattr} mass="0.001" radius="0">
 <elasticity young="{young}" poisson="{poisson}" damping="0"/>{eq}<contact contype="0" conaffinity="0"/></flexcomp>
</body></worldbody></mujoco>"""


def xml2(count=(6, 6), L=(0.01, 0.01), young=E, poisson=0.45, thick=1e-3, elastic2d="stretch"):
    sp = [L[i] / (count[i] - 1) for i in range(2)]
    return f"""<mujoco><option gravity="0 0 0" integrator="discrete" solver="CG" timestep="1e-4"/>
<worldbody><body name="b"><flexcomp name="f" type="grid" dim="2" count="{count[0]} {count[1]} 1"
 spacing="{sp[0]} {sp[1]} 0.001" mass="0.001" radius="1e-5">
 <elasticity young="{young}" poisson="{poisson}" thickness="{thick}" elastic2d="{elastic2d}" damping="0"/>
 <contact contype="0" conaffinity="0"/></flexcomp></body></worldbody></mujoco>"""


def parent_R(rig):
    """Rotation of the flex's parent body 'b' (world <- parent)."""
    b = mujoco.mj_name2id(rig.m, mujoco.mjtObj.mjOBJ_BODY, "b")
    return rig.d.xmat[b].reshape(3, 3).copy(), rig.d.xpos[b].copy()


def force_fn_spring(rig):
    return rig.forces(rig.d.qfrc_spring)


def force_fn_strain_static(rig):
    """Static elastic force of strain-equality rows: f_row = -K*I*r/R (a = 0, v = 0)."""
    m, d = rig.m, rig.d
    nv = m.nv
    q = np.zeros(nv)
    J = d.efc_J.reshape(d.nefc, nv) if not mujoco.mj_isSparse(m) else None
    assert J is not None, "use dense jacobian"
    for r in range(d.nefc):
        if d.efc_type[r] != mujoco.mjtConstraint.mjCNSTR_EQUALITY:
            continue
        K, B, I, P = d.efc_KBIP.reshape(-1, 4)[r]
        fr = -K * I * d.efc_pos[r] / d.efc_R[r]
        q += J[r] * fr
    return rig.forces(q)


def C_matrix(rig, force_fn, R=np.eye(3), c=None, eps=1e-6, V=None):
    """Effective 6x6 stiffness in the parent frame from 21 energy products."""
    X = rig.X0
    c = X.mean(0) if c is None else c
    Xl = (X - c) @ R  # parent-frame coordinates (R maps parent->world, so world->parent is R^T)
    U = []
    Fs = []
    for k in range(6):
        e = voigt_strain(k, eps)
        Uw = (Xl @ e.T) @ R.T
        rig.set_disp(Uw)
        U.append(Uw)
        Fs.append(force_fn(rig))
    C = np.zeros((6, 6))
    for a in range(6):
        for b in range(6):
            C[a, b] = -np.sum(Fs[b] * U[a]) / (V * eps * eps)
    C = 0.5 * (C + C.T)
    # isotropic fit
    lam = np.mean([C[0, 1], C[0, 2], C[1, 2]])
    mu = np.mean([C[3, 3], C[4, 4], C[5, 5]])
    Ciso = iso_C(lam, mu)
    aniso = np.linalg.norm(C - Ciso) / np.linalg.norm(C)
    return C, lam, mu, aniso


def ref_fit(rig, tets, model, R=np.eye(3), finite=0.01):
    """Fit (mu, lam) of a reference tet implementation to MuJoCo's nodal forces."""
    m, d = rig.m, rig.d
    # vertex <-> body map (dof="full", no pins): flex vertex v belongs to body flex_vertbodyid[v]
    vb = m.flex_vertbodyid[m.flex_vertadr[0]:m.flex_vertadr[0] + m.flex_vertnum[0]]
    bidx = {b: i for i, b in enumerate(rig.bodies)}
    order = np.array([bidx[b] for b in vb])
    X = rig.vert0[:len(vb)].copy()
    c = X.mean(0)
    res = {}
    cases = {
        "random_small": None,
        "uniaxial_1pct": np.diag([1, 1, 1 - finite]),
        "shear_1pct": np.eye(3) + finite * np.outer([1, 0, 0], [0, 0, 1]),
        "general_1pct": np.eye(3) + finite * rng.uniform(-1, 1, (3, 3)),
    }
    for name, Fp in cases.items():
        if Fp is None:
            Uv = 1e-7 * rng.standard_normal(X.shape)
        else:
            Fw = R @ Fp @ R.T
            Uv = (X - c) @ (Fw - np.eye(3)).T
        Ub = np.zeros((len(rig.bodies), 3))
        Ub[order] = Uv
        rig.set_disp(Ub)
        x = d.flexvert_xpos[:len(vb)].copy()  # (nflexvert, 3) array
        fmj = rig.forces(d.qfrc_spring)[order]
        fmu, flam = tet_force_basis(X, x, tets, model)
        mu_f, lam_f, r = fit_two(fmj, fmu, flam)
        res[name] = dict(mu=mu_f, lam=lam_f, resid=r)
    return res


def tets_of(m):
    ne = m.flex_elemnum[0]
    a = m.flex_elemdataadr[0]
    return m.flex_elem[a:a + 4 * ne].reshape(ne, 4).copy()


def run_full(label, nu, xml_kw=None, model_kind="svk", spec_fn=None, remap=False, finite=True):
    xml_kw = dict(xml_kw or {})
    lam, mu = lame(E, nu)
    yin, pin = (remap_full(E, nu) if remap else (E, nu))
    xml_kw.update(young=yin, poisson=pin)
    if spec_fn is None:
        m, warn = load(xml3(**xml_kw))
    else:
        spec = mujoco.MjSpec.from_string(xml3(**xml_kw))
        spec_fn(spec)
        m, warn = compile_spec(spec)
    rig = Rig(m)
    R, _ = parent_R(rig)
    count = xml_kw.get("count", (5, 5, 5))
    L = xml_kw.get("L", (0.01, 0.01, 0.01))
    V = L[0] * L[1] * L[2]
    C, lam_e, mu_e, an = C_matrix(rig, force_fn_spring, R=R, V=V)
    lamref = lam if lam > 0 else mu  # nu = 0: lambda reported relative to mu
    out = dict(label=label, nu=nu, young_in=yin, poisson_in=pin, lam_ratio=lam_e / lamref, mu_ratio=mu_e / mu,
               M_ratio=(lam_e + 2 * mu_e) / (lam + 2 * mu), K_ratio=(lam_e + 2 * mu_e / 3) / (lam + 2 * mu / 3),
               aniso=an, C=C, warnings=warn)
    if finite and m.flex_interp[0] == 0:
        fit = ref_fit(rig, tets_of(m), model_kind, R=R)
        out["ref_fit"] = {k: dict(mu_ratio=v["mu"] / mu, lam_ratio=v["lam"] / lamref, resid=v["resid"])
                          for k, v in fit.items()}
    return out


def run_interp(label, nu, dof, cells, count=(5, 5, 5), pose="", joint="", equality="", solref=None,
               integrator="discrete"):
    lam, mu = lame(E, nu)
    opt = 'jacobian="dense"' if equality else ""
    # a strain equality cannot be combined with young > 0 (user_objects.cc:7318): young = 0 there
    xml = xml3(count=count, poisson=nu, dof=dof, cells=cells, pose=pose, joint=joint, equality=equality,
               integrator=integrator, extra_opt=opt, young=0 if equality else E)
    if solref:
        xml = xml.replace(f'<edge equality="{equality}"/>', f'<edge equality="{equality}" solref="{solref}"/>')
    m, warn = load(xml)
    rig = Rig(m)
    R, _ = parent_R(rig)
    V = 0.01 ** 3
    fn = force_fn_strain_static if equality else force_fn_spring
    C, lam_e, mu_e, an = C_matrix(rig, fn, R=R, V=V)
    out = dict(label=label, nu=nu, lam_ratio=lam_e / lam, mu_ratio=mu_e / mu, aniso=an, C=C, warnings=warn,
               qfrc_spring_max=float(np.abs(rig.d.qfrc_spring).max()))
    if equality:
        E_eff = mu_e * (3 * lam_e + 2 * mu_e) / (lam_e + mu_e)
        nu_eff = lam_e / (2 * (lam_e + mu_e))
        node_mass = float(np.mean([m.body_mass[b] for b in rig.bodies]))
        d = rig.d
        rows = [r for r in range(d.nefc) if d.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        out.update(E_eff=E_eff, nu_eff=nu_eff, node_mass=node_mass, nrows=len(rows),
                   K_row=float(d.efc_KBIP.reshape(-1, 4)[rows[0], 0]), I_row=float(d.efc_KBIP.reshape(-1, 4)[rows[0], 2]),
                   R_row=float(d.efc_R[rows[0]]), diagA_row=float(d.efc_diagA[rows[0]]))
    return out


def run_membrane(label, nu, thick=1e-3):
    lam, mu = lame(E, nu)
    lam_ps = 2 * mu * lam / (lam + 2 * mu)  # plane-stress lambda = E nu / (1 - nu^2)
    m, warn = load(xml2(poisson=nu, thick=thick))
    rig = Rig(m)
    nt = m.flex_elemnum[0]
    a = m.flex_elemdataadr[0]
    tris = m.flex_elem[a:a + 3 * nt].reshape(nt, 3)
    vb = m.flex_vertbodyid[:m.flex_vertnum[0]]
    bidx = {b: i for i, b in enumerate(rig.bodies)}
    order = np.array([bidx[b] for b in vb])
    X = rig.vert0[:len(vb)].copy()
    c = X.mean(0)
    res = {}
    cases = {"random_small": None,
             "uniaxial_small": np.diag([1 + 1e-6, 1, 1]),
             "shear_small": np.eye(3) + 1e-6 * np.outer([1, 0, 0], [0, 1, 0]),
             "biaxial_1pct": np.diag([1.01, 1.01, 1]),
             "general_1pct": np.eye(3) + 0.01 * np.array([[0.3, -0.7, 0], [0.5, -0.2, 0], [0, 0, 0]])}
    for name, Fp in cases.items():
        if Fp is None:
            Uv = 1e-7 * rng.standard_normal(X.shape)
            Uv[:, 2] = 0  # in-plane random field
        else:
            Uv = (X - c) @ (Fp - np.eye(3)).T
        Ub = np.zeros((len(rig.bodies), 3))
        Ub[order] = Uv
        rig.set_disp(Ub)
        x = rig.d.flexvert_xpos[:len(vb)].copy()
        fmj = rig.forces(rig.d.qfrc_spring)[order]
        fmu, flam = tri_force_basis(X, x, tris, thick)
        mu_f, lam_f, r = fit_two(fmj, fmu, flam)
        lr = lam_ps if lam_ps > 0 else mu
        l3 = lam if lam > 0 else mu
        res[name] = dict(mu_ratio=mu_f / mu, lam2_over_lam_planestress=lam_f / lr, lam2_over_lam3d=lam_f / l3,
                         resid=r)
    return dict(label=label, nu=nu, thickness=thick, fits=res, warnings=warn)


def main():
    results = []
    # --- dof=full StVK: nu sweep, cube 10 mm 5x5x5
    for nu in (0.0, 0.3, 0.45, 0.49):
        results.append(run_full(f"full StVK cube 5^3 nu={nu}", nu))
    # --- remap workaround
    for nu in (0.45, 0.49):
        results.append(run_full(f"full StVK remapped nu={nu}", nu, remap=True))
    # --- thin layer geometry of the study (11x15x3 over 51x71x1.5 mm)
    results.append(run_full("full StVK layer 11x15x3 nu=0.45", 0.45,
                            xml_kw=dict(count=(11, 15, 3), L=(0.051, 0.071, 0.0015))))
    # --- rotated + translated fixed parent, and a rotated parent on a free joint (articulated path)
    pose = 'pos="0.1 -0.2 0.3" euler="30 45 60"'
    results.append(run_full("full StVK rotated parent nu=0.45", 0.45, xml_kw=dict(pose=pose)))
    results.append(run_full("full StVK rotated parent w/ freejoint nu=0.45", 0.45,
                            xml_kw=dict(pose=pose, joint='<freejoint/>')))
    # --- SNH via mjSpec
    def snh(spec):
        spec.flexes[0].elastic3d = 1
    for nu in (0.3, 0.45):
        results.append(run_full(f"full SNH (elastic3d=1) nu={nu}", nu, spec_fn=snh, model_kind="snh"))
    # --- interpolated
    for dof, cells in (("trilinear", (1, 1, 1)), ("trilinear", (2, 2, 2)), ("quadratic", (1, 1, 1)),
                       ("quadratic", (2, 2, 2))):
        results.append(run_interp(f"{dof} cells={cells} nu=0.45", 0.45, dof, cells))
    results.append(run_interp("trilinear cells=(2,2,2) rotated parent nu=0.45", 0.45, "trilinear", (2, 2, 2),
                              pose=pose))
    # --- interpolated + strain equality: elasticity moves to the constraint solver
    for nu, sr in ((0.45, None), (0.1, None), (0.45, "0.001 1"), (0.45, "0.01 1")):
        results.append(run_interp(f"trilinear+strain eq nu={nu} solref={sr}", nu, "trilinear", (2, 2, 2),
                                  equality="strain", solref=sr, integrator="Euler"))
    # --- membranes (dim=2, elastic2d=stretch)
    mem = [run_membrane(f"membrane nu={nu}", nu) for nu in (0.0, 0.3, 0.45)]
    mem.append(run_membrane("membrane nu=0.3 thick=2e-3", 0.3, thick=2e-3))
    p = dump("p0_patch_ext.json", dict(solids=results, membranes=mem))
    print("wrote", p)
    print(f"{'case':55s} {'lam_eff/lam':>11s} {'mu_eff/mu':>10s} {'M/M':>7s} {'aniso':>8s}")
    for r in results:
        print(f"{r['label']:55s} {r['lam_ratio']:11.5f} {r['mu_ratio']:10.5f} "
              f"{r.get('M_ratio', float('nan')):7.4f} {r['aniso']:8.1e}")
        if "ref_fit" in r:
            for k, v in r["ref_fit"].items():
                print(f"    ref-element fit {k:16s} lam/lam={v['lam_ratio']:.5f} mu/mu={v['mu_ratio']:.5f} "
                      f"resid={v['resid']:.1e}")
        if "E_eff" in r:
            print(f"    strain-eq: E_eff={r['E_eff']:.4g} Pa nu_eff={r['nu_eff']:.4f} rows={r['nrows']} "
                  f"node_mass={r['node_mass']:.3g} K={r['K_row']:.3g} I={r['I_row']:.3g} R={r['R_row']:.3g} "
                  f"diagA={r['diagA_row']:.3g} qfrc_spring_max={r['qfrc_spring_max']:.1e}")
        if r["warnings"]:
            print("    warnings:", r["warnings"])
    for r in mem:
        print(r["label"], "thick", r["thickness"])
        for k, v in r["fits"].items():
            print(f"    {k:15s} mu_eff/mu={v['mu_ratio']:.5f}  lam2_eff/lam_planestress={v['lam2_over_lam_planestress']:.5f}"
                  f"  lam2_eff/lam3D={v['lam2_over_lam3d']:.5f} resid={v['resid']:.1e}")


if __name__ == "__main__":
    main()
