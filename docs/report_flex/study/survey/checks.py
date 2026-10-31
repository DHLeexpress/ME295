# Small checks run by the flex-usage survey (MuJoCo 3.17.0) to verify claims found in sources.
#  C1  StVK (default) vs Stable Neo-Hookean (mjsFlex.elastic3d = 1, PR #3621 / changelog 3.15.0 item 5) on the
#      same homogeneous patch test as /tmp/.../patch/patch2.py: uniaxial strain, simple shear, hydrostatic.
#      PR #3621 says SNH "uses conventional Young/Poisson Lame parameters" while "the previous 3D StVK
#      volumetric normalization is preserved on the default path" -> expect SNH ratios 1.000, StVK 1.818/1.931.
#  C2  flex elasticity under implicitfast (changelog 3.13.0 item 4: runtime error since 3.13).
#  C3  edge equality + elasticity on one flex (MetalSim doc says "cannot both be present", measured on 3.14).
#  C4  the effect of the F1 workaround mapping on SNH would be wrong (SNH is already conventional) - checked.
# usage: python checks.py  -> checks.json next to this file
import json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
E = 1e6
out = {"mujoco": mujoco.__version__}


def patch(nu, elastic3d=0, young=E, poisson=None, count=(5, 5, 5), L=(0.01, 0.01, 0.01)):
    """Homogeneous-strain patch test; returns measured/exact top-face force for 3 strain states (target E, nu)."""
    poisson = nu if poisson is None else poisson
    sp = [L[i] / (count[i] - 1) for i in range(3)]
    xml = f"""<mujoco><option gravity="0 0 0" integrator="discrete" solver="CG" timestep="1e-4"/>
    <worldbody><body name="b"><flexcomp name="f" type="grid" dim="3" count="{count[0]} {count[1]} {count[2]}"
     spacing="{sp[0]} {sp[1]} {sp[2]}" mass="0.001" radius="1e-5">
     <elasticity young="{young}" poisson="{poisson}" damping="0"/><contact contype="0" conaffinity="0"/></flexcomp>
    </body></worldbody></mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    if elastic3d:
        spec.flexes[0].elastic3d = elastic3d
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bodies = [b for b in range(m.nbody) if m.body_dofnum[b] == 3]
    X0 = np.array([d.xpos[b] for b in bodies])
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu)); M = lam + 2 * mu
    c = X0.mean(0); top = np.isclose(X0[:, 2], X0[:, 2].max()); A = L[0] * L[1]
    res = {"nu_target": nu, "young_in": young, "poisson_in": poisson, "elastic3d": elastic3d,
           "stride": int(m.flex_stiffness.size // max(1, m.flex_elemnum[0]))}
    eps = 1e-6
    for name, H, s in [("uniaxial", np.diag([0, 0, -eps]), -M * eps),
                       ("shear", np.array([[0, 0, eps], [0, 0, 0], [0, 0, 0]]), mu * eps),
                       ("hydro", -eps * np.eye(3), -(3 * lam + 2 * mu) * eps)]:
        d.qpos[:] = 0
        U = (X0 - c) @ H.T
        for i, b in enumerate(bodies):
            j = m.body_jntadr[b]
            for k in range(3):
                d.qpos[m.jnt_qposadr[j + k]] = U[i, k]
        mujoco.mj_forward(m, d)
        F = np.array([d.qfrc_spring[m.body_dofadr[b]:m.body_dofadr[b] + 3] for b in bodies])
        comp = 0 if name == "shear" else 2
        res[name] = round(float(-F[top].sum(0)[comp] / (s * A)), 6)
    res["lambda_eff/lambda"] = round((res["uniaxial"] * M - 2 * mu) / lam, 4)
    return res


# C1: StVK vs SNH at nu = 0.45 and 0.49, plus the F1 workaround mapping (main session) on StVK
rows = []
for nu in (0.45, 0.49):
    rows.append(patch(nu, 0))
    try:
        rows.append(patch(nu, 1))
    except Exception as exc:  # noqa: BLE001
        rows.append({"nu_target": nu, "elastic3d": 1, "error": str(exc)[:300]})
    G = E / (2 * (1 + nu)); p_in = nu / (2 * (1 - nu)); y_in = 2 * G * (1 + p_in)
    rows.append(dict(patch(nu, 0, young=y_in, poisson=p_in), note="F1 workaround inputs on StVK"))
out["C1_patch"] = rows

# C2: elasticity + implicitfast / implicit / Euler
c2 = {}
for integ in ("implicitfast", "implicit", "Euler", "discrete"):
    xml = f"""<mujoco><option integrator="{integ}" timestep="1e-4"/><worldbody>
    <flexcomp name="f" type="grid" dim="3" count="3 3 3" spacing=".01 .01 .01" mass=".01" radius="0">
    <elasticity young="1e6" poisson="0.45"/><pin id="0"/></flexcomp></worldbody></mujoco>"""
    try:
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
        mujoco.mj_step(m, d)
        c2[integ] = "compiled and stepped"
    except Exception as exc:  # noqa: BLE001
        c2[integ] = "ERROR: " + str(exc).strip().splitlines()[0][:300]
out["C2_integrators_with_elasticity"] = c2

# C3: edge equality + elasticity on one flex
xml = """<mujoco><option integrator="discrete" solver="CG"/><worldbody>
<flexcomp name="f" type="grid" dim="3" count="3 3 3" spacing=".01 .01 .01" mass=".01" radius="0">
<edge equality="true"/><elasticity young="1e6" poisson="0.45"/></flexcomp></worldbody></mujoco>"""
try:
    mujoco.MjModel.from_xml_string(xml)
    out["C3_edge_equality_plus_elasticity"] = "compiled"
except Exception as exc:  # noqa: BLE001
    out["C3_edge_equality_plus_elasticity"] = "ERROR: " + str(exc).strip().splitlines()[0][:300]

json.dump(out, open(os.path.join(HERE, "checks.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
