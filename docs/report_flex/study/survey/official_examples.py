# Tabulate the official MuJoCo 3.17.0 flex example models (model/flex, model/flex/mechanisms): the flex settings
# (from mjSpec), the compiled sizes, whether the flex hangs on a moving body, and a short stepping test.
# Also an explicit-stability (CFL-like) number for each elastic 3D flex: h * c_p / l_min, with
#   c_p = sqrt((lambda + 2 mu) / rho), rho = flexcomp mass / tetrahedral volume, l_min = shortest edge.
# usage: python official_examples.py [seconds_per_model]  -> official_examples.json
import glob, json, math, os, sys, time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "<scratch>/src/mujoco/model/flex"
TSIM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
INTEG = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
files = sorted(glob.glob(SRC + "/*.xml") + glob.glob(SRC + "/mechanisms/*.xml"))
files = [f for f in files if os.path.basename(f) != "scene.xml"]
rows = []
for f in files:
    row = {"file": os.path.relpath(f, SRC)}
    try:
        spec = mujoco.MjSpec.from_file(f)
        m = spec.compile()
    except Exception as exc:  # noqa: BLE001
        row["compile_error"] = str(exc).strip().splitlines()[0][:300]
        rows.append(row); continue
    o = m.opt
    row["option"] = dict(timestep=o.timestep, integrator=mujoco.mjtIntegrator(o.integrator).name,
                         solver=mujoco.mjtSolver(o.solver).name, cone=mujoco.mjtCone(o.cone).name,
                         iterations=int(o.iterations), tolerance=o.tolerance, impratio=o.impratio)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    flexes = []
    for i, fs in enumerate(spec.flexes):
        fl = {"name": fs.name, "dim": int(m.flex_dim[i]), "interp": int(m.flex_interp[i]),
              "cellnum": [int(v) for v in m.flex_cellnum[i]] if m.flex_interp[i] else None,
              "nvert": int(m.flex_vertnum[i]), "nelem": int(m.flex_elemnum[i]), "nnode": int(m.flex_nodenum[i]),
              "young": fs.young, "poisson": fs.poisson, "damping": fs.damping, "thickness": fs.thickness,
              "elastic2d": int(fs.elastic2d), "radius": float(m.flex_radius[i]), "margin": float(m.flex_margin[i]),
              "solref": [float(v) for v in m.flex_solref[i]], "solimp": [float(v) for v in m.flex_solimp[i]],
              "edgeequality": int(m.flex_edgeequality[i]), "rigid": bool(m.flex_rigid[i])}
        # moving attachment: any vertex/node body whose ancestors (excluding itself) carry dofs
        adr = m.flex_nodeadr[i] if m.flex_interp[i] else m.flex_vertadr[i]
        num = m.flex_nodenum[i] if m.flex_interp[i] else m.flex_vertnum[i]
        bodies = (m.flex_nodebodyid if m.flex_interp[i] else m.flex_vertbodyid)[adr:adr + num]
        moving = False
        for b in set(int(x) for x in bodies):
            p = int(m.body_parentid[b]) if m.body_dofnum[b] == 3 else b  # pinned vertices sit on the parent itself
            while p > 0:
                if m.body_dofnum[p] > 0:
                    moving = True; break
                p = int(m.body_parentid[p])
        fl["attached_to_moving_body"] = moving
        # explicit-stability number for elastic 3D non-interpolated flexes
        if fl["dim"] == 3 and fs.young > 0:
            va, vn = m.flex_vertadr[i], m.flex_vertnum[i]
            X = d.flexvert_xpos[va:va + vn]
            E4 = m.flex_elem[m.flex_elemdataadr[i]:m.flex_elemdataadr[i] + 4 * m.flex_elemnum[i]].reshape(-1, 4)
            vol = float(np.abs(np.einsum("ij,ij->i", np.cross(X[E4[:, 1]] - X[E4[:, 0]], X[E4[:, 2]] - X[E4[:, 0]]),
                                         X[E4[:, 3]] - X[E4[:, 0]])).sum() / 6)
            mass = float(sum(m.body_mass[int(b)] for b in m.flex_vertbodyid[va:va + vn] if m.body_dofnum[int(b)] == 3))
            nfree = sum(1 for b in m.flex_vertbodyid[va:va + vn] if m.body_dofnum[int(b)] == 3)
            mass_total = mass * vn / max(nfree, 1)  # flexcomp mass is divided over all points, pinned included
            ea = m.flex_edgeadr[i]; en = m.flex_edgenum[i]
            ed = m.flex_edge[ea:ea + en]
            lmin = float(np.min(np.linalg.norm(X[ed[:, 0]] - X[ed[:, 1]], axis=1)))
            E, nu = fs.young, fs.poisson
            lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
            rho = mass_total / vol if vol > 0 else float("nan")
            cp = math.sqrt((lam + 2 * mu) / rho)
            fl.update(volume_m3=vol, rho_kg_m3=rho, l_min_m=lmin, c_p_m_s=cp, cfl=o.timestep * cp / lmin,
                      size_m=[float(v) for v in (X.max(0) - X.min(0))])
        flexes.append(fl)
    row["flexes"] = flexes
    # stepping test
    nstep = int(round(TSIM / o.timestep))
    t0 = time.perf_counter(); bad = None
    try:
        for k in range(nstep):
            mujoco.mj_step(m, d)
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number or not np.isfinite(d.qpos).all():
                bad = k; break
    except Exception as exc:  # noqa: BLE001
        row["step_error"] = str(exc).strip().splitlines()[0][:300]
    dt = time.perf_counter() - t0
    row["step"] = dict(sim_time=TSIM, nstep=nstep, wall_s=round(dt, 3), us_per_step=round(1e6 * dt / max(1, nstep), 1),
                       diverged_at_step=bad, nv=int(m.nv), ncon_end=int(d.ncon),
                       warnings={mujoco.mjtWarning(w).name: int(d.warning[w].number)
                                 for w in range(int(mujoco.mjtWarning.mjNWARNING)) if d.warning[w].number})
    rows.append(row)
    print(json.dumps({"file": row["file"], "step": row.get("step"), "err": row.get("compile_error") or row.get("step_error")}), flush=True)
json.dump({"mujoco": mujoco.__version__, "sim_time_per_model": TSIM, "rows": rows},
          open(os.path.join(HERE, "official_examples.json"), "w"), indent=1)
