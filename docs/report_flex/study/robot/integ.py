"""Minimal repro under each integrator / solver combination (islands on/off). python integ.py [xml] [h]"""
import sys, mujoco, json
xml = sys.argv[1]; h = float(sys.argv[2]) if len(sys.argv) > 2 else None
for integ in ("Euler", "RK4", "implicit", "implicitfast", "discrete"):
    for solver in ("PGS", "CG", "Newton"):
        for island in (True, False):
            try:
                m = mujoco.MjModel.from_xml_path(xml)
                m.opt.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + integ.upper())
                m.opt.solver = getattr(mujoco.mjtSolver, "mjSOL_" + solver.upper())
                if h: m.opt.timestep = h
                if not island: m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
                d = mujoco.MjData(m); bad = None
                for k in range(200):
                    mujoco.mj_step(m, d)
                    if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: bad = k + 1; break
                r = f"first BADQACC {bad}"
            except Exception as e:
                r = "ERROR " + str(e).splitlines()[0][:110]
            print(f"{integ:13s} {solver:7s} islands {'on ' if island else 'off'}: {r}", flush=True)
