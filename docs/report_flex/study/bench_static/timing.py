# Cost side of the trade-off: MuJoCo compile time, passive-force evaluation time and mj_step time (pinned pad,
# gravity, no contact, integrator=discrete, default opt.iterations=100 and tolerance 1e-8) per model, and
# whether MuJoCo's effective-metric PCG warns. Usage: python timing.py '<json config>' out.jsonl
import sys, os, json, time
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def run(c, nstep=20):
    t0 = time.time()
    m = mujoco.MjModel.from_xml_string(make_xml(c, pinned=True, gravity=True))
    tcomp = time.time() - t0
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q0 = d.qpos.copy()
    # passive force only (what FD calls)
    ts = []
    for _ in range(5):
        t1 = time.perf_counter(); passive_force(m, d, q0); ts.append(time.perf_counter() - t1)
    tpass = float(np.median(ts))
    # full steps
    mujoco.mj_resetData(m, d)
    tst = []
    for _ in range(nstep):
        t1 = time.perf_counter(); mujoco.mj_step(m, d); tst.append(time.perf_counter() - t1)
    warn = int(d.warning[mujoco.mjtWarning.mjWARN_INERTIA].number)
    return dict(id=cfg_id(c), config=describe(c), nv=int(m.nv), nbody=int(m.nbody), compile_s=tcomp,
                passive_s=tpass, step_s_median=float(np.median(tst)), step_s_min=float(np.min(tst)),
                inertia_warnings_in_steps=warn, finite=bool(np.isfinite(d.qpos).all()),
                max_disp_um_after_steps=float(np.abs(d.qpos).max() * 1e6))


if __name__ == "__main__":
    c = json.loads(sys.argv[1])
    r = run(c)
    with open(sys.argv[2], "a") as f:
        f.write(json.dumps(to_json(r)) + "\n")
    print(r["id"], "%.2f ms/step" % (1e3 * r["step_s_median"]), "warn", r["inertia_warnings_in_steps"])
