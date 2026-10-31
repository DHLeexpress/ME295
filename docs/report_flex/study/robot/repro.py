"""Reproduce F4: full flex on the robot, home pose, servo holding; log the first steps.

python repro.py '{"grid": "11x15x3", "steps": 10}'  -> prints per-step diagnostics
"""
import sys
import json
import numpy as np
import mujoco
from common import load, reset_home, nbad, warns, flex_dofs

o = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
m, d = load(o.get("grid", "11x15x3"), o)
reset_home(m, d)
fd = flex_dofs(m)
arm = np.arange(6)
grip = np.array([m.joint(n).dofadr[0] for n in ("left_driver_joint", "left_follower_joint", "left_inner_knuckle_joint",
                                               "right_driver_joint", "right_follower_joint", "right_inner_knuckle_joint")])
print(f"nv {m.nv} flex dofs {len(fd)}  opt: h {m.opt.timestep} solver {m.opt.solver} it {m.opt.iterations} tol {m.opt.tolerance}")
for k in range(o.get("steps", 10)):
    mujoco.mj_step(m, d)
    qa = d.qacc
    print(f"step {k+1:3d} t {d.time:.4f} bad {nbad(d)} ncon {d.ncon:5d} nefc {d.nefc:6d} niter {int(d.solver_niter[0]):4d} "
          f"nefmdof {d.nefmdof} efm_active {d.efm_active} | max|qacc| arm {np.abs(qa[arm]).max():.3e} grip {np.abs(qa[grip]).max():.3e} "
          f"flex {np.abs(qa[fd]).max():.3e} | max|qvel| arm {np.abs(d.qvel[arm]).max():.2e} flex {np.abs(d.qvel[fd]).max():.2e} "
          f"| max|qpos flex| {np.abs(d.qpos[fd]).max():.2e}", flush=True)
print("warnings", warns(d))
