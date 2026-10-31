"""Item 4: the effective-metric solve (mj_effSolve, engine_metric.c:1638) for our layer.

For each configuration we report what mj_effBuild assembled (nefmK = CSR nonzeros of the flex stiffness,
nefmdof = number of 3x3 diagonal blocks of the preconditioner) and the number of PCG iterations mj_effSolve
needs to reach opt.tolerance. The count is measured without instrumenting MuJoCo: mj_effSolve warns
(mjWARN_INERTIA) iff it stops at opt.iterations with the residual above tolerance, so the smallest
opt.iterations that does not warn is the iteration count (bisection, one mj_forward per probe, same state).
Configurations: flex pinned to a world-fixed body (MuJoCo's "simple" path: assembled CSR + 3x3 blocks)
vs the same flex pinned to a body on a hinge (articulated attachment: matrix-free J'KJ, no blocks) vs the
robot; mesh sweep. Outputs: out/i1_metric_pcg.json
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import mujoco

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import load, dump

SCR = "<scratch>"


def xml_layer(count, parent="fixed", young=1e6, poisson=0.45, h=0.0005, tol=1e-8):
    A, B, T = 0.051, 0.071, 0.0015
    sp = f"{A / (count[0] - 1):.7f} {B / (count[1] - 1):.7f} {T / (count[2] - 1):.7f}"
    joint = {"fixed": "", "hinge": '<joint name="hj" type="hinge" axis="1 0 0"/>',
             "slide": '<joint name="sj" type="slide" axis="0 0 1"/>', "free": "<freejoint/>"}[parent]
    inertial = '<inertial pos="0 0 0.01" mass="0.12" diaginertia="4e-4 6e-5 4e-4"/>' if parent != "fixed" else ""
    return f"""<mujoco><option integrator="discrete" solver="CG" timestep="{h}" tolerance="{tol}" cone="elliptic"
 impratio="100"><flag refsafe="disable"/></option>
<default><default class="pdms"><joint type="slide" armature="0" damping="0" frictionloss="0"/></default></default>
<worldbody><body name="pdms" pos="0 0 0.1" childclass="pdms">{joint}{inertial}
<flexcomp name="pdms_flex" type="grid" dim="3" count="{count[0]} {count[1]} {count[2]}" spacing="{sp}" radius="0"
 mass="0.0052414"><elasticity young="{young}" poisson="{poisson}" damping="0"/>
 <contact contype="2" conaffinity="2" condim="3" priority="2" friction="1 0.005 0.0001" solref="0.0001 1"
  solimp="0.99 0.99 0.001" selfcollide="none"/>
 <pin gridrange="0 0 {count[2]-1} {count[0]-1} {count[1]-1} {count[2]-1}"/></flexcomp></body></worldbody></mujoco>"""


def pcg_iters(m, d, state_fn, nmax=200000):
    """Smallest opt.iterations for which mj_effSolve does not warn (= its PCG iteration count)."""
    W = mujoco.mjtWarning.mjWARN_INERTIA

    def warns(n):
        m.opt.iterations = n
        state_fn(d)
        d.warning[W].number = 0
        mujoco.mj_forward(m, d)
        return d.warning[W].number > 0

    lo, hi = 0, 1
    while warns(hi):
        lo, hi = hi, hi * 2
        if hi > nmax:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if warns(mid):
            lo = mid
        else:
            hi = mid
    return hi


def case(args):
    label, count, parent, rhs = args
    m, _ = load(xml_layer(count, parent))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(1)
    qv = 1e-4 * rng.standard_normal(m.nv)

    def state(dd):
        dd.qpos[:] = m.qpos0
        dd.qvel[:] = qv if rhs == "gravity+qvel" else 0

    state(d)
    m.opt.iterations = 100
    t0 = time.time()
    mujoco.mj_forward(m, d)
    t_fwd = time.time() - t0
    info = dict(label=label, count=count, parent=parent, rhs=rhs, nv=int(m.nv), nefmK=int(d.nefmK),
                nefmdof=int(d.nefmdof), efm_active=int(d.efm_active), ms_forward_it100=1e3 * t_fwd)
    t0 = time.time()
    it = pcg_iters(m, d, state)
    info["pcg_iterations_to_tol"] = it
    info["probe_wall_s"] = time.time() - t0
    m.opt.iterations = max(it or 100, 1)
    state(d)
    t0 = time.time()
    mujoco.mj_forward(m, d)
    info["ms_forward_converged"] = 1e3 * (time.time() - t0)
    return info


def robot_case(_):
    m = mujoco.MjModel.from_xml_path(f"{SCR}/asm/11x15x3/scene.xml")
    d = mujoco.MjData(m)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")

    def state(dd):
        mujoco.mj_resetDataKeyframe(m, dd, k)

    state(d)
    m.opt.iterations = 100
    mujoco.mj_forward(m, d)
    info = dict(label="robot 11x15x3 (home keyframe)", nv=int(m.nv), nefmK=int(d.nefmK), nefmdof=int(d.nefmdof),
                efm_active=int(d.efm_active))
    t0 = time.time()
    info["pcg_iterations_to_tol"] = pcg_iters(m, d, state)
    info["probe_wall_s"] = time.time() - t0
    return info


def main():
    jobs = []
    for cnt in ((6, 8, 2), (11, 15, 2), (11, 15, 3), (21, 29, 3), (11, 15, 5), (21, 29, 5), (41, 57, 5)):
        for parent in ("fixed", "hinge"):
            jobs.append((f"{cnt} {parent}", cnt, parent, "gravity"))
    jobs.append(("(11,15,3) fixed qvel", (11, 15, 3), "fixed", "gravity+qvel"))
    jobs.append(("(11,15,3) hinge qvel", (11, 15, 3), "hinge", "gravity+qvel"))
    jobs.append(("(11,15,3) slide", (11, 15, 3), "slide", "gravity"))
    jobs.append(("(11,15,3) free", (11, 15, 3), "free", "gravity"))
    with Pool(12) as pool:
        res = pool.map(case, jobs)
        rob = pool.map(robot_case, [0])
    for r in res + rob:
        print(r)
    p = dump("i1_metric_pcg.json", dict(cases=res, robot=rob))
    print("wrote", p)


if __name__ == "__main__":
    main()
