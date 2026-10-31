"""Item 1: what the compiler makes of our flexcomp (data model, tetrahedralisation, pins, mass, bodies/joints).

(A) the exact flexcomp of asm/11x15x3/pdms_left.xml under a world-fixed parent body;
(B) the same flex inside the robot (asm/11x15x3/scene.xml): inherited joint defaults, attachment path.
Outputs: out/m1_compile_inspect.json + printed summary.
"""
import sys
from collections import Counter

import numpy as np
import mujoco

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import load, dump

SCR = "<scratch>"
ROBOT = f"{SCR}/asm/11x15x3/scene.xml"

FLEX = """<flexcomp name="pdms_flex" type="grid" dim="3" count="11 15 3"
              spacing="0.0051000 0.0050714 0.0007500" radius="0"
              mass="0.0052414" group="3" rgba="0.95 0.5 0.1 0.6">
      <elasticity young="1e+06" poisson="0.45" damping="0"/>
      <contact contype="2" conaffinity="2" condim="3" priority="2" friction="1 0.005 0.0001"
               solref="0.0001 1" solimp="0.99 0.99 0.001" selfcollide="none"/>
      <pin gridrange="0 0 2 10 14 2"/>
    </flexcomp>"""

XML_A = f"""<mujoco><option integrator="discrete" solver="CG" timestep="0.0005" cone="elliptic" impratio="100">
<flag refsafe="disable"/></option>
<default><default class="pdms"><joint type="slide" armature="0" damping="0" frictionloss="0"/></default></default>
<worldbody><body name="pdms" pos="0 0 0.1" childclass="pdms">{FLEX}</body></worldbody></mujoco>"""


def flex_body_simple(m, body):
    """Python replica of mj_flexBodySimple (engine_core_util.c:1258)."""
    body = m.body_weldid[body]
    if m.body_dofnum[body] != 3 or m.body_jntnum[body] != 3 or \
            m.body_dofnum[m.body_weldid[m.body_parentid[body]]] != 0:
        return 0
    ja = m.body_jntadr[body]
    for j in range(3):
        if m.jnt_type[ja + j] != mujoco.mjtJoint.mjJNT_SLIDE:
            return 0
        if np.max(np.abs(m.jnt_axis[ja + j] - np.eye(3)[j])) > 1e-15:
            return 0
    return 1


def flex_simple(m, f):
    """Python replica of mj_flexSimple (engine_core_util.c:1277)."""
    for v in range(m.flex_vertadr[f], m.flex_vertadr[f] + m.flex_vertnum[f]):
        b = m.body_weldid[m.flex_vertbodyid[v]]
        if m.body_dofnum[b] and not flex_body_simple(m, b):
            return 0
    return 1


def inspect_flex(m, f, rho=965.0):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    va, nv_ = m.flex_vertadr[f], m.flex_vertnum[f]
    ne = m.flex_elemnum[f]
    tets = m.flex_elem[m.flex_elemdataadr[f]:m.flex_elemdataadr[f] + 4 * ne].reshape(ne, 4)
    X = d.flexvert_xpos[va:va + nv_].copy()  # (nflexvert, 3) in the bindings
    vb = m.flex_vertbodyid[va:va + nv_]
    parent = Counter(vb).most_common(1)[0][0]
    pinned = vb == parent
    # tets per hex and the shared diagonal of every hex (grid index from position)
    Xl = (X - d.xpos[parent]) @ d.xmat[parent].reshape(3, 3)  # parent frame
    mins = Xl.min(0)
    sp = np.array([0.0051, 0.0050714, 0.00075])
    gi = np.rint((Xl - mins) / sp).astype(int)
    hexid = []
    for t in tets:
        hexid.append(tuple(gi[t].min(0)))
    per_hex = Counter(hexid)
    # shared edge of the 6 tets of hex (0,0,0)
    h0 = [i for i, h in enumerate(hexid) if h == (0, 0, 0)]
    edges = Counter()
    for i in h0:
        t = tets[i]
        for a in range(4):
            for b in range(a + 1, 4):
                edges[tuple(sorted((t[a], t[b])))] += 1
    shared = [e for e, c in edges.items() if c == 6]
    diag = [(gi[e[0]].tolist(), gi[e[1]].tolist()) for e in shared]
    # free-face (z index 0) triangle diagonals: edges with both ends at iz=0 that are not grid-aligned
    face_diag = Counter()
    for t in tets:
        for a in range(4):
            for b in range(a + 1, 4):
                p, q = gi[t[a]], gi[t[b]]
                if p[2] == 0 and q[2] == 0 and abs(p[0] - q[0]) == 1 and abs(p[1] - q[1]) == 1:
                    dvec = (q - p) * np.sign((q - p)[0])
                    face_diag[tuple(dvec.tolist())] += 1
    # element layers and active elements
    layers = Counter(m.flex_elemlayer[m.flex_elemadr[f]:m.flex_elemadr[f] + ne].tolist())
    active = int(np.sum(m.flex_elemlayer[m.flex_elemadr[f]:m.flex_elemadr[f] + ne] < m.flex_activelayers[f]))
    # mass: MuJoCo vs consistent lumping (volume/4 per tet vertex)
    vol = np.abs(np.einsum("ij,ij->i", np.cross(X[tets[:, 1]] - X[tets[:, 0]], X[tets[:, 2]] - X[tets[:, 0]]),
                           X[tets[:, 3]] - X[tets[:, 0]])) / 6
    lump = np.zeros(nv_)
    np.add.at(lump, tets.ravel(), np.repeat(vol / 4, 4))
    lump *= rho
    mj_mass = np.array([m.body_mass[b] if not p else 0.0 for b, p in zip(vb, pinned)])
    free = ~pinned
    ratio = mj_mass[free] / lump[free]
    # vertex body joints
    jb = [b for b in set(vb[free].tolist())]
    dofs = np.concatenate([np.arange(m.body_dofadr[b], m.body_dofadr[b] + 3) for b in jb])
    out = dict(
        nvert=int(nv_), nelem=int(ne), nedge=int(m.flex_edgenum[f]), npinned=int(pinned.sum()),
        nfree=int(free.sum()), nbody_created=len(jb), ndof=int(len(dofs)),
        tets_per_hex=sorted(set(per_hex.values())), nhex=len(per_hex),
        hex000_shared_diagonal_gridcoords=diag, freeface_diagonal_dirs=dict((str(k), v) for k, v in face_diag.items()),
        elem_layers=dict(layers), activelayers=int(m.flex_activelayers[f]), active_elems=active,
        volume=float(vol.sum()), mass_attr_total=float(mj_mass.sum() + 0), mass_lumped_free=float(lump[free].sum()),
        mass_lumped_all=float(lump.sum()),
        vertex_body_mass=float(mj_mass[free][0]), mass_ratio_mj_over_lumped=[float(ratio.min()), float(ratio.max())],
        pinned_vertbody=m.body(parent).name, pinned_vert_local=m.flex_vert[va:va + 3].tolist(),
        dof_armature=sorted(set(m.dof_armature[dofs].tolist())), dof_damping=sorted(set(m.dof_damping[dofs].tolist())),
        dof_frictionloss=sorted(set(m.dof_frictionloss[dofs].tolist())),
        jnt_stiffness=sorted(set(m.jnt_stiffness[[m.dof_jntid[k] for k in dofs]].tolist())),
        body_invweight0_trans=[float(np.ravel(m.body_invweight0)[2 * b]) for b in jb[:3]],
        inv_vertex_mass=float(1 / mj_mass[free][0]),
        flex_simple=flex_simple(m, f), flex_rigid=int(m.flex_rigid[f]), flex_centered=int(m.flex_centered[f]),
        flex_interp=int(m.flex_interp[f]), radius=float(m.flex_radius[f]), margin=float(m.flex_margin[f]),
        gap=float(m.flex_gap[f]), solref=m.flex_solref[f].tolist(), solimp=m.flex_solimp[f].tolist(),
        priority=int(m.flex_priority[f]), condim=int(m.flex_condim[f]), friction=m.flex_friction[f].tolist(),
        selfcollide=int(m.flex_selfcollide[f]), stiffness_record_len=int(m.nflexstiffness // max(m.nflexelem, 1)),
        parent_weld_dofnum=int(m.body_dofnum[m.body_weldid[parent]]),
        parent_weld_body=m.body(m.body_weldid[parent]).name,
    )
    return out


def main():
    res = {}
    mA, warnA = load(XML_A)
    res["A_fixed"] = inspect_flex(mA, 0)
    res["A_fixed"]["compile_warnings"] = warnA
    with open(ROBOT) as fh:
        pass
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mB = mujoco.MjModel.from_xml_path(ROBOT)
    fL = mujoco.mj_name2id(mB, mujoco.mjtObj.mjOBJ_FLEX, "left_pdms_flex")
    res["B_robot_left"] = inspect_flex(mB, fL)
    res["B_robot_left"]["compile_warnings"] = [str(x.message) for x in w]
    res["B_robot_opt"] = dict(timestep=mB.opt.timestep, integrator=int(mB.opt.integrator), solver=int(mB.opt.solver),
                              iterations=int(mB.opt.iterations), tolerance=mB.opt.tolerance,
                              ls_iterations=int(mB.opt.ls_iterations), cone=int(mB.opt.cone),
                              impratio=mB.opt.impratio, disableflags=int(mB.opt.disableflags),
                              enableflags=int(mB.opt.enableflags), nv=int(mB.nv), nflex=int(mB.nflex),
                              memory_narena=int(mB.narena), njmax=int(mB.njmax), nconmax=int(mB.nconmax))
    p = dump("m1_compile_inspect.json", res)
    print("wrote", p)
    for k, v in res.items():
        print("==", k)
        for kk, vv in v.items():
            print(f"   {kk}: {vv}")


if __name__ == "__main__":
    main()
