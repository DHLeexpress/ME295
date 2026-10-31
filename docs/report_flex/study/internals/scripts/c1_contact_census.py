"""Item 3a: what a 3D flex (our 11x15x3 layer, radius 0) generates as contacts with rigid geoms.

The flex is undeformed (qpos0); a rigid geom (mocap) overlaps the free face by delta. We list the contacts:
count, the flex element / vertex ids, layer of the element, position relative to the element's vertices,
the normal, and the weights with which each contact's force is spread to vertices (replica of
mj_elemVertWeight, engine_core_util.c:618: inverse distances, normalised), checked against efc_J.
Outputs: out/c1_contact_census.json
"""
import sys
from collections import Counter

import numpy as np
import mujoco

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import load, dump

FLEX = """<flexcomp name="pdms_flex" type="grid" dim="3" count="{cnt}"
              spacing="{sp}" radius="{radius}" mass="0.0052414">
      <elasticity young="{young}" poisson="{poisson}" damping="0"/>
      <contact contype="2" conaffinity="2" condim="3" priority="2" friction="1 0.005 0.0001"
               solref="{solref}" solimp="0.99 0.99 0.001" selfcollide="none" {extra_contact}/>
      <pin gridrange="0 0 {zl} {xl} {yl} {zl}"/>
    </flexcomp>"""


def layer_xml(geom, zc, count=(11, 15, 3), radius=0.0, solref="0.0001 1", refsafe=False, young=1e6, poisson=0.45,
              timestep=0.0005, iterations=100, tolerance=1e-8, extra_contact="", dof="", gravity="0 0 0",
              mocap=True, extra_body=""):
    A, B, T = 0.051, 0.071, 0.0015
    sp = f"{A / (count[0] - 1):.7f} {B / (count[1] - 1):.7f} {T / (count[2] - 1):.7f}"
    flex = FLEX.format(cnt=f"{count[0]} {count[1]} {count[2]}", sp=sp, radius=radius, young=young, poisson=poisson,
                       solref=solref, extra_contact=extra_contact, zl=count[2] - 1, xl=count[0] - 1, yl=count[1] - 1)
    if dof:
        flex = flex.replace('type="grid"', f'type="grid" dof="{dof}"')
    flag = '<flag refsafe="disable"/>' if not refsafe else ""
    body = (f'<body name="tool" mocap="true" pos="0 0 {zc}">{geom}</body>' if mocap else
            f'<body name="tool" pos="0 0 {zc}"><joint name="tz" type="slide" axis="0 0 1"/>{geom}{extra_body}</body>')
    return f"""<mujoco><option integrator="discrete" solver="CG" timestep="{timestep}" cone="elliptic" impratio="100"
 gravity="{gravity}" iterations="{iterations}" tolerance="{tolerance}">{flag}</option>
<default><default class="pdms"><joint type="slide" armature="0" damping="0" frictionloss="0"/></default></default>
<worldbody><body name="pdms" pos="0 0 0.1" childclass="pdms">{flex}</body>
{body}</worldbody></mujoco>"""


GEOMS = {
    # tool geoms; their upper extreme is placed delta above the free face z_f = 0.1 - T/2
    "box_platen": ('<geom type="box" size="0.03 0.04 0.005" contype="2" conaffinity="2"/>', 0.005),
    "sphere_R10": ('<geom type="sphere" size="0.010" contype="2" conaffinity="2"/>', 0.010),
    "capsule_R5_x": ('<geom type="capsule" size="0.005 0.01" euler="0 90 0" contype="2" conaffinity="2"/>', 0.005),
    "plane": ('<geom type="plane" size="0.1 0.1 0.01" contype="2" conaffinity="2"/>', 0.0),
    "cylinder_punch_a5": ('<geom type="cylinder" size="0.005 0.005" contype="2" conaffinity="2"/>', 0.005),
}


def elem_vert_weight(m, d, f, e, point):
    """Replica of mj_elemVertWeight (engine_core_util.c:618) for a contact on element e."""
    dim = m.flex_dim[f]
    ed = m.flex_elem[m.flex_elemdataadr[f] + e * (dim + 1):m.flex_elemdataadr[f] + (e + 1) * (dim + 1)]
    va = m.flex_vertadr[f]
    w = np.array([1.0 / max(1e-15, np.linalg.norm(point - d.flexvert_xpos[va + v])) for v in ed])
    return ed, w / w.sum()


def census(name, geom, top, delta=10e-6, **kw):
    zf = 0.1 - 0.00075
    zc = zf + delta - top
    m, warn = load(layer_xml(geom, zc, **kw))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    f = 0
    va, nv_ = m.flex_vertadr[f], m.flex_vertnum[f]
    X = d.flexvert_xpos[va:va + nv_]
    pbody = m.body("pdms").id
    vb = m.flex_vertbodyid[va:va + nv_]
    zloc = X[:, 2] - 0.1
    surf = np.isclose(zloc, -0.00075)
    out = dict(name=name, ncon=int(d.ncon), nefc=int(d.nefc))
    if d.ncon == 0:
        return out
    elems = [int(d.contact[i].elem[1]) for i in range(d.ncon)]
    verts = [int(d.contact[i].vert[1]) for i in range(d.ncon)]
    dist = np.array([d.contact[i].dist for i in range(d.ncon)])
    pos = np.array([d.contact[i].pos for i in range(d.ncon)])
    nrm = np.array([d.contact[i].frame[:3] for i in range(d.ncon)])
    lay = m.flex_elemlayer[m.flex_elemadr[f]:m.flex_elemadr[f] + m.flex_elemnum[f]]
    out.update(per_elem_layer=dict(Counter(int(lay[e]) for e in elems if e >= 0)),
               vertex_contacts=int(sum(v >= 0 for v in verts)), elem_contacts=int(sum(e >= 0 for e in elems)),
               dist_min=float(dist.min()), dist_max=float(dist.max()), dist_mean=float(dist.mean()),
               normal_mean=nrm.mean(0).tolist(), normal_min_z=float(np.abs(nrm[:, 2]).min()),
               pos_z_rel_face_um=((pos[:, 2] - zf) * 1e6).round(3).tolist()[:12])
    # weights: fraction of each contact's weight on non-surface vertices, and number of distinct contacts per
    # surface vertex region; check efc_J against the replica
    w_int = []
    jerr = []
    ncov = Counter()
    for i in range(d.ncon):
        e = elems[i]
        if e < 0:
            continue
        ed, w = elem_vert_weight(m, d, f, e, pos[i])
        w_int.append(float(sum(wk for vk, wk in zip(ed, w) if not surf[vk])))
        for vk, wk in zip(ed, w):
            if surf[vk]:
                ncov[int(vk)] += 1
        # compare normal Jacobian row with sum_k w_k n . e_axis of vertex dofs
        adr = d.contact[i].efc_address
        if adr >= 0:
            Jrow = np.zeros(m.nv)
            mujoco.mj_sparse2dense if False else None
            if mujoco.mj_isSparse(m):
                rn, ra = d.efc_J_rownnz[adr], d.efc_J_rowadr[adr]
                Jrow[d.efc_J_colind[ra:ra + rn]] = d.efc_J[ra:ra + rn]
            else:
                Jrow = d.efc_J[adr * m.nv:(adr + 1) * m.nv]
            Jpred = np.zeros(m.nv)
            n = d.contact[i].frame[:3]
            for vk, wk in zip(ed, w):
                b = vb[vk]
                if m.body_dofnum[b] == 3:
                    a = m.body_dofadr[b]
                    Jpred[a:a + 3] += wk * n  # slides along world axes (parent unrotated)
            jerr.append(float(np.abs(Jrow - Jpred).max() / max(np.abs(Jrow).max(), 1e-12)))
    if w_int:
        out.update(weight_on_interior_vertices_mean=float(np.mean(w_int)),
                   weight_on_interior_vertices_max=float(np.max(w_int)),
                   efcJ_vs_inverse_distance_replica_maxrelerr=float(max(jerr)) if jerr else None,
                   surface_vertices_touched=len(ncov), contacts_per_touched_surface_vertex_mean=float(
                       np.mean(list(ncov.values()))))
    # contact parameters as seen by the solver
    i0 = d.contact[0].efc_address
    if i0 >= 0:
        kbip = d.efc_KBIP.reshape(-1, 4)[i0]
        out.update(efc_diagA=float(d.efc_diagA[i0]), efc_R=float(d.efc_R[i0]), K=float(kbip[0]), B=float(kbip[1]),
                   imp=float(kbip[2]), inv_vertex_mass=float(1 / m.body_mass[vb[~(vb == pbody)][0]]),
                   static_k_per_contact_pred=float(kbip[0] * kbip[2] * kbip[2] / ((1 - kbip[2]) * d.efc_diagA[i0])))
    out["warnings"] = warn
    return out


def main():
    res = []
    for name, (g, top) in GEOMS.items():
        r = census(name, g, top)
        res.append(r)
        print(name, {k: v for k, v in r.items() if k not in ("pos_z_rel_face_um", "warnings")})
    # radius > 0: vertex spheres inflate the elements
    r = census("box_platen radius=0.2mm", GEOMS["box_platen"][0], 0.005, radius=0.0002)
    res.append(r)
    print(r["name"], {k: v for k, v in r.items() if k not in ("pos_z_rel_face_um", "warnings")})
    # activelayers=2
    r = census("box_platen activelayers=3", GEOMS["box_platen"][0], 0.005, extra_contact='activelayers="3"')
    res.append(r)
    print(r["name"], {k: v for k, v in r.items() if k not in ("pos_z_rel_face_um", "warnings")})
    # refsafe enabled: what happens to K
    r = census("box_platen refsafe=on", GEOMS["box_platen"][0], 0.005, refsafe=True)
    res.append(r)
    print(r["name"], {k: v for k, v in r.items() if k not in ("pos_z_rel_face_um", "warnings")})
    # interpolated flex: contacts reduced to mjMAXCONPAIR
    for dof in ("trilinear",):
        r = census(f"box_platen dof={dof} (1 cell)", GEOMS["box_platen"][0], 0.005, dof=dof)
        res.append(r)
        print(r["name"], {k: v for k, v in r.items() if k not in ("pos_z_rel_face_um", "warnings")})
    p = dump("c1_contact_census.json", res)
    print("wrote", p)


if __name__ == "__main__":
    main()
