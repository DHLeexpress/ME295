"""Build the MuJoCo assets for the custom blade (gripper_dohyun_vN.stl).

The blade replaces the Robotiq AGC fingertip: it bolts to the 2F-140 inner finger through the
same lug hole. On its flat object-facing side (-z in the STL) there is a thin extrusion, the
PDMS layer. This script

  1. finds the PDMS extrusion (the largest flat -z face that is a full rectangle, and the
     plate face it stands proud of) and splits the STL into blade body + PDMS for colouring,
  2. convex-decomposes the blade (CoACD) for collision; pieces under the PDMS are cut back to
     the plate face so the PDMS is what touches objects,
  3. writes the MuJoCo include files:

    model/<stem>_visual/<stem>_{body,pdms}.stl    visual meshes (STL units)
    model/<stem>_collision/<stem>_NN.stl          convex pieces (metres)
    assembly/blade_assets.xml                     <mesh> entries
    assembly/blade_geoms_{left,right}.xml         collision pieces (MuJoCo will not include
                                                  one file twice)
    assembly/pdms_{left,right}.xml                the PDMS layer as a deformable body, bonded to the
                                                  blade at its back face, either
                                                  --pdms-model flex: a 3D finite-element flex (MuJoCo
                                                    flexcomp, linear tetrahedra), back-face vertices
                                                    pinned to the blade, contact on its free face; or
                                                  --pdms-model cells: a grid of cells, each a column of
                                                    PDMS on three elastic slide joints (compression +
                                                    two shear directions), one contact point per cell
                                                    (thin-layer / Winkler limit)

Usage: python scripts_sim/make_blade_assets.py model/gripper_dohyun_v1.stl [--young 1e6 ...]
       python scripts_sim/make_blade_assets.py model/gripper_dohyun_v1.stl --pdms-only --pdms-model cells
Re-run whenever the blade STL or the PDMS parameters change (--pdms-only skips the convex
decomposition and rewrites only assembly/pdms_*.xml).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import coacd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def flat_levels(mesh, normal):
    """Planar regions facing `normal`: {offset along normal: face indices}."""
    sel = np.where(mesh.face_normals @ normal > 0.999)[0]
    groups = defaultdict(list)
    for i, level in zip(sel, np.round(mesh.triangles_center[sel] @ normal, 2)):
        groups[level].append(i)
    return groups


def find_pdms(mesh):
    """PDMS extrusion on the -z side: (x/y footprint min, max, z of its face, z of the plate)."""
    groups = flat_levels(mesh, np.array([0.0, 0.0, -1.0]))
    face = max(groups.values(), key=lambda g: mesh.area_faces[g].sum())
    v = mesh.triangles[face].reshape(-1, 3)
    lo, hi = v.min(0)[:2], v.max(0)[:2]
    fill = mesh.area_faces[face].sum() / np.prod(hi - lo)
    z_face = v[0, 2]
    # the plate face the extrusion stands on: the nearest -z facing plane behind it
    behind = [-lvl for lvl in groups if z_face < -lvl <= z_face + 3.0]
    if fill < 0.95 or not behind:
        raise SystemExit(f"no PDMS extrusion found (fill {fill:.2f}); check the blade STL")
    return lo, hi, z_face, min(behind)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stl", type=Path)
    ap.add_argument("--scale", type=float, default=1e-3, help="STL units -> metres (STL is in mm)")
    ap.add_argument("--threshold", type=float, default=0.03, help="CoACD concavity threshold")
    ap.add_argument("--resolution", type=int, default=200, help="CoACD remesh resolution (sets surface error)")
    # PDMS layer (placeholders until the material is chosen: Sylgard-184-like), linear elastic
    ap.add_argument("--pdms-model", choices=["flex", "cells"], default="cells",
                    help="flex: 3D finite elements (MuJoCo flexcomp); cells: thin-layer (Winkler) cells")
    ap.add_argument("--thickness", type=float, default=1.5e-3,
                    help="PDMS thickness (m). The STL extrusion is the part proud of the blade face (0.5 mm); "
                         "the rest sits in a pocket behind it, so the back face is this far behind the free face")
    ap.add_argument("--young", type=float, default=1.0e6, help="PDMS Young's modulus (Pa)")
    ap.add_argument("--poisson", type=float, default=0.45, help="PDMS Poisson's ratio (real PDMS: ~0.49)")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="cells: Kelvin-Voigt time constant (s), damping = tau * stiffness; flex: Rayleigh "
                         "damping (s). 0 = linear elastic, no damping (default)")
    ap.add_argument("--density", type=float, default=965.0, help="PDMS density (kg/m^3)")
    ap.add_argument("--cells", default="5x7", help="cells: grid across x (blade width) x along y (finger)")
    ap.add_argument("--flex-grid", default="6x8x2",
                    help="flex: vertices across x, along y, through the thickness. 6x8x2 (one element per taxel, one "
                         "through the thickness) is the robot choice of docs/report_flex (21 ms per step with the "
                         "stiff contact); 41x57x3 is the fixed-rig choice. More than 3 vertices through the thickness "
                         "makes the contact softer, not the result better (lighter vertices)")
    # contact on the free face
    ap.add_argument("--mu", type=float, default=1.0, help="PDMS sliding friction")
    ap.add_argument("--solref", default=None,
                    help="contact solref (time constant, damping ratio). cells (default 0.003 1): the "
                         "compliance of the object's surface (e.g. a book cover), not of the PDMS. flex "
                         "(default 0.0001 1, below the time step: needs <flag refsafe='disable'/>): a stiff "
                         "contact, so the deformation is the PDMS's")
    ap.add_argument("--solimp", default=None, help="contact solimp (dmin dmax width); cells default "
                    "0.9 0.95 0.002, flex 0.99 0.99 0.001")
    ap.add_argument("--pdms-only", action="store_true", help="only rewrite assembly/pdms_*.xml")
    args = ap.parse_args()

    stl = args.stl.resolve()
    s = args.scale
    raw = trimesh.load(stl, force="mesh")

    # 1. PDMS extrusion and visual split
    lo, hi, z_face, z_plate = find_pdms(raw)
    tri = raw.triangles
    tol = 0.05
    in_pdms = (np.all((tri[:, :, 0] >= lo[0] - tol) & (tri[:, :, 0] <= hi[0] + tol), axis=1)
               & np.all((tri[:, :, 1] >= lo[1] - tol) & (tri[:, :, 1] <= hi[1] + tol), axis=1)
               & np.all((tri[:, :, 2] >= z_face - tol) & (tri[:, :, 2] <= z_plate + tol), axis=1))
    header = (f"<!-- generated by scripts_sim/make_blade_assets.py from "
              f"{stl.relative_to(ROOT / 'model').as_posix()}; do not edit -->\n")
    write_pdms(args, lo, hi, z_face, z_plate, header)
    if args.pdms_only:
        return
    vis = stl.parent / f"{stl.stem}_visual"
    vis.mkdir(exist_ok=True)
    raw.submesh([np.where(~in_pdms)[0]], append=True).export(vis / f"{stl.stem}_body.stl")
    raw.submesh([np.where(in_pdms)[0]], append=True).export(vis / f"{stl.stem}_pdms.stl")

    # 2. collision pieces
    mesh = raw.copy()
    mesh.merge_vertices()
    mesh.apply_scale(s)
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=args.threshold,
                            preprocess_mode="on", preprocess_resolution=args.resolution, max_convex_hull=40)
    out = stl.parent / f"{stl.stem}_collision"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.stl"):
        old.unlink()
    names, cut = [], 0
    for v, f in parts:
        piece = trimesh.Trimesh(np.asarray(v), np.asarray(f))
        b = piece.bounds / s
        over_pdms = (b[0, 0] < hi[0] and b[1, 0] > lo[0] and b[0, 1] >= lo[1] - 2.0 and b[1, 1] > lo[1]
                     and b[0, 2] < z_plate)
        if over_pdms:  # plate piece: cut it back to the plate face so only the PDMS sticks out
            # convex piece: the hull of the sliced surface is the capped piece
            piece = piece.slice_plane([0, 0, z_plate * s], [0, 0, 1]).convex_hull
            cut += 1
        if piece.volume < 1e-9:  # drop slivers (< 1 mm^3)
            continue
        name = f"{stl.stem}_{len(names):02d}"
        piece.export(out / f"{name}.stl")
        names.append(name)

    # 3. include files
    rel = lambda p: p.relative_to(ROOT / "model").as_posix()
    sc = f'scale="{s} {s} {s}"'
    assets = [f'  <mesh name="blade_visual" file="{rel(vis / (stl.stem + "_body.stl"))}" {sc}/>',
              f'  <mesh name="pdms_visual" file="{rel(vis / (stl.stem + "_pdms.stl"))}" {sc}/>']
    assets += [f'  <mesh name="{n}" file="{rel(out / (n + ".stl"))}"/>' for n in names]
    (ROOT / "assembly/blade_assets.xml").write_text(header + "<mujoco>\n" + "\n".join(assets) + "\n</mujoco>\n")

    for side in ("left", "right"):
        geoms = [f'  <geom name="{side}_blade_{i:02d}" class="blade_collision" mesh="{n}"/>'
                 for i, n in enumerate(names)]
        (ROOT / f"assembly/blade_geoms_{side}.xml").write_text(
            header + "<mujoco>\n" + "\n".join(geoms) + "\n</mujoco>\n")
    print(f"PDMS split: {in_pdms.sum()} triangles; {len(names)} convex pieces, {cut} cut back to the plate")


def flex_lame_inputs(E, nu):
    """young/poisson to give a MuJoCo 3.17 dof="full" flex so that it behaves as linear elasticity with (E, nu).

    Its StVK metric (user_flexcomp.cc, MetricTensor) weights (tr E)^2 by lambda instead of lambda/2, i.e. the
    flex has (mu, 2 lambda). Keep mu = E/(2(1+nu)) and halve lambda: nu_in/(1 - 2 nu_in) = nu/(2(1 - 2 nu)).
    """
    G = E / (2 * (1 + nu))
    nu_in = nu / (2 * (1 - nu))
    return 2 * G * (1 + nu_in), nu_in


def write_pdms(args, lo, hi, z_face, z_plate, header):
    """assembly/pdms_{left,right}.xml: the PDMS layer as a deformable body, bonded to the blade at its back face.

    PDMS frame ({side}_pdms, welded to the fingertip, at the middle of the layer): x across the blade width, y
    toward the tip, z the face normal, +z into the layer. The free face is at z = -t/2; the STL extrusion is the
    part of the layer that stands proud of the blade face, the rest of the thickness t sits in a pocket.

    flex   a 3D flexcomp grid (linear tetrahedra; MuJoCo's elasticity is Saint Venant-Kirchhoff: Hooke's law on
           the Green strain, i.e. linear elasticity at these micro-strains), vertices of the back face pinned to
           the PDMS body, Young's modulus E, Poisson's ratio nu, Rayleigh damping tau (0: none). MuJoCo 3.17's
           dof="full" StVK energy is mu tr(E^2) + lambda (tr E)^2, twice the standard lambda term (patch tests:
           shear exact, uniaxial strain 1.82x at nu = 0.45), so flex_lame_inputs() passes the young/poisson
           whose doubled lambda and unchanged mu are those of (E, nu).
    cells  nx x ny columns, each a point mass on three slide joints with
               k_n = E (1 - nu) / ((1 + nu) (1 - 2 nu)) * A / t     confined compression
               k_s = E / (2 (1 + nu)) * A / t                       shear
           and damping tau * k (0: none). This is the thin-layer (Winkler) limit of linear elasticity: each
           column deforms under its own traction only.
    """
    s = args.scale
    a, b = (hi - lo) * s
    t = args.thickness
    t_proud = (z_plate - z_face) * s
    cx, cy = (lo + hi) / 2 * s
    cz = z_face * s + t / 2   # middle of the layer (STL z grows into the blade)
    E, nu, rho = args.young, args.poisson, args.density
    M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    flex = args.pdms_model == "flex"
    solref = args.solref or ("0.0001 1" if flex else "0.003 1")
    solimp = args.solimp or ("0.99 0.99 0.001" if flex else "0.9 0.95 0.002")
    for side in ("left", "right"):
        n = f"{side}_pdms"
        ref = f"""  <body name="{n}" pos="{cx:.5f} {cy:.5f} {cz:.5f}" childclass="pdms">
    <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>
    <geom name="{n}_visual" type="mesh" mesh="pdms_visual" pos="{-cx:.5f} {-cy:.5f} {-cz:.5f}"
          contype="0" conaffinity="0" group="2" material="pdms_mat"/>
    <geom name="{n}" type="box" size="{a / 2:.5f} {b / 2:.5f} {t / 2:.5f}" contype="0" conaffinity="0"
          group="5" rgba="0.95 0.5 0.1 0.3"/>
    <site name="{n}_surface" type="box" pos="0 0 {-t / 2:.5f}" size="{a / 2:.5f} {b / 2:.5f} 0.0001" group="4"/>"""
        if flex:
            vx, vy, vz = (int(v) for v in args.flex_grid.lower().split("x"))
            E_in, nu_in = flex_lame_inputs(E, nu)
            body = f"""{ref}
    <flexcomp name="{n}_flex" type="grid" dim="3" count="{vx} {vy} {vz}"
              spacing="{a / (vx - 1):.7f} {b / (vy - 1):.7f} {t / (vz - 1):.7f}" radius="0"
              mass="{rho * a * b * t:.6g}" group="3" rgba="0.95 0.5 0.1 0.6">
      <!-- E = {E:g} Pa, nu = {nu:g}, entered as young/poisson that give MuJoCo's doubled lambda its intended value -->
      <elasticity young="{E_in:.6g}" poisson="{nu_in:.6f}" damping="{args.tau:g}"/>
      <contact contype="2" conaffinity="2" condim="3" priority="2" friction="{args.mu:g} 0.005 0.0001"
               solref="{solref}" solimp="{solimp}" selfcollide="none"/>
      <pin gridrange="0 0 {vz - 1} {vx - 1} {vy - 1} {vz - 1}"/>
    </flexcomp>
  </body>"""
            what = (f"a 3D finite-element flex of {vx} x {vy} x {vz} vertices (linear tetrahedra, Saint Venant-Kirchhoff\n"
                    f"       elasticity = Hooke's law at these strains, young/poisson remapped for MuJoCo's doubled lambda;\n"
                    f"       Rayleigh damping {args.tau:g} s). The {vx * vy} vertices of the\n"
                    f"       back face are pinned to {n} (bonded); the free face (z = -t/2) and the sides are free.\n"
                    f"       Contact: the flex's own surface, contype/conaffinity 2 (objects need conaffinity bit 2), priority 2,\n"
                    f"       friction {args.mu:g}, solref {solref} below the 0.5 ms step (refsafe disabled in the option) and solimp\n"
                    f"       {solimp}: a stiff contact, so the deformation is the PDMS's own. {n}_flex vertex positions\n"
                    f"       relative to {n} are the layer's displacement field.")
            summary = f"flex {vx} x {vy} x {vz} vertices"
        else:
            nx, ny = (int(v) for v in args.cells.lower().split("x"))
            dx, dy = a / nx, b / ny
            area = dx * dy
            kn, ks = M * area / t, G * area / t
            mass = rho * area * t
            contact = (f'priority="2" condim="3" friction="{args.mu:g} 0.005 0.0001" solref="{solref}" '
                       f'solimp="{solimp} 0.5 2" contype="2" conaffinity="1"')
            damp = lambda k: f' damping="{args.tau * k:.6g}"' if args.tau > 0 else ""
            cells = []
            for j in range(ny):
                for i in range(nx):
                    c = f"{n}_{i}_{j}"
                    x, y = cx + (i + 0.5) * dx - a / 2, cy + (j + 0.5) * dy - b / 2
                    cells.append(f"""  <body name="{c}" pos="{x:.5f} {y:.5f} {cz:.5f}" childclass="pdms">
    <joint name="{c}_x" axis="1 0 0" stiffness="{ks:.6g}"{damp(ks)}/>
    <joint name="{c}_y" axis="0 1 0" stiffness="{ks:.6g}"{damp(ks)}/>
    <joint name="{c}_z" axis="0 0 1" stiffness="{kn:.6g}"{damp(kn)}/>
    <geom name="{c}" type="ellipsoid" pos="0 0 {(t_proud - t) / 2:.5f}" size="{dx / 2:.5f} {dy / 2:.5f} {t_proud / 2:.5f}"
          mass="{mass:.6g}" {contact}
          group="3" rgba="0.95 0.5 0.1 0.5"/>
  </body>""")
            body = f"{ref}\n  </body>\n" + "\n".join(cells)
            what = (f"{nx} x {ny} cells ({dx * 1e3:.2f} x {dy * 1e3:.2f} mm): each a column of PDMS on three slide joints in the\n"
                    f"       PDMS frame, k_n = E(1-nu)/((1+nu)(1-2nu)) A/t = {kn:.4g} N/m, k_s = E/(2(1+nu)) A/t = {ks:.4g} N/m,\n"
                    f"       damping {args.tau:g} s x stiffness (thin-layer / Winkler limit: each column deforms under its own\n"
                    f"       traction only). The joint positions {n}_i_j_x/y/z are the layer's deformation (m). Contact: one\n"
                    f"       point per cell (a flat ellipsoid filling the cell's protruding part), priority 2, friction {args.mu:g}, solref {solref},\n"
                    f"       solimp {solimp}: the compliance of the object's surface, not of the PDMS.")
            summary = f"{nx} x {ny} cells: k_n {kn:.4g} N/m, k_s {ks:.4g} N/m per cell"
        (ROOT / f"assembly/pdms_{side}.xml").write_text(f"""{header}<mujoco>
  <!-- PDMS layer {a * 1e3:.1f} x {b * 1e3:.1f} x {t * 1e3:.2f} mm on the blade's flat face ({t_proud * 1e3:.2f} mm proud of it, the rest in
       a pocket), linear elastic (placeholders): E = {E / 1e6:g} MPa, nu = {nu:g}, {rho:g} kg/m^3; bonded at its back face.
       Model: {what}
       {n} is the welded reference body (PDMS frame at mid-thickness: x across the blade, y toward the tip, +z into
       the layer): visual mesh, a non-colliding envelope box for distance queries, and the free-face site. -->
{body}
</mujoco>
""")
    # solver settings that depend on the PDMS model, merged into the <option> of ur10e_2f140.xml by an include
    if flex:
        opt = '<option solver="CG" iterations="1000" tolerance="1e-10">\n    <flag island="disable"/>\n  </option>'
        why = ("CG: the only solver MuJoCo allows for a flex pinned to a moving body. 1000 iterations, tolerance 1e-10: the\n"
               "       stiff flex contact needs a converged solve, otherwise transient overloads invert the layer (the StVK\n"
               "       edge-length energy has no inversion barrier). Islands disabled: MuJoCo 3.17's island constraint solver\n"
               "       leaves the stiffness of a flex with articulated attachments out of its metric (engine_metric.c,\n"
               "       flexMatrixFree), and the layer diverges within a few steps.")
    else:
        opt = '<option solver="Newton"/>'
        why = ("Newton: converges the ~250 contact rows of a grasp in a few iterations; CG at its 100-iteration cap left up\n"
               "       to 10 N of force imbalance on a grasped book.")
    (ROOT / "assembly/pdms_option.xml").write_text(f"""{header}<mujoco>
  <!-- Solver settings for the PDMS model ({args.pdms_model}), merged into the <option> of ur10e_2f140.xml.
       {why} -->
  {opt}
</mujoco>
""")
    print(f"PDMS {a * 1e3:.1f} x {b * 1e3:.1f} x {t * 1e3:.2f} mm ({t_proud * 1e3:.2f} mm proud), {summary} "
          f"(face z = {z_face:.2f}, plate z = {z_plate:.2f} in STL units)")


if __name__ == "__main__":
    main()
