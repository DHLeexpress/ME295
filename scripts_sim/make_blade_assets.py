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
    assembly/pdms_{left,right}.xml                the PDMS layer as a deformable body: a grid of
                                                  cells, each a column of PDMS on three elastic
                                                  slide joints (compression + two shear
                                                  directions), bonded to the blade; one contact
                                                  point per cell (flat ellipsoid)

Usage: python scripts_sim/make_blade_assets.py model/gripper_dohyun_v1.stl [--young 1e6 ...]
       python scripts_sim/make_blade_assets.py model/gripper_dohyun_v1.stl --pdms-only --cells 5x7
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
    # PDMS layer (placeholders until the material is chosen: Sylgard-184-like)
    ap.add_argument("--young", type=float, default=1.0e6, help="PDMS Young's modulus (Pa)")
    ap.add_argument("--poisson", type=float, default=0.45, help="PDMS Poisson's ratio (real PDMS: ~0.49)")
    ap.add_argument("--tau", type=float, default=1e-3,
                    help="Kelvin-Voigt time constant of the layer (s): damping = tau * stiffness")
    ap.add_argument("--density", type=float, default=965.0, help="PDMS density (kg/m^3)")
    ap.add_argument("--cells", default="5x7", help="cell grid across x (blade width) x along y (finger)")
    # contact on the free face
    ap.add_argument("--mu", type=float, default=1.0, help="PDMS sliding friction")
    ap.add_argument("--solref", default="0.005 1",
                    help="contact solref of each cell (time constant, damping ratio): the compliance of the "
                         "object's surface (e.g. a book cover), not of the PDMS")
    ap.add_argument("--solimp", default="0.9 0.95 0.002", help="contact solimp of each cell (dmin dmax width)")
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


def write_pdms(args, lo, hi, z_face, z_plate, header):
    """assembly/pdms_{left,right}.xml: the PDMS layer as a deformable body (lumped thin layer).

    The layer (a x b x t) is split into nx x ny cells. Each cell is a column of PDMS bonded to the
    blade at its back face: a point mass on three slide joints in the PDMS frame (x across the blade,
    y toward the tip, z the face normal, +z into the layer) with linear-elastic stiffness
        k_n = E (1 - nu) / ((1 + nu) (1 - 2 nu)) * A / t     confined compression
        k_s = E / (2 (1 + nu)) * A / t                       shear
    and Kelvin-Voigt damping tau * k. For t much smaller than the cell size this is the thin-layer limit
    of linear elasticity (a Winkler foundation): each column deforms under its own traction only.
    """
    s = args.scale
    nx, ny = (int(v) for v in args.cells.lower().split("x"))
    a, b = (hi - lo) * s
    t = (z_plate - z_face) * s
    cx, cy = (lo + hi) / 2 * s
    cz = (z_face + z_plate) / 2 * s
    dx, dy = a / nx, b / ny
    area = dx * dy
    E, nu = args.young, args.poisson
    kn = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)) * area / t
    ks = E / (2 * (1 + nu)) * area / t
    mass = args.density * area * t
    contact = (f'priority="2" condim="3" friction="{args.mu:g} 0.005 0.0001" solref="{args.solref}" '
               f'solimp="{args.solimp} 0.5 2" contype="2" conaffinity="1"')
    for side in ("left", "right"):
        n = f"{side}_pdms"
        cells = []
        for j in range(ny):
            for i in range(nx):
                c = f"{n}_{i}_{j}"
                x, y = cx + (i + 0.5) * dx - a / 2, cy + (j + 0.5) * dy - b / 2
                cells.append(f"""  <body name="{c}" pos="{x:.5f} {y:.5f} {cz:.5f}" childclass="pdms_cell">
    <joint name="{c}_x" axis="1 0 0" stiffness="{ks:.6g}" damping="{args.tau * ks:.6g}"/>
    <joint name="{c}_y" axis="0 1 0" stiffness="{ks:.6g}" damping="{args.tau * ks:.6g}"/>
    <joint name="{c}_z" axis="0 0 1" stiffness="{kn:.6g}" damping="{args.tau * kn:.6g}"/>
    <geom name="{c}" type="ellipsoid" size="{dx / 2:.5f} {dy / 2:.5f} {t / 2:.5f}" mass="{mass:.6g}" {contact}
          group="3" rgba="0.95 0.5 0.1 0.5"/>
  </body>""")
        (ROOT / f"assembly/pdms_{side}.xml").write_text(f"""{header}<mujoco>
  <!-- PDMS layer {a * 1e3:.1f} x {b * 1e3:.1f} x {t * 1e3:.2f} mm on the blade's flat face, as a deformable body of
       {nx} x {ny} cells ({dx * 1e3:.2f} x {dy * 1e3:.2f} mm; i across the blade width x, j along the finger y).
       Material (placeholders): E = {E / 1e6:g} MPa, nu = {nu:g}, {args.density:g} kg/m^3, Kelvin-Voigt time {args.tau * 1e3:g} ms.
       Each cell is a column of PDMS bonded to the blade at its back face: a point mass on three slide joints in
       the PDMS frame (+z into the layer), compression stiffness k_n = E(1-nu)/((1+nu)(1-2nu)) A/t = {kn:.4g} N/m,
       shear stiffness k_s = E/(2(1+nu)) A/t = {ks:.4g} N/m, damping {args.tau * 1e3:g} ms x stiffness. The joint
       positions {n}_i_j_x/y/z are the layer's deformation (m). Thin-layer limit of linear elasticity (Winkler):
       the layer is {t * 1e3:.2f} mm thick, so each column deforms under its own traction only.
       Contact: one point per cell (a flat ellipsoid filling the cell: its apex is the free face and it does not
       reach past the layer), priority 2, friction {args.mu:g}; the
       contact solref/solimp stand for the compliance of the object's surface, not of the PDMS.
       {n} is the welded reference body (PDMS frame): visual mesh, a non-colliding envelope box for distance
       queries, and the surface site. -->
  <body name="{n}" pos="{cx:.5f} {cy:.5f} {cz:.5f}">
    <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>
    <geom name="{n}_visual" type="mesh" mesh="pdms_visual" pos="{-cx:.5f} {-cy:.5f} {-cz:.5f}"
          contype="0" conaffinity="0" group="2" material="pdms_mat"/>
    <geom name="{n}" type="box" size="{a / 2:.5f} {b / 2:.5f} {t / 2:.5f}" contype="0" conaffinity="0"
          group="5" rgba="0.95 0.5 0.1 0.3"/>
    <site name="{n}_surface" type="box" pos="0 0 {-t / 2:.5f}" size="{a / 2:.5f} {b / 2:.5f} 0.0001" group="4"/>
  </body>
""" + "\n".join(cells) + "\n</mujoco>\n")
    print(f"PDMS {a * 1e3:.1f} x {b * 1e3:.1f} x {t * 1e3:.2f} mm, {nx} x {ny} cells: k_n {kn:.4g} N/m, "
          f"k_s {ks:.4g} N/m per cell (face z = {z_face:.2f}, plate z = {z_plate:.2f} in STL units)")


if __name__ == "__main__":
    main()
