"""Figures of the blade-to-finger connection: exploded view and the section at full closure.

    python scripts_sim/plot_connection.py   # -> docs/figures/sim_exploded.png, sim_closed_section.png

Exploded view (fingertip frame): 2F-140 inner finger, blade body and PDMS layer pulled apart
along the shared screw axis, with the axis drawn. Closed section: both fingers at full closure
(q = 0.7 rad) cut through the gripper's mid-plane, with the gaps measured from the mesh vertices.
"""
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless Linux; macOS/Windows keep their default
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import sim_gripper as sim  # noqa: E402

ROOT = sim.ROOT
FIG = sim.FIG
MODEL = ROOT / "model"
FINGER_MESH = MODEL / "robotiq_2f140/visual/robotiq_2f_140_inner_finger.stl"
HOLE = np.array([-0.2, 0.0, 10.05])  # finger screw hole in the fingertip frame (mm); y = bar end face
BAR_END_Y = -1.24
AGC_FACE_Z = -3.9  # where the AGC pad's contact face would be (fingertip frame, mm)
COLORS = {"finger": "#3a3a3a", "blade": "#9aa0a6", "pdms": "#f07f1a"}


def parts_in_fingertip_frame(m):
    """Meshes (mm) of the inner finger, blade body and PDMS in the fingertip frame."""
    t = sim.blade_mount(m) * 1e3
    tip_in_finger = np.array([0, 45.7554015893473, -27.2203446692936])
    finger = trimesh.load(FINGER_MESH)
    finger.apply_scale(1e3)
    finger.apply_translation(-tip_in_finger)
    out = {"finger": finger}
    for part in ("body", "pdms"):
        mesh = trimesh.load(f"{sim.BLADE}_{part}.stl")
        mesh.apply_translation(t)
        out["blade" if part == "body" else "pdms"] = mesh
    return out


def shade(mesh, color, light=(0.4, -0.6, 0.7)):
    light = np.asarray(light) / np.linalg.norm(light)
    rgb = np.array(matplotlib.colors.to_rgb(color))
    k = 0.45 + 0.55 * np.abs(mesh.face_normals @ light)
    return np.clip(rgb[None, :] * k[:, None], 0, 1)


def exploded(m):
    parts = parts_in_fingertip_frame(m)
    pull = {"finger": np.zeros(3), "blade": np.array([0, 45.0, 0]), "pdms": np.array([0, 45.0, -45.0])}
    fig = plt.figure(figsize=(16, 8.5))
    views = [(dict(elev=8, azim=-8), "side view (looking along the fingertip x axis)"),
             (dict(elev=22, azim=-55), "oblique view")]
    for k, (view, title) in enumerate(views):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for name in ("finger", "blade", "pdms"):
            mesh = parts[name]
            tri = mesh.triangles + pull[name]
            ax.add_collection3d(Poly3DCollection(tri, facecolors=shade(mesh, COLORS[name]), edgecolor="none"))
        # shared screw axis through the finger's bar end face and the (pulled) blade lug
        ys = np.array([BAR_END_Y - 25, BAR_END_Y + pull["blade"][1] + 15])
        ax.plot(HOLE[0] + 0 * ys, ys, HOLE[2] + 0 * ys, "--", color="#d62728", lw=2.2, zorder=10)
        for y in (BAR_END_Y, BAR_END_Y + pull["blade"][1]):
            ax.scatter([HOLE[0]], [y], [HOLE[2]], color="#d62728", s=40, depthshade=False, zorder=11)
        ax.set_xlim(-60, 60), ax.set_ylim(-70, 170), ax.set_zlim(-80, 100)
        ax.set_box_aspect((120, 240, 180))
        ax.view_init(**view)
        ax.set_xlabel("x (mm)"), ax.set_ylabel("y: toward fingertip (mm)"), ax.set_zlabel("z: away from object (mm)")
        ax.set_title(title, fontsize=12)
        if k == 0:
            ax.text(0, -65, -62, "2F-140 inner finger\n(distal bar, end face y = -1.2 mm)", color=COLORS["finger"], fontsize=10)
            ax.text(0, 40, 92, "blade gripper_dohyun_v1\n(pulled +45 mm along the screw axis)", color="#555", fontsize=10)
            ax.text(0, 45, -62, "PDMS layer 51 x 71 x 0.5 mm\n(pulled a further 45 mm\ntoward the object)",
                    color=COLORS["pdms"], fontsize=10)
            ax.text(0, -65, -30, "shared M5 screw axis\n(AGC lug hole = blade lug hole)", color="#d62728", fontsize=10)
    fig.suptitle("Exploded view: the blade replaces the AGC fingertip and bolts to the inner finger "
                 "through the shared lug hole", fontsize=13)
    fig.tight_layout()
    path = FIG / "sim_exploded.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def body_mesh_in_base(m, d, body, mesh, offset=np.zeros(3)):
    """Mesh (m, in the body frame after `offset`) expressed in the gripper_base frame (mm)."""
    b, base = m.body(body).id, m.body("gripper_base").id
    Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
    R0, p0 = d.xmat[base].reshape(3, 3), d.xpos[base]
    v = (R0.T @ ((Rb @ (mesh.vertices + offset).T).T + pb - p0).T).T
    return trimesh.Trimesh(v * 1e3, mesh.faces, process=False)


def closed_section(m, d):
    sim.reset(m, d)
    sim.set_fingers_kinematic(m, d, 0.7)
    t = sim.blade_mount(m)
    meshes = []
    for side in sim.SIDES:
        for name, file in [("finger", FINGER_MESH), ("blade", f"{sim.BLADE}_body.stl"), ("pdms", f"{sim.BLADE}_pdms.stl")]:
            mesh = trimesh.load(file)
            if name == "finger":
                meshes.append((name, body_mesh_in_base(m, d, f"{side}_inner_finger", mesh)))
            else:
                mesh.apply_scale(1e-3)
                meshes.append((name, body_mesh_in_base(m, d, f"{side}_fingertip", mesh, t)))
        for link in ("outer_finger", "inner_knuckle", "outer_knuckle"):
            mesh = trimesh.load(MODEL / f"robotiq_2f140/visual/robotiq_2f_140_{link}.stl")
            meshes.append(("link", body_mesh_in_base(m, d, f"{side}_{link}", mesh)))

    fig, ax = plt.subplots(figsize=(9, 12))
    style = {"finger": (COLORS["finger"], 0.9), "blade": (COLORS["blade"], 0.9), "pdms": (COLORS["pdms"], 1.0),
             "link": ("#6b6b6b", 0.35)}
    for name, mesh in meshes:
        sec = mesh.section(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0])  # gripper mid-plane
        if sec is None:
            continue
        color, alpha = style[name]
        for e in sec.entities:
            v = sec.vertices[e.points]
            ax.fill(v[:, 1], v[:, 2], color=color, alpha=alpha, lw=0.6, ec="k" if name != "link" else color)

    # exact gaps along the jaw (base y) from all vertices
    def closest(name):
        l = max((mm.vertices for n, mm in meshes[:len(meshes) // 2] if n == name), key=len)
        r = max((mm.vertices for n, mm in meshes[len(meshes) // 2:] if n == name), key=len)
        return l[np.argmax(l[:, 1])], r[np.argmin(r[:, 1])]
    for name, label in [("blade", "closest blade surfaces"), ("pdms", "PDMS faces")]:
        pl, pr = closest(name)
        z = pl[2] if name == "blade" else 0.5 * (pl[2] + pr[2])
        gap = pr[1] - pl[1]
        ax.annotate("", xy=(pl[1], z), xytext=(pr[1], z), arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.8))
        ax.text(pr[1] + 3, z, f"{label}\n{gap:.1f} mm", color="#d62728", va="center", fontsize=11)
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_aspect("equal")
    ax.set_xlabel("jaw direction (mm)")
    ax.set_ylabel("along the fingers, away from the base (mm)")
    ax.invert_yaxis()
    ax.set_title("Section through the gripper mid-plane at full closure (q = 0.7 rad)\n"
                 "grey: blade  orange: PDMS  dark: inner finger  light: other links", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIG / "sim_closed_section.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m, d = sim.load()
    exploded(m)
    closed_section(m, d)


if __name__ == "__main__":
    main()
