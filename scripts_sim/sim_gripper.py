"""Load the UR10e + 2F-140 + blade assembly in MuJoCo, check open/close and contact, save snapshots.

    python scripts_sim/sim_gripper.py               # all checks + snapshots -> docs/figures/sim_*.png
    python scripts_sim/sim_gripper.py --no-render   # checks only (no OpenGL needed)

The blade (gripper_dohyun_v1) replaces the AGC fingertip: it is bolted to the inner finger
through the shared lug hole, and objects are held by its thin PDMS layer.

Checks:
  0. kinematic gaps at full closure (q = 0.7 rad): closest blade surfaces and PDMS faces,
     exact from the mesh vertices
  1. open/close sweep: ctrl 0..255 -> gaps; no gripper self-contact
  2. grasp-and-lift: close on a 300 g box between the PDMS faces, lift the tool 10 cm with
     differential IK; report force per part (pdms / blade / link), what the deformable PDMS layer
     carries (stiffness x deformation) and how far the box slipped
  3. squeeze force at the max force setting for 40-100 mm wide boxes (PDMS only)
Rendering is headless on Linux without a display (MUJOCO_GL=egl). The exploded view and the closed section
come from scripts_sim/plot_connection.py.
"""
import argparse
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless Linux; macOS/Windows keep their default
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assembly/scene.xml"
FIG = ROOT / "docs/figures"
BLADE = ROOT / "model/gripper_dohyun_v1_visual/gripper_dohyun_v1"
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
SIDES = ("left", "right")


def collision_geoms(m, prefix):
    return [i for i in range(m.ngeom) if m.geom(i).name.startswith(prefix) and "visual" not in m.geom(i).name]


def blade_mount(m):
    """Blade STL -> fingertip frame translation (m), read from the assembly."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    return np.array(spec.frame("left_blade_mount").pos)


def set_fingers_kinematic(m, d, q):
    """Pose the linkage directly (all mimic joints) without dynamics."""
    for name, k in [("left_driver_joint", 1), ("left_follower_joint", 1), ("left_inner_knuckle_joint", -1),
                    ("right_driver_joint", -1), ("right_follower_joint", 1), ("right_inner_knuckle_joint", -1)]:
        d.qpos[m.joint(name).qposadr[0]] = k * q
    mujoco.mj_forward(m, d)


def gaps(m, d):
    """Closest distances (m) between the left and right blade pieces and PDMS boxes (collision geometry)."""
    dist = lambda a, b: mujoco.mj_geomDistance(m, d, a, b, 0.5, None)
    lb, rb = collision_geoms(m, "left_blade_"), collision_geoms(m, "right_blade_")
    return {"blade": min(dist(a, b) for a in lb for b in rb),
            "pdms": dist(m.geom("left_pdms").id, m.geom("right_pdms").id)}


def exact_gaps(m, d):
    """Gaps (m) along the jaw axis from the true mesh vertices (no convex-hull padding)."""
    t = blade_mount(m)
    fingertip = {s: m.body(f"{s}_fingertip").id for s in SIDES}
    left, right = (d.xpos[fingertip[s]] for s in SIDES)
    jaw = (right - left) / np.linalg.norm(right - left)
    mid = 0.5 * (left + right)
    out = {}
    for part in ("body", "pdms"):
        v = trimesh.load(f"{BLADE}_{part}.stl").vertices * 1e-3 + t
        proj = {s: ((d.xmat[fingertip[s]].reshape(3, 3) @ v.T).T + d.xpos[fingertip[s]] - mid) @ jaw for s in SIDES}
        out["blade" if part == "body" else "pdms"] = proj["right"].min() - proj["left"].max()
    return out


def fmt_gaps(g):
    return "  ".join(f"{k} {v * 1e3:6.1f}" for k, v in g.items()) + " mm"


def reset(m, d):
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.ctrl[:6] = HOME
    d.ctrl[m.actuator("fingers").id] = 0
    mujoco.mj_forward(m, d)


def load(box_half=(0.025, 0.04, 0.03)):
    """Compile the scene with a given test-box size (half extents, m) and a pedestal narrower
    than the box, so the blades can straddle it. Sizes go through MjSpec because MuJoCo's
    broadphase bounds are fixed at compile time."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.geom("box").size = list(box_half)
    spec.geom("table_top").size = [0.6 * box_half[0], 0.6 * box_half[1], 0.15]
    m = spec.compile()
    return m, mujoco.MjData(m)


def place_box_between_pdms(m, d):
    """Stand the box on the pedestal, centred between the two PDMS faces where they will meet it.
    The fingertips swing in an arc (about 24 mm toward the tip from open to closed), so the
    PDMS centre is taken at the opening where the PDMS gap equals the box width."""
    reset(m, d)
    park_box(m, d)
    step(m, d, 0.5)  # let the arm settle under the gripper's weight first
    arm_q, arm_ctrl = d.qpos[:6].copy(), d.ctrl[:6].copy()
    width = 2 * m.geom("box").size[0]
    lo, hi = 0.0, 0.7
    for _ in range(30):  # PDMS gap decreases with q
        q = 0.5 * (lo + hi)
        set_fingers_kinematic(m, d, q)
        lo, hi = (q, hi) if gaps(m, d)["pdms"] > width else (lo, q)
    left, right = d.site("left_pdms_surface").xpos.copy(), d.site("right_pdms_surface").xpos.copy()
    reset(m, d)
    d.qpos[:6], d.ctrl[:6] = arm_q, arm_ctrl
    mid = 0.5 * (left + right)
    jaw = right - left
    yaw = np.arctan2(jaw[1], jaw[0])  # box x spans the jaw direction
    quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    box_q = m.joint("box").qposadr[0]
    d.qpos[box_q:box_q + 3] = mid
    d.qpos[box_q + 3:box_q + 7] = quat
    table, top = m.body("table").id, m.geom("table_top").id
    m.body_pos[table] = [mid[0], mid[1], mid[2] - m.geom("box").size[2] - m.geom_size[top][2]]
    m.body_quat[table] = quat
    mujoco.mj_forward(m, d)


def part_of(name):
    for part in ("pdms", "blade"):
        if part in name:
            return part
    return "link"


def finger_forces(m, d, obj_geom):
    """Contact force on the object per side and part: {(side, part): [normal, tangential]} in N."""
    out = {}
    f6 = np.zeros(6)
    for i, c in enumerate(d.contact[:d.ncon]):
        if obj_geom not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == obj_geom else c.geom1
        body = m.body(m.geom_bodyid[other]).name
        side = next((s for s in SIDES if body.startswith(s)), None)
        if side is None:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        key = (side, part_of(m.geom(other).name or body))
        out[key] = out.get(key, np.zeros(2)) + [f6[0], np.linalg.norm(f6[1:3])]
    return out


def pdms_state(m, d):
    """What the deformable PDMS layer carries, per side: normal and shear force from its elasticity
    (cell stiffness x deformation, summed over the cells, N) and the largest compression and shear
    displacement of a cell (um). Quasi-statically the force equals the contact force on the layer."""
    out = {}
    for side in SIDES:
        u, f = {}, {}
        for j in range(m.njnt):
            name = m.joint(j).name
            if name.startswith(f"{side}_pdms_"):
                q = d.qpos[m.jnt_qposadr[j]]
                u.setdefault(name[-1], []).append(q)
                f.setdefault(name[-1], []).append(m.jnt_stiffness[j] * q)
        ux, uy = np.array(u["x"]), np.array(u["y"])
        out[side] = {"normal": float(np.sum(f["z"])), "shear": float(np.hypot(np.sum(f["x"]), np.sum(f["y"]))),
                     "compression_um": 1e6 * float(np.max(u["z"])), "shear_um": 1e6 * float(np.hypot(ux, uy).max())}
    return out


def contact_names(m, d):
    nm = lambda g: m.geom(g).name or f"<{m.body(m.geom_bodyid[g]).name}>"
    return sorted({tuple(sorted((nm(c.geom1), nm(c.geom2)))) for c in d.contact[:d.ncon]})


def gripper_ramp(m, d, target, seconds, speed=150.0):
    """Drive the fingers toward ctrl `target` at a Robotiq-like jaw speed (mm/s, both jaws
    together over the 140 mm stroke) and keep stepping for `seconds` in total."""
    a = m.actuator("fingers").id
    rate = 255.0 * speed / 140.0  # ctrl units per second
    for _ in range(int(seconds / m.opt.timestep)):
        delta = target - d.ctrl[a]
        d.ctrl[a] += np.clip(delta, -rate * m.opt.timestep, rate * m.opt.timestep)
        mujoco.mj_step(m, d)


def gripper_self_contacts(m, d):
    gripper = {m.body(b).id for b in range(m.nbody)
               if m.body(b).name.startswith(("left_", "right_", "gripper_base"))}
    return sorted({tuple(sorted((m.geom(c.geom1).name, m.geom(c.geom2).name))) for c in d.contact[:d.ncon]
                   if m.geom_bodyid[c.geom1] in gripper and m.geom_bodyid[c.geom2] in gripper})


def step(m, d, seconds):
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)


def ik_lift(m, d, dz, seconds=1.5):
    """Move the flange straight up by dz, holding orientation (damped least squares)."""
    site = m.site("attachment_site").id
    start = d.site_xpos[site].copy()
    target_rot = d.site_xmat[site].reshape(3, 3).copy()
    n = int(seconds / m.opt.timestep)
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    q_cmd = d.ctrl[:6].copy()
    for k in range(n):
        goal = start + np.array([0, 0, dz * min(1.0, (k + 1) / (0.8 * n))])
        mujoco.mj_jacSite(m, d, jacp, jacr, site)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        R = d.site_xmat[site].reshape(3, 3)
        rot_err = 0.5 * (np.cross(R[:, 0], target_rot[:, 0]) + np.cross(R[:, 1], target_rot[:, 1])
                         + np.cross(R[:, 2], target_rot[:, 2]))
        err = np.concatenate([goal - d.site_xpos[site], rot_err])
        q_cmd += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err) * 50.0 * m.opt.timestep  # 50 /s
        d.ctrl[:6] = q_cmd
        mujoco.mj_step(m, d)


def park_box(m, d):
    box_q = m.joint("box").qposadr[0]
    d.qpos[box_q:box_q + 3] = [5, 5, 0.1]
    m.body_pos[m.body("table").id][2] = -1
    mujoco.mj_forward(m, d)


def check_open_close(m, d):
    print("1) open/close sweep (dynamic, no object)")
    reset(m, d)
    park_box(m, d)
    rows = []
    for ctrl in [0, 64, 128, 192, 255, 0]:
        gripper_ramp(m, d, ctrl, 2.0)
        q = d.joint("left_driver_joint").qpos[0]
        g = gaps(m, d)
        rows.append((ctrl, q, g, gripper_self_contacts(m, d)))
        print(f"   ctrl {ctrl:3d}: q = {q:.3f} rad   {fmt_gaps(g)}   self-contacts: {rows[-1][3] or 'none'}")
    return rows


def check_grasp(m, d, render_cb=None):
    print("2) grasp-and-lift on a 300 g box between the PDMS faces")
    place_box_between_pdms(m, d)
    box, g = m.body("box").id, m.geom("box").id
    step(m, d, 0.5)  # settle on the pedestal
    gripper_ramp(m, d, 255, 2.5)
    z0 = d.xpos[box][2].copy()
    rel0 = d.xpos[box] - d.site("pinch").xpos
    f = finger_forces(m, d, g)
    print(f"   closed: q = {d.joint('left_driver_joint').qpos[0]:.3f} rad; force by part "
          + ", ".join(f"{s} {p} N={v[0]:.1f}" for (s, p), v in sorted(f.items())))
    ik_lift(m, d, 0.10)
    step(m, d, 1.0)
    f = finger_forces(m, d, g)
    lifted = d.xpos[box][2] - z0
    slip = np.linalg.norm(d.xpos[box] - d.site("pinch").xpos - rel0)
    print(f"   lifted: box rose {lifted * 1e3:.1f} mm, slip relative to the gripper {slip * 1e3:.2f} mm")
    print("   holding: " + " | ".join(f"{s} {p}: N = {v[0]:.1f} N, T = {v[1]:.2f} N" for (s, p), v in sorted(f.items()))
          + f"  (box weight {0.3 * 9.81:.2f} N)")
    for side, r in pdms_state(m, d).items():
        print(f"   {side} PDMS layer: normal {r['normal']:.1f} N, shear {r['shear']:.2f} N (stiffness x deformation); "
              f"max compression {r['compression_um']:.2f} um, max shear displacement {r['shear_um']:.2f} um")
    if render_cb:
        render_cb("grasp")
    return lifted, slip, f


def check_widths(widths=(0.04, 0.06, 0.10)):
    print("3) squeeze force vs object width (max force setting)")
    out = []
    for w in widths:
        m, d = load((w / 2, 0.04, 0.03))
        g = m.geom("box").id
        place_box_between_pdms(m, d)
        step(m, d, 0.3)
        gripper_ramp(m, d, 255, 2.0)
        f = finger_forces(m, d, g)
        pdms = [f.get((s, "pdms"), np.zeros(2))[0] for s in SIDES]
        others = sorted({p for (s, p) in f if p != "pdms"})
        out.append((w, *pdms, others))
        print(f"   {w * 1e3:5.0f} mm: q = {d.joint('left_driver_joint').qpos[0]:.3f}  PDMS N {pdms[0]:6.1f} / "
              f"{pdms[1]:6.1f} N   other parts touching: {others or 'none'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    m, d = load()
    print(f"loaded {SCENE.relative_to(ROOT)}: nq={m.nq} nu={m.nu} ngeom={m.ngeom}")

    print("0) gaps at full closure (q = 0.7 rad, kinematic)")
    reset(m, d)
    set_fingers_kinematic(m, d, 0.7)
    print(f"   exact (mesh vertices): {fmt_gaps(exact_gaps(m, d))}")
    print(f"   collision geometry:    {fmt_gaps(gaps(m, d))}")
    set_fingers_kinematic(m, d, 0.0)
    print(f"   open (q = 0), exact:   {fmt_gaps(exact_gaps(m, d))}")

    renderer = None
    if not args.no_render:
        FIG.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(m, height=900, width=1200)

    def render(lookat, distance, azimuth, elevation, contacts=False, axis=None):
        opt = mujoco.MjvOption()
        if contacts:
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        renderer.update_scene(d, camera=cam, scene_option=opt)
        if axis is not None:  # draw a line segment (e.g. the shared screw axis)
            scn = renderer.scene
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array([0.9, 0.1, 0.1, 1.0], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0012, axis[0], axis[1])
            scn.ngeom += 1
        return renderer.render()

    def save(name, img):
        import matplotlib.image as mpimg
        path = FIG / f"sim_{name}.png"
        mpimg.imsave(path, img)
        print(f"   saved {path.relative_to(ROOT)}")

    def jaw_azimuth():
        jaw = d.site("right_pdms_surface").xpos - d.site("left_pdms_surface").xpos
        return np.degrees(np.arctan2(jaw[1], jaw[0])) + 90  # look at the finger plane face-on

    rows = check_open_close(m, d)

    if renderer:
        reset(m, d)
        park_box(m, d)
        view = d.site("pinch").xpos + [0, 0, -0.02]
        save("assembly_open", render(view, 0.8, jaw_azimuth(), -12))
        gripper_ramp(m, d, 255, 2.0)
        save("assembly_closed", render(view, 0.8, jaw_azimuth(), -12))

        # connection close-up of the left fingertip (fingers open): profile, profile with the blade
        # see-through and the shared screw axis in red, back (finger side), object side
        reset(m, d)
        park_box(m, d)
        tip = m.body("left_fingertip").id
        R, p = d.xmat[tip].reshape(3, 3), d.xpos[tip]
        hole = lambda y: p + R @ np.array([-0.0002, y, 0.01005])
        axis = (hole(-0.03), hole(0.012))
        look = hole(0.015)
        jaw = d.site("right_pdms_surface").xpos - d.site("left_pdms_surface").xpos
        jd = np.degrees(np.arctan2(jaw[1], jaw[0]))
        mat = m.material("blade_mat").id
        profile = render(look, 0.21, jd + 90, -5, axis=axis)
        m.mat_rgba[mat, 3] = 0.25
        seethrough = render(look, 0.21, jd + 90, -5, axis=axis)
        m.mat_rgba[mat, 3] = 1.0
        back = render(look, 0.21, jd, -10)
        front = render(look, 0.21, jd + 300, 15)
        grid = np.concatenate([np.concatenate([profile, seethrough], 1), np.concatenate([back, front], 1)], 0)
        save("connection_closeup", grid[::2, ::2])
        reset(m, d)

    def grasp_snap(tag):
        save(tag, render(d.site("pinch").xpos + [0, 0, -0.08], 0.6, jaw_azimuth() + 25, -15, contacts=True))
    lifted, slip, f = check_grasp(m, d, grasp_snap if renderer else None)

    widths = check_widths()

    ok_close = rows[0][1] < 0.05 and rows[5][1] < 0.05 and rows[4][1] > 0.69 and not any(r[3] for r in rows)
    pdms_only = set(f) == {("left", "pdms"), ("right", "pdms")}
    ok_grasp = lifted > 0.08 and slip < 0.005 and pdms_only and min(v[0] for v in f.values()) > 0.3 * 9.81 / 2
    ok_width = all(min(l, r) > 50 and not others for w, l, r, others in widths)
    print(f"\nopen/close: {'PASS' if ok_close else 'FAIL'}   grasp-and-lift (PDMS only): "
          f"{'PASS' if ok_grasp else 'FAIL'}   widths: {'PASS' if ok_width else 'FAIL'}")


if __name__ == "__main__":
    main()
