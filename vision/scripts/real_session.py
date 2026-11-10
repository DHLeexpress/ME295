"""Top-down grasping on the real UR10e with the wrist ZED, slow and checked: every point of the arm, gripper and
camera at <= 1 cm/s by default, every move planned and checked before it is sent. The same code rehearses in the
simulator (--sim).

    python vision/scripts/real_session.py status                         # read-only
    python vision/scripts/real_session.py goto-view [--execute]          # slow movej to the view configuration
    python vision/scripts/real_session.py observe                        # objects on the table (read-only)
    python vision/scripts/real_session.py grasp [--target N] [--controller scripted|pbvs|mpc] [--execute]
                                         # grasp, lift 3 cm, hold 2 s, put it back, retreat, back to the view
    --sim: the simulated tabletop, starting from the real robot's current joints (rehearsal of the same sequence)

Without --execute nothing moves: plans and checks are printed. On the real robot the ZED SDK needs the zed group
(sg zed -c "..."); the robot must be in remote control. Limits (three independent layers): the joint-move plan
(vision.motion_check: fastest point <= v_max), the safety filter on every twist (table / workspace barrier,
v_max, w_max) and a hard cap in RTDERobot.speedl; a monitor thread stops everything if the measured flange speed
exceeds 2.5 x v_max or the safety mode leaves NORMAL.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from vision import WristCamera  # noqa: E402
from vision.controllers import CONTROLLERS, run_grasp  # noqa: E402
from vision.controllers.base import rotvec_to_mat, tool_yaw  # noqa: E402
from vision.motion_check import plan_joint_move  # noqa: E402
from vision.safety import ENVELOPE_BOXES, SafetyConfig, SafetyFilter  # noqa: E402
from vision.grippers import profile as gripper_profile  # noqa: E402
from vision.tabletop import TabletopParams, draw_tabletop, plan_topdown_grasp, segment_objects  # noqa: E402

HOST = "192.168.0.2"
Z_TABLE = 0.0065            # ChArUco board top in the calibration (base frame)
# view configuration reachable from the robot's 2026-11-09 pose with a short, checked move: the wrist-ZED ray through
# pixel (0.5, 0.22) is vertical and hits (0.10, -0.50) from 0.70 m; the workspace +-10 x +-7 cm (objects <= 16 cm)
# projects into the image band the gripper leaves free (margin 4.3 %); elbow up
Q_VIEW_REAL = np.radians([85.457, -75.837, 71.498, 99.143, 91.04, 175.501])


class StopRobot(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- cells
class RealCell:
    sim = False

    def __init__(self, a, control: bool, camera: bool):
        from vision.stream import CalibratedStream, RTDEPoses
        self.a = a
        self.poses = RTDEPoses(a.host)
        self.rob = self.gripper = self.cam = self.stream = None
        self.w_max = a.w_max
        self.trip = None
        self._stop = False
        import rtde_receive
        self._mon = rtde_receive.RTDEReceiveInterface(a.host, 100.0)
        self.max_flange_speed = 0.0
        self._th = threading.Thread(target=self._monitor, daemon=True)
        self._th.start()
        if control:
            from vision.real_robot import RTDERobot
            from shear_agent.robot.real_ur import RobotiqSocket
            self.rob = RTDERobot(a.host, v_cap=a.v_max, w_cap=a.w_max, acc_cap=max(0.05, 2 * a.v_max))
            self.gripper = RobotiqSocket(a.host, speed=0, force=a.force)
        if camera:
            from vision.zed import ZedCamera
            wrist = WristCamera.active()
            self.cam = ZedCamera(serial=wrist.serial or None, depth_mode="NEURAL", max_depth=1.2)
            self.stream = CalibratedStream(self.cam, self.poses, wrist)

    def _monitor(self):
        lim = 2.5 * self.a.v_max
        while not self._stop:
            try:
                v = float(np.linalg.norm(self._mon.getActualTCPSpeed()[:3]))
                sm = self._mon.getSafetyMode()
            except Exception as e:  # noqa: BLE001
                self.trip = f"RTDE monitor lost: {e}"
                return
            self.max_flange_speed = max(self.max_flange_speed, v)
            if v > lim:
                self.trip = f"flange speed {1e3 * v:.1f} mm/s > {1e3 * lim:.0f} mm/s"
            if sm != 1:
                self.trip = f"safety mode {sm} (not NORMAL)"
            time.sleep(0.01)

    def check(self):
        if self.trip:
            self.halt()
            raise StopRobot(self.trip)

    def halt(self):
        if self.rob is not None and self.rob.rc is not None:
            try:
                self.rob.rc.speedStop(0.5)
                self.rob.rc.stopJ(0.5)
            except Exception:  # noqa: BLE001
                pass

    def q(self):
        return np.asarray(self.poses.latest()[2], float)

    def clock(self):
        return time.monotonic()

    def sleep(self, dt):
        time.sleep(dt)

    def grab(self):
        return self.stream.read()

    def movej(self, q, plan):
        """Async moveJ at the planned speed; watched until it arrives (or the monitor trips)."""
        q = np.asarray(q, float)
        self.rob.rc.speedStop(0.5)                 # leave speedL mode first: a moveJ sent during it is not run
        time.sleep(0.3)
        if not self.rob.rc.moveJ(q.tolist(), float(plan["vel"]), float(plan["acc"]), True):
            raise StopRobot("the controller rejected the moveJ")
        t_end = time.monotonic() + 1.5 * plan["duration"] + 10.0
        last = 0.0
        while True:
            self.check()
            err = float(np.abs(self.q() - q).max())
            if err < 2e-3 and float(np.linalg.norm(self._mon.getActualQd())) < 1e-3:
                return
            if time.monotonic() > t_end:
                self.halt()
                raise StopRobot("moveJ did not arrive in time")
            if time.monotonic() - last > 5.0:
                last = time.monotonic()
                print(f"   ... {np.degrees(err):5.1f} deg to go (largest joint)", flush=True)
            time.sleep(0.02)

    def close(self):
        self._stop = True
        if self.rob is not None:
            try:
                self.rob.close()
            except Exception:  # noqa: BLE001
                pass
        if self.gripper is not None:
            self.gripper.close()
        if self.cam is not None:
            self.cam.close()
        self.poses.close()
        try:
            self._mon.disconnect()
        except Exception:  # noqa: BLE001
            pass


class SimCell:
    sim = True

    def __init__(self, a, q_start):
        from vision.sim.tabletop import TabletopEnv, sample_object
        self.a = a
        self.env = TabletopEnv(sample_object(np.random.default_rng(a.seed)), scale=0.5, q_start=q_start)
        self.rob, self.gripper = self.env.rob, self.env.gripper
        self.gripper.set_force(a.force)
        self.trip = None
        self.max_flange_speed = 0.0
        self._prev = None
        self.max_point_speed = 0.0
        self.phase = "start"
        self.phase_max = {}
        sf = SafetyFilter()
        self.rob.callbacks.append(lambda r: self._speed(r, sf))

    def _speed(self, r, sf):
        F = r.flange_pose()
        P = np.vstack([F[:3, 3], sf.tool_points(F)])
        if self._prev is not None:
            sp = np.linalg.norm(P - self._prev, axis=1) / r.dt
            self.max_point_speed = max(self.max_point_speed, float(sp.max()))
            self.phase_max[self.phase] = max(self.phase_max.get(self.phase, 0.0), float(sp.max()))
            self.max_flange_speed = max(self.max_flange_speed, float(sp[0]))
        self._prev = P

    def check(self):
        pass

    def halt(self):
        self.rob.stopj(0.5)

    def q(self):
        return np.asarray(self.rob.getj(), float)

    def clock(self):
        return self.rob.time

    def sleep(self, dt):
        self.rob.sleep(dt)

    def grab(self):
        f = self.env.grab()
        f.meta = {**(f.meta or {}), "T_base_flange": self.rob.flange_pose()}
        return f

    def movej(self, q, plan):
        self.rob.movej(list(q), acc=plan["acc"], vel=plan["vel"], wait=False)
        t_end = self.rob.time + 1.5 * plan["duration"] + 10.0
        self.rob.sleep(0.05)
        while self.rob.is_program_running():
            if self.rob.time > t_end:
                raise StopRobot("moveJ did not arrive in time")
            self.rob.sleep(0.05)
        self.rob.sleep(0.3)

    def close(self):
        self.env.close()


# ----------------------------------------------------------------------------------------------- steps
def goto(cell, q_target, a, label="view configuration"):
    q0 = cell.q()
    plan = plan_joint_move(q0, q_target, v_max=0.95 * a.v_max, z_table=Z_TABLE)   # 5 % below: tracking overshoot
    print(f"move to the {label}: largest joint move {np.degrees(plan.get('lead_rad', 0)):.1f} deg, fastest point "
          f"travels {plan.get('path_len_max_m', 0):.2f} m -> {plan['duration']:.0f} s at <= {1e3 * a.v_max:.0f} mm/s "
          f"(leading joint {np.degrees(plan['vel']):.2f} deg/s); table: {plan.get('table')}; self-collision: "
          f"{'ok' if plan.get('self_ok', True) else 'RISK'} (closest pair {1e3 * plan.get('min_self_dist', np.inf):.0f} mm)")
    print(f"   from {np.round(np.degrees(q0), 1).tolist()}\n   to   {np.round(np.degrees(q_target), 1).tolist()}")
    if not plan["ok"]:
        raise StopRobot(f"move rejected: {plan['msg']}")
    if plan["duration"] == 0.0:
        return
    if not a.execute:
        print("   dry run: not moving (add --execute)")
        return
    t0 = time.time()
    cell.movej(q_target, plan)
    print(f"   arrived in {time.time() - t0:.0f} s (max flange speed seen {1e3 * cell.max_flange_speed:.1f} mm/s)")


def params_for(a, cell, z_guess):
    prof = gripper_profile(a.gripper)
    return TabletopParams(z_table_guess=z_guess, envelope=prof.envelope, x_range=(-0.35, 0.60), y_range=(-0.95, -0.15)), prof


def envelope_check(frame, tip_z):
    """Does the real gripper fit the model's envelope? Its own points (near the tool axis, below the flange) in the
    flange frame: the lowest point along the tool must not pass the envelope's tips (0.338 m)."""
    T = frame.meta.get("T_base_flange") if frame.meta else None
    if T is None:
        return None
    pts, _ = frame.points_base(stride=2)
    Ti = np.linalg.inv(T)
    pf = pts @ Ti[:3, :3].T + Ti[:3, 3]
    near = (np.abs(pf[:, 0]) < 0.12) & (np.abs(pf[:, 1]) < 0.07) & (pf[:, 2] > 0.15) & (pf[:, 2] < 0.45)
    if near.sum() < 50:
        return {"points": int(near.sum())}
    z = np.percentile(pf[near, 2], [50, 99])
    return {"points": int(near.sum()), "z50": float(z[0]), "z99": float(z[1]), "fits": bool(z[1] <= tip_z + 0.005)}


def observe(cell, a, out: Path):
    import cv2
    f = cell.grab()
    for _ in range(3 if not cell.sim else 0):     # a few frames for NEURAL depth to settle
        f = cell.grab()
    params, prof = params_for(a, cell, Z_TABLE if not cell.sim else 0.0)
    objs, zt = segment_objects(f, f.meta.get("T_base_flange"), params)
    yaw = tool_yaw(f.meta["T_base_flange"][:3, :3])
    grasps = [plan_topdown_grasp(o, Z_TABLE if not cell.sim else 0.0, prefer_yaw=yaw, profile=prof) for o in objs]
    print(f"gripper profile: {prof.name}")
    out.mkdir(parents=True, exist_ok=True)
    img = draw_tabletop(f, objs, None, zt)
    for k, o in enumerate(objs):
        Ti = np.linalg.inv(f.T_base_cam)
        uv = f.intr.project((Ti[:3, :3] @ o.center + Ti[:3, 3])[None])[0]
        cv2.putText(img, str(k), (int(uv[0]) + 6, int(uv[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
    cv2.imwrite(str(out / "observe.jpg"), img[..., ::-1])
    f.save(out / "observe.npz")
    print(f"table plane z = {zt if zt is None else round(zt, 4)} (safety filter uses {Z_TABLE if not cell.sim else 0.0}); "
          f"{len(objs)} object(s); overlay -> {out / 'observe.jpg'}")
    for k, (o, g) in enumerate(zip(objs, grasps)):
        print(f"  [{k}] {o.summary()}\n       -> {g.summary()}")
    env = envelope_check(f, prof.tip_z) if not cell.sim else None
    if env:
        print(f"gripper self-check: {env}")
    return f, objs, zt, grasps, env


def height_map(frame, params, cell_m=0.01):
    pts, _ = frame.points_base(stride=2)
    x0, x1 = params.x_range
    y0, y1 = params.y_range
    sel = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1) & (pts[:, 2] < 0.6)
    p = pts[sel]
    ij = ((p[:, :2] - [x0, y0]) / cell_m).astype(int)
    H = np.full((int((x1 - x0) / cell_m) + 1, int((y1 - y0) / cell_m) + 1), np.nan)
    order = np.argsort(p[:, 2])
    H[ij[order, 0], ij[order, 1]] = p[order, 2]          # highest point per cell
    return H


def scene_clear(cell, a, wait=1.5):
    """Refuse to move when the workspace is not still (someone placing objects) or something reaches into the
    camera's view from outside (an arm): two looks `wait` s apart must agree, and no segment taller than 25 cm may
    touch the image border."""
    params, _ = params_for(a, cell, Z_TABLE)
    f1 = cell.grab()
    t_end = time.time() + wait
    while time.time() < t_end:
        f2 = cell.grab()
    H1, H2 = height_map(f1, params), height_map(f2, params)
    both = np.isfinite(H1) & np.isfinite(H2)
    changed = float(np.mean(np.abs(H1[both] - H2[both]) > 0.02)) if both.any() else 1.0
    objs, _ = segment_objects(f2, f2.meta.get("T_base_flange"), params)
    Ti = np.linalg.inv(f2.T_base_cam)
    border = []
    for o in objs:
        uv = f2.intr.project(o.points @ Ti[:3, :3].T + Ti[:3, 3])
        edge = (uv[:, 0] < 8) | (uv[:, 0] > f2.intr.width - 8) | (uv[:, 1] < 8) | (uv[:, 1] > f2.intr.height - 8)
        if o.height > 0.25 or (edge.mean() > 0.01 and o.height > 0.15):
            border.append(o.summary())
    if changed > 0.01:
        return False, f"the workspace changed between two looks ({100 * changed:.1f} % of cells moved > 2 cm): someone working there?"
    if border:
        return False, f"something tall reaches into the view from outside (an arm?): {border[0]}"
    return True, "workspace still, nothing reaching in"


def speed_until(cell, safety, v6, done, timeout, period=0.02):
    t_end = cell.clock() + timeout
    while cell.clock() < t_end:
        cell.check()
        if done():
            break
        T = cell.rob.flange_pose()
        v, _ = safety.filter_twist(T, np.asarray(cell.rob.getl()[:3]), v6)
        cell.rob.speedl(list(v), acc=0.05, min_time=2 * period)
        cell.sleep(period)
    cell.rob.speedl([0, 0, 0, 0, 0, 0], acc=0.05, min_time=period)
    cell.sleep(0.3)


# grip force: two pads, friction on both, lifting at <= 1 cm/s: F >= SF * m g / (2 mu), at least the 2F-140's minimum
FORCE_RANGE = (10.0, 125.0)        # N per finger over the Robotiq force register 0..255 (2F-140 data sheet)
OBJECTS = {                        # name: (mass kg, friction mu, max force N for fragile things, note)
    "apple": (0.20, 0.6, 30.0, "fruit: firm grip would bruise"),
    "bottle": (0.55, 0.35, 40.0, "PET bottle, assume full (0.5 L)"),
    "cylinder": (0.15, 0.4, 25.0, "thin-walled plastic container"),
    "cone": (0.05, 0.4, 25.0, "3D-printed PLA"),
}
SAFETY_FACTOR = 2.0


def grip_force(name: str):
    m, mu, fmax, note = OBJECTS.get(name, (0.3, 0.4, 40.0, "unknown object"))
    need = m * 9.81 / (2 * mu)
    F = float(np.clip(SAFETY_FACTOR * need, FORCE_RANGE[0], fmax))
    reg = int(round(255 * (F - FORCE_RANGE[0]) / (FORCE_RANGE[1] - FORCE_RANGE[0])))
    return F, reg, f"{name}: {F:.0f} N/finger (register {reg}); m~{m:.2f} kg, mu~{mu}, x{SAFETY_FACTOR:g} -> needs {need:.1f} N ({note})"


class Recorder:
    """Wrist-ZED video at a fixed rate with a HUD; frames arrive through CalibratedStream.listeners."""

    def __init__(self, path: Path, state: dict, fps: float = 15.0, size=(960, 540)):
        import imageio_ffmpeg
        self.path, self.state, self.fps, self.size = path, state, fps, size
        self.latest = None
        self.w = imageio_ffmpeg.write_frames(str(path), size, fps=fps, quality=7, macro_block_size=1)
        self.w.send(None)
        self.n = 0
        self._stop = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def on_frame(self, f):
        self.latest = f.rgb

    def _run(self):
        import cv2
        t_next = time.time()
        while not self._stop:
            if self.latest is not None:
                img = cv2.resize(self.latest, self.size)
                st = self.state
                lines = [st.get("title", ""), f"phase: {st.get('phase', '')}   TCP z {st.get('z', float('nan')):.3f} m   "
                         f"gripper pos {st.get('gpos', '-')} obj {st.get('gobj', '-')}   t {time.time() - st.get('t0', time.time()):5.1f} s",
                         st.get("force", "")]
                cv2.rectangle(img, (0, 0), (self.size[0], 74), (0, 0, 0), -1)
                for k, l in enumerate(lines):
                    cv2.putText(img, l[:110], (8, 20 + 23 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                self.w.send(np.ascontiguousarray(img))
                self.n += 1
            t_next += 1.0 / self.fps
            time.sleep(max(0.0, t_next - time.time()))

    def close(self):
        self._stop = True
        self.th.join(timeout=2.0)
        self.w.close()


def pick(cell, a, out: Path, name: str, xy, k_pick=1, n_picks=1):
    """Look, choose the object nearest to xy, check, grasp with its grip force, lift, hold, put it back, retreat,
    return to the view configuration. Real robot: records the wrist-ZED video out/<name>.mp4."""
    out.mkdir(parents=True, exist_ok=True)
    F, reg, why = grip_force(name)
    print(f"\n=== pick {k_pick}/{n_picks}: {name} near {np.round(xy, 3).tolist()} ===\n{why}")
    f, objs, zt, grasps, env = observe(cell, a, out / name)
    if not objs:
        raise StopRobot("no object on the table")
    if env and env.get("fits") is False:
        raise StopRobot(f"the real gripper reaches further than the profile's envelope: {env}")
    k = int(np.argmin([np.linalg.norm(o.center[:2] - np.asarray(xy)) for o in objs]))
    if np.linalg.norm(objs[k].center[:2] - np.asarray(xy)) > 0.04:
        raise StopRobot(f"no object within 4 cm of {xy}")
    g0 = grasps[k]
    print(f"target [{k}]: {objs[k].summary()}\n   {g0.summary()}")
    if not g0.ok:
        raise StopRobot(f"target not graspable: {g0.reason}")
    prof = gripper_profile(a.gripper)
    z_tab = Z_TABLE if not cell.sim else 0.0
    if zt is not None and zt > z_tab:
        z_tab = float(zt)
    safety = SafetyFilter(SafetyConfig(z_table=z_tab, v_max=a.v_max, w_max=a.w_max, point_v_max=a.v_max,
                                       tool_points=prof.tool_points()))
    from shear_agent.robot import SimRobot
    sim = SimRobot(verbose=False)
    T_tcp = np.eye(4)
    T_tcp[2, 3] = prof.tcp_z(g0.width)
    lift = g0.T_tcp.copy()
    lift[2, 3] += a.lift
    for nm, T in (("pre-grasp", g0.T_pre), ("grasp", g0.T_tcp), ("lift", lift)):
        qq, err = sim.ik(T @ np.linalg.inv(T_tcp), cell.q(), iters=300)
        r = safety.check_q(qq, T_tcp) if err < 1e-5 else None
        print(f"  check {nm:9s}: {'OK' if r and r.safe else 'FAIL'} {r.message if r else 'no IK'}")
        if not (r and r.safe):
            raise StopRobot(f"{nm} fails the checks")
    if not cell.sim:
        ok, why_s = scene_clear(cell, a)
        print(f"  scene check: {why_s}")
        if not ok and a.execute:
            raise StopRobot(f"not moving: {why_s}")
    if not a.execute:
        print("  dry run: not grasping (add --execute)")
        return None
    state = {"title": f"pick {k_pick}/{n_picks}: {name}   (UR10e + Robotiq 2F-140, wrist ZED Mini, <= {1e3 * a.v_max:.0f} mm/s)",
             "force": f"grip force {why}", "phase": "start", "t0": time.time()}
    rec = None
    if not cell.sim:
        st = cell.gripper._get("STA")
        if st != 3:
            print("  activating the gripper (its fingers open and close once) ...")
            cell.gripper.activate(timeout=15.0)
        cell.gripper.set_speed(0)
        cell.gripper.set_force(reg)
        rec = Recorder(out / f"{k_pick:02d}_{name}.mp4", state)
        cell.stream.listeners.append(rec.on_frame)
    else:
        cell.gripper.set_force(reg)
    v = a.v_max
    kw = {"scripted": {"v_approach": 0.9 * v, "v_descend": 0.7 * v, "acc": max(0.02, 2 * v), "kp": 1.0},
          "pbvs": {"v_max": 0.9 * v, "v_descend": 0.7 * v, "k": 1.0},
          "mpc": {"v_max": (0.9 * v, 0.9 * v, 0.7 * v), "a_max": 0.02}}[a.controller]
    ctrl = CONTROLLERS[a.controller](prof.register_for_width, force=reg, lift=a.lift, hold=2.0, v_lift=0.8 * v,
                                     k_rot=0.5, w_max=a.w_max, close_timeout=8.0, blocked_abort=6.0, **kw)
    est = None
    if cell.sim:
        stt = {"prev": g0.position[:2].copy()}

        def perceive():
            fr = cell.grab()
            ob, _ = segment_objects(fr, cell.rob.flange_pose(), params_for(a, cell, 0.0)[0])
            if not ob:
                return None
            o = min(ob, key=lambda o: np.linalg.norm(o.center[:2] - stt["prev"]))
            stt["prev"] = o.center[:2].copy()
            return plan_topdown_grasp(o, z_tab, prefer_yaw=tool_yaw(rotvec_to_mat(cell.rob.getl()[3:])),
                                      stamp=cell.rob.time, profile=prof)
    else:
        from vision.real_robot import AsyncGraspEstimator
        est = AsyncGraspEstimator(cell.stream, params_for(a, cell, Z_TABLE)[0], z_table=z_tab, track_xy=g0.position,
                                  yaw_hint=lambda: tool_yaw(rotvec_to_mat(cell.rob.getl()[3:])), profile=prof,
                                  segment_every=3)
        t_wait = time.time() + 5
        while est.latest is None and time.time() < t_wait:
            time.sleep(0.05)
        perceive = est.perceive

    def on_step(obs, cmd):
        cell.check()
        state.update(phase=cmd.phase, z=float(obs.tcp_pose[2]), gpos=obs.gripper_position, gobj=obs.gripper_object)
        if cell.sim:
            cell.phase = f"grasp:{cmd.phase}"
        if est is not None and cmd.phase == "approach" and est.stale(1.0):
            raise StopRobot("camera stream stalled")
    t0 = time.time()
    result = {"name": name, "force_N": F, "register": reg}
    try:
        res = run_grasp(ctrl, cell.rob, cell.gripper, perceive, safety, prof.tcp_pose6, cell.rob.flange_pose,
                        perception_hz=2.0, max_time=a.max_time, on_step=on_step, clock=cell.clock)
        print(f"  grasp: {res['status']} in {res.get('time', 0):.0f} s; barrier interventions "
              f"{res.get('safety_interventions')}; gripper pos {cell.gripper.get_position()} obj {cell.gripper.get_object_status()}")
        result.update({k: v for k, v in res.items() if k != "log"})
        if cell.sim:
            print(f"  sim ground truth: {cell.env.metrics()}")
        goal = ctrl.goal
        if res["status"] == "done" and goal is not None:
            state["phase"] = "put back"
            if cell.sim:
                cell.phase = "put back"
            z_put = goal.position[2] + 0.003
            f0 = np.asarray(cell.rob.get_tcp_force()[:3])
            speed_until(cell, safety, [0, 0, -0.7 * v, 0, 0, 0],
                        lambda: (state.update(z=float(cell.rob.getl()[2])) or cell.rob.getl()[2] <= z_put)
                        or np.linalg.norm(np.asarray(cell.rob.get_tcp_force()[:3]) - f0) > 15.0,
                        timeout=a.lift / (0.7 * v) + 10)
        if goal is not None:
            state["phase"] = "release"
            cell.gripper.set_position(prof.register_for_width(goal.opening))
            t_end = cell.clock() + 8.0
            cell.sleep(0.5)
            while cell.gripper.get_object_status() == 0 and cell.clock() < t_end:
                state.update(gpos=cell.gripper.get_position(), gobj=0)
                cell.sleep(0.1)
            state.update(gpos=cell.gripper.get_position(), gobj=cell.gripper.get_object_status())
            state["phase"] = "retreat"
            if cell.sim:
                cell.phase = "retreat"
            z_up = cell.rob.getl()[2] + 0.10
            speed_until(cell, safety, [0, 0, 0.9 * v, 0, 0, 0],
                        lambda: state.update(z=float(cell.rob.getl()[2])) or cell.rob.getl()[2] >= z_up,
                        timeout=0.10 / (0.9 * v) + 10)
        state["phase"] = "return to the view"
        if cell.sim:
            cell.phase = "return"
        goto(cell, Q_VIEW_REAL, a, "view configuration (return)")
        state["phase"] = "done"
        time.sleep(0.5 if not cell.sim else 0.0)
    finally:
        if est is not None:
            est.close()
        if rec is not None:
            cell.stream.listeners.remove(rec.on_frame)
            rec.close()
            print(f"  video -> {rec.path} ({rec.n} frames)")
            result["video"] = str(rec.path)
    result["wall_s"] = time.time() - t0
    (out / f"{k_pick:02d}_{name}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def grasp(cell, a, out: Path):
    xy = a.target_xy
    if xy is None:
        f, objs, *_ = observe(cell, a, out)
        if not objs:
            raise StopRobot("no object on the table")
        k = a.target if a.target is not None else int(np.argmin([np.linalg.norm(o.center[:2] - [0.10, -0.50]) for o in objs]))
        xy = objs[k].center[:2]
    pick(cell, a, out, a.name or "object", xy)


def concat_videos(paths, out_path):
    import subprocess
    import imageio_ffmpeg
    lst = out_path.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                    str(lst), "-c", "copy", str(out_path)], check=True)
    lst.unlink()


def pick_all(cell, a, out: Path):
    picks = []
    for spec in a.pick:
        nm, x, y = spec.split(":")[:3]
        picks.append((nm, (float(x), float(y))))
    results = []
    for k, (nm, xy) in enumerate(picks, 1):
        try:
            results.append(pick(cell, a, out, nm, xy, k, len(picks)))
        except StopRobot as e:
            print(f"  {nm}: STOPPED: {e}")
            results.append({"name": nm, "status": f"stopped: {e}"})
            if cell.trip:          # a monitor trip ends the session; a planning refusal only skips this object
                raise
            if a.execute and not cell.sim:
                goto(cell, Q_VIEW_REAL, a, "view configuration (after a skipped pick)")
    vids = [r["video"] for r in results if r and r.get("video")]
    if len(vids) > 1:
        concat_videos(vids, out / "all_picks.mp4")
        print(f"combined video -> {out / 'all_picks.mp4'}")
    print("\nsummary:")
    for r in results:
        if r:
            print(f"  {r.get('name')}: {r.get('status')}  force {r.get('force_N', float('nan')):.0f} N (register {r.get('register')})")
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))


def status(a):
    import socket
    s = socket.create_connection((a.host, 29999), timeout=3)
    s.recv(1024)
    for c in ("is in remote control", "robotmode", "safetystatus", "programState"):
        s.sendall((c + "\n").encode())
        print(f"{c:22s}: {s.recv(1024).decode().strip()}")
    s.close()
    from vision.stream import RTDEPoses
    p = RTDEPoses(a.host)
    _, T, q = p.latest()
    mm, deg = p.check_tcp()
    p.close()
    print(f"q (deg) {np.round(np.degrees(q), 1).tolist()}; flange {np.round(T[:3, 3], 3).tolist()}; TCP check {mm:.1f} mm / {deg:.2f} deg")
    print(f"view configuration (deg) {np.round(np.degrees(Q_VIEW_REAL), 1).tolist()}")
    print(f"safety at the current pose: {SafetyFilter(SafetyConfig(z_table=Z_TABLE)).check_q(q).message}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["status", "goto-view", "observe", "grasp", "pick-all", "rehearse"])
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--v-max", type=float, default=0.01, help="m/s, every point (default 1 cm/s)")
    ap.add_argument("--w-max", type=float, default=None, help="rad/s (default: 3 x v-max, i.e. 0.03 at 1 cm/s; every "
                    "point stays under v-max anyway)")
    ap.add_argument("--force", type=int, default=50, help="Robotiq force register (0..255)")
    ap.add_argument("--lift", type=float, default=0.05)
    ap.add_argument("--pick", action="append", default=[], help="pick-all: name:x:y (name in OBJECTS sets the force)")
    ap.add_argument("--name", help="grasp: object name (sets the grip force)")
    ap.add_argument("--controller", choices=sorted(CONTROLLERS), default="scripted")
    ap.add_argument("--target", type=int)
    ap.add_argument("--target-xy", type=float, nargs=2, help="grasp the object nearest to this (x, y) in the base frame")
    ap.add_argument("--gripper", choices=["bare", "blade"], help="finger profile (vision/grippers.py); default: "
                    "bare on the real robot (standard 2F-140 fingertips), blade in the simulator")
    ap.add_argument("--max-time", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--q-start", type=float, nargs=6, help="--sim: start joints (deg); default the real robot's")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "vision" / "real_session" / time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    if a.w_max is None:
        a.w_max = 3.0 * a.v_max
    if a.v_max > 0.05 and not a.sim:
        sys.exit("--v-max above 5 cm/s is not allowed on the real robot from this script")
    if a.step == "status":
        return status(a)
    if a.gripper is None:
        a.gripper = "blade" if (a.sim or a.step == "rehearse") else "bare"
    if a.sim or a.step == "rehearse":
        a.sim = True
        if a.q_start is None:
            from vision.stream import RTDEPoses
            p = RTDEPoses(a.host)
            q0 = np.asarray(p.latest()[2], float)
            p.close()
        else:
            q0 = np.radians(a.q_start)
        cell = SimCell(a, q0)
    else:
        cell = RealCell(a, control=a.execute and a.step in ("goto-view", "grasp", "pick-all"),
                        camera=a.step in ("observe", "grasp", "pick-all"))
    try:
        if a.step == "goto-view":
            goto(cell, Q_VIEW_REAL, a)
        elif a.step == "observe":
            observe(cell, a, a.out)
        elif a.step == "grasp":
            grasp(cell, a, a.out)
        elif a.step == "pick-all":
            pick_all(cell, a, a.out)
        elif a.step == "rehearse":
            a.execute = True
            cell.phase = "goto view"
            goto(cell, Q_VIEW_REAL, a)
            cell.phase = "observe"
            grasp(cell, a, a.out)
    except StopRobot as e:
        print(f"STOPPED: {e}")
        cell.halt()
    finally:
        cell.close()


if __name__ == "__main__":
    main()
