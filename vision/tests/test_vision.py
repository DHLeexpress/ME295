"""Tests for vision/ (offline, simulator only; no camera or robot needed).

    python -m unittest discover -s vision/tests -t .
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", ROOT / "scripts_sim"):
    sys.path.insert(0, str(p))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from vision import WristCamera  # noqa: E402
from vision.camera import Intrinsics  # noqa: E402
from vision.safety import SafetyConfig, SafetyFilter, fk_frames  # noqa: E402
from vision.sim_zed import SimZedCamera, add_wrist_zed, camera_pose  # noqa: E402


class TestCamera(unittest.TestCase):
    def test_deproject_project_roundtrip(self):
        i = Intrinsics(64, 48, 50.0, 51.0, 30.2, 25.1)
        z = np.full((48, 64), 0.8)
        xyz = i.deproject(z)
        uv = i.project(xyz.reshape(-1, 3))
        u, v = np.meshgrid(np.arange(64), np.arange(48))
        self.assertLess(np.abs(uv - np.c_[u.ravel(), v.ravel()]).max(), 1e-4)
        s = i.scaled(0.5)
        self.assertAlmostEqual(s.fx, 25.0)

    def test_mujoco_camera_matches_opencv_projection_and_planar_depth(self):
        """Spheres at known points in the camera frame land where K projects them; depth is planar z."""
        wrist = WristCamera.active(warn=False)
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
        fl = spec.worldbody.add_body()
        fl.name = "flange"
        add_wrist_zed(spec, wrist, housing=False)
        T = wrist.T_flange_cam
        pts_cam = np.array([[0.0, 0.0, 0.6], [0.15, 0.08, 0.6], [-0.2, -0.1, 0.5], [0.1, -0.12, 0.4]])
        for k, p in enumerate(pts_cam):
            g = spec.worldbody.add_geom()
            g.name, g.type, g.size, g.pos = f"s{k}", mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], (T[:3, :3] @ p + T[:3, 3]).tolist()
        wall = spec.worldbody.add_geom()
        wall.type, wall.size = mujoco.mjtGeom.mjGEOM_BOX, [2, 2, 0.001]
        wall.pos = (T[:3, :3] @ [0, 0, 1.0] + T[:3, 3]).tolist()
        wall.quat = [float(v) for v in __import__("vision.sim_zed", fromlist=["_mat2quat"])._mat2quat(T[:3, :3])]
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        cam = SimZedCamera(m)
        f = cam.grab(d, segmentation=True)
        cam.close()
        seg = f.meta["segmentation"][..., 0]
        for k, p in enumerate(pts_cam):
            ys, xs = np.nonzero(seg == m.geom(f"s{k}").id)
            uv = wrist.intr.project(p[None])[0]
            self.assertLess(np.hypot(xs.mean() - uv[0], ys.mean() - uv[1]), 0.6)
        corner = f.xyz[5, 5]
        self.assertAlmostEqual(float(corner[2]), 1.0 - 0.001, places=3)      # planar depth to the wall
        np.testing.assert_allclose(f.T_base_cam, T, atol=1e-6)


class TestCalibration(unittest.TestCase):
    def test_snapshot_matches_bundle(self):
        snap = WristCamera.load()
        live = WristCamera.from_zed_bundle()
        self.assertLess(np.abs(snap.T_flange_cam - live.T_flange_cam).max(), 1e-5)

    def test_sim_camera_pose_is_flange_times_calibration(self):
        from shear_agent.robot import SimRobot
        spec = mujoco.MjSpec.from_file(str(ROOT / "assembly" / "scene.xml"))
        wrist = WristCamera.active(warn=False)
        add_wrist_zed(spec, wrist)
        rob = SimRobot(model=spec.compile(), verbose=False)
        rob.reset([0.3, -1.2, 1.5, -1.6, -1.5, 0.4])
        np.testing.assert_allclose(camera_pose(rob.m, rob.d), rob.flange_pose() @ wrist.T_flange_cam, atol=1e-9)

    def test_dh_fk_matches_simulator(self):
        from shear_agent.robot import SimRobot
        rob = SimRobot(verbose=False)
        q = [1.2, -1.0, 1.4, -2.0, -1.2, 0.7]
        np.testing.assert_allclose(fk_frames(q)[-1], rob._fk_flange(q), atol=1e-9)


class TestSafety(unittest.TestCase):
    def setUp(self):
        from vision.sim.tabletop import Q_VIEW
        self.sf = SafetyFilter(SafetyConfig(z_table=0.0))
        self.q_view = Q_VIEW

    def test_view_configuration_is_safe(self):
        self.assertTrue(self.sf.check_q(self.q_view).safe)

    def test_tool_below_table_is_unsafe(self):
        T = np.eye(4)
        T[:3, :3] = np.diag([1, -1, -1])          # tool pointing down
        T[:3, 3] = [0.1, -0.5, 0.30]              # blade tips 3.8 cm below the table
        self.assertFalse(self.sf.check_pose(T).safe)
        T[2, 3] = 0.40
        self.assertTrue(self.sf.check_pose(T).safe)

    def test_path_into_table_blocked(self):
        q1 = self.q_view.copy()
        q1[1] += 0.9                               # shoulder down: drives the tool into the table
        ok, rep, idx = self.sf.check_path(self.q_view, q1)
        self.assertFalse(ok)
        self.assertGreater(idx, 0)

    def test_barrier_limits_descent_near_table(self):
        T = np.eye(4)
        T[:3, :3] = np.diag([1, -1, -1])
        T[:3, 3] = [0.1, -0.5, 0.36]
        tcp = (T @ np.array([0, 0, 0.29, 1]))[:3]
        lowest = self.sf.tool_points(T)[:, 2].min()
        v, active = self.sf.filter_twist(T, tcp, [0, 0, -0.2, 0, 0, 0])
        self.assertTrue(active)
        self.assertGreaterEqual(v[2], -self.sf.cfg.gamma * (lowest - self.sf.cfg.margin_tool) - 1e-6)
        v2, active2 = self.sf.filter_twist(T, tcp, [0, 0, 0.05, 0, 0, 0])
        self.assertFalse(active2)
        np.testing.assert_allclose(v2, [0, 0, 0.05, 0, 0, 0])


class TestPerception(unittest.TestCase):
    def test_tabletop_object_and_grasp(self):
        from vision.sim.tabletop import TabletopEnv, sample_object
        from vision.tabletop import plan_topdown_grasp, segment_objects
        obj = sample_object(np.random.default_rng(6))
        env = TabletopEnv(obj, depth="exact", scale=0.5)
        try:
            objs, zt = segment_objects(env.grab(), env.rob.flange_pose())
            self.assertEqual(len(objs), 1)
            self.assertLess(abs(zt), 0.003)
            self.assertLess(np.linalg.norm(objs[0].center[:2] - env.p0[:2]), 0.005)
            self.assertLess(abs(objs[0].width - obj.grasp_width), 0.012)
            g = plan_topdown_grasp(objs[0], zt)
            self.assertTrue(g.ok)
            self.assertTrue(SafetyFilter(SafetyConfig(z_table=0.0)).check_pose(
                g.T_tcp @ np.linalg.inv(np.eye(4) + np.diag([0, 0, 0, 0]) * 0) @ np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.29], [0, 0, 0, 1]])).safe)
        finally:
            env.close()

    def test_shelf_nominal_target(self):
        import run_librarian as rl
        from shear_agent.robot import SimRobot, mat_to_pose, pose_to_mat
        from vision.look import look_pose
        from vision.shelf import estimate_shelf
        wrist = WristCamera.active(warn=False)
        spec = mujoco.MjSpec.from_file(str(rl.SCENE))
        add_wrist_zed(spec, wrist)
        m = spec.compile()
        rob = SimRobot(model=m, verbose=False)
        rob.reset(rl.start_q(rob))
        rob.set_tcp(rl.TCP)
        rob.movel(list(mat_to_pose(look_pose(wrist, (0.0, rl.SPINE_Y, 0.53)) @ pose_to_mat(rl.TCP))), acc=0.5, vel=0.25)
        cam = SimZedCamera(m, scale=0.5)
        est = estimate_shelf(cam.grab(rob.d))
        cam.close()
        b = est.select("dark blue")
        self.assertIsNotNone(b)
        self.assertLess(abs(b.center[0] - 0.0), 0.002)
        self.assertLess(abs(b.thickness - 0.040), 0.003)
        self.assertLess(abs(est.spine_y - rl.SPINE_Y), 0.002)

    def test_view_configuration_sees_the_workspace(self):
        from vision.sim.tabletop import Q_VIEW, WORKSPACE_CENTER, WORKSPACE_HALF
        wrist = WristCamera.active(warn=False)
        T = fk_frames(Q_VIEW)[-1] @ wrist.T_flange_cam
        Ti = np.linalg.inv(T)
        c, h = np.asarray(WORKSPACE_CENTER), np.asarray(WORKSPACE_HALF)
        pts = np.array([[c[0] + a * h[0], c[1] + b * h[1], z] for a in (-1, 1) for b in (-1, 1) for z in (0, 0.16)])
        uv = wrist.intr.project(pts @ Ti[:3, :3].T + Ti[:3, 3])
        self.assertTrue(np.all((uv[:, 0] > 0) & (uv[:, 0] < wrist.intr.width)))
        self.assertTrue(np.all((uv[:, 1] > 0) & (uv[:, 1] < 0.45 * wrist.intr.height)))    # clear of the gripper


class TestControllers(unittest.TestCase):
    def _grasp(self, x=0.1, n=5000, width=0.06, stamp=1.0):
        from vision.tabletop import TableObject, plan_topdown_grasp
        o = TableObject(points=np.zeros((1, 3)), rgb=np.zeros((1, 3)), center=np.array([x, -0.5, 0.04]), yaw=0.0,
                        length=0.08, width=width, height=0.08, top_z=0.08, n=n)
        return plan_topdown_grasp(o, 0.0, stamp=stamp)

    def test_goal_filter_rejects_occluded_estimates(self):
        from vision.controllers import GoalFilter
        f = GoalFilter()
        f.update(self._grasp(stamp=1), tcp_z=0.5)
        g = f.update(self._grasp(x=0.12, n=1200, width=0.03, stamp=2), tcp_z=0.4)   # fewer points, shrunk
        self.assertAlmostEqual(g.position[0], 0.1)
        f.update(self._grasp(x=0.12, n=1100, width=0.03, stamp=3), tcp_z=0.4)
        self.assertTrue(f.frozen)
        f2 = GoalFilter()
        f2.update(self._grasp(stamp=1), tcp_z=0.5)
        g2 = f2.update(self._grasp(x=0.11, n=6000, stamp=2), tcp_z=0.4)            # consistent: accepted
        self.assertAlmostEqual(g2.position[0], 0.105, places=6)

    def test_mpc_moves_toward_goal_within_limits(self):
        from vision.controllers import MPCGrasp
        c = MPCGrasp(lambda w: 100)
        c.reset_approach(None)
        g = self._grasp()
        u = c.solve(np.array([0.2, -0.45, 0.35]), g)
        self.assertIsNotNone(u)
        self.assertTrue(np.all(np.abs(u) <= c.v_max + 1e-6))
        self.assertLess(u[0], 0)                       # toward x = 0.1
        self.assertLessEqual(np.linalg.norm(u), np.linalg.norm(c.a_max * c.dt * np.ones(3)) + 1e-6)  # first step: accel


if __name__ == "__main__":
    unittest.main()
