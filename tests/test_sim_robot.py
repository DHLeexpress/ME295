import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_sim"))

try:
    import mujoco  # noqa: F401
except ImportError:  # the simulator is optional for the rest of the repo
    mujoco = None

if mujoco is not None:
    import numpy as np

    import sim_gripper as sg
    from shear_agent.control import Command, Controller, run
    from shear_agent.robot import (PDMSSkin, SimRobot, SimRobotiqGripper, SkinSensor, mat_to_pose,
                                   pose_to_mat)

SCENE = ROOT / "assembly" / "scene.xml"
TCP = (0, 0, 0.292, 0, 0, 0)


def ur10e_dh(q):
    """Universal Robots' published UR10e DH kinematics (flange = tool0)."""
    d = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]
    a = [0, -0.6127, -0.57155, 0, 0, 0]
    alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
    T = np.eye(4)
    for i in range(6):
        ct, st, ca, sa = np.cos(q[i]), np.sin(q[i]), np.cos(alpha[i]), np.sin(alpha[i])
        T = T @ np.array([[ct, -st * ca, st * sa, a[i] * ct], [st, ct * ca, -ct * sa, a[i] * st],
                          [0, sa, ca, d[i]], [0, 0, 0, 1]])
    return T


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestSimRobot(unittest.TestCase):
    def setUp(self):
        self.rob = SimRobot(SCENE, verbose=False)
        self.gripper = SimRobotiqGripper(self.rob)
        self.rob.set_tcp(TCP)
        park = self.rob.m.joint("box").qposadr[0]
        self.rob.d.qpos[park:park + 3] = [5, 5, 0.1]

    def test_kinematics_match_ur_controller(self):
        rng = np.random.default_rng(0)
        for q in rng.uniform(-np.pi, np.pi, (20, 6)):
            T = self.rob._fk_flange(q)
            np.testing.assert_allclose(T, ur10e_dh(q), atol=1e-9)

    def test_pose_round_trip(self):
        pose = [0.3, -0.2, 0.5, 0.4, -1.2, 2.0]
        np.testing.assert_allclose(mat_to_pose(pose_to_mat(pose)), pose, atol=1e-12)

    def test_moves_reach_their_targets(self):
        rob = self.rob
        rob.sleep(0.5)
        target = np.array(rob.getl())
        target[:3] += (0.0, -0.10, 0.05)
        rob.movel(target, acc=0.5, vel=0.1)
        np.testing.assert_allclose(rob.getl()[:3], target[:3], atol=2e-4)
        q = np.array(rob.getj())
        q[5] += 0.3
        rob.movej(q, acc=1.0, vel=0.5)
        np.testing.assert_allclose(rob.getj(), q, atol=1e-3)

    def test_tcp_force_zero_at_rest_and_payload(self):
        rob = self.rob
        rob.sleep(1.0)
        self.assertLess(np.linalg.norm(rob.get_tcp_force()), 0.05)
        rob.set_payload(rob.payload_mass - 0.5)  # under-declared payload reads as 0.5 kg pulling down
        rob.sleep(0.2)
        self.assertAlmostEqual(rob.get_tcp_force()[2], -0.5 * 9.81, delta=0.1)

    def test_nonblocking_move_and_stop(self):
        rob = self.rob
        rob.sleep(0.3)
        start = np.array(rob.getl())
        rob.movel(start + [0, 0, 0.2, 0, 0, 0], acc=0.5, vel=0.1, wait=False)
        rob.sleep(0.5)
        self.assertTrue(rob.is_program_running())
        rob.stopl(acc=1.0)
        rob.sleep(0.5)
        self.assertFalse(rob.is_program_running())
        z = rob.getl()[2] - start[2]
        self.assertTrue(0.01 < z < 0.2)

    def test_gripper_registers(self):
        g = self.gripper
        g.gripper_action(128)
        self.assertLessEqual(abs(g.get_position() - 128), 2)
        self.assertEqual(g.get_object_status(), 3)
        g.close_gripper()
        self.assertGreaterEqual(g.get_position(), 253)
        g.open_gripper()
        self.assertLessEqual(g.get_position(), 2)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestBookshelfScene(unittest.TestCase):
    def test_books_rest_where_the_task_expects_them(self):
        rob = SimRobot(ROOT / "assembly" / "scene_bookshelf.xml", verbose=False)
        m, d = rob.m, rob.d
        books = [b for b in range(m.nbody) if m.body(b).name.startswith("book_")]
        p0 = {b: d.xpos[b].copy() for b in books}
        rob.sleep(0.5)
        self.assertLess(max(np.linalg.norm(d.xpos[b] - p0[b]) for b in books), 1e-3)
        np.testing.assert_allclose(d.xpos[m.body("book_target").id], (0.0, 0.905, 0.535), atol=1e-3)
        half = lambda name: m.geom_size[m.geom(name).id][0]
        left = d.xpos[m.body("book_L1").id][0] + half("book_L1")
        right = d.xpos[m.body("book_R1").id][0] - half("book_R1")
        self.assertAlmostEqual(left, -0.030, delta=1e-3)   # 10 mm of air beside the 40 mm target book
        self.assertAlmostEqual(right, 0.030, delta=1e-3)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestSkinAndController(unittest.TestCase):
    def setUp(self):
        m, d = sg.load()
        sg.place_box_between_pdms(m, d)
        self.rob = SimRobot(model=m, verbose=False)
        self.rob.d.qpos[:] = d.qpos
        self.rob.reset(self.rob.getj())
        self.rob.d.qpos[:] = d.qpos
        mujoco.mj_forward(m, self.rob.d)
        self.rob.set_tcp(TCP)
        self.gripper = SimRobotiqGripper(self.rob, force=40)

    def test_skin_matches_bond_sensor_and_detects_object(self):
        rob, g = self.rob, self.gripper
        rob.sleep(0.3)
        g.close_gripper()
        self.assertEqual(g.get_object_status(), 2)  # stopped on an object while closing
        frame = PDMSSkin(rob.m, "left").read(rob.d)
        self.assertGreater(frame.total_normal, 10.0)
        self.assertAlmostEqual(frame.total_normal, abs(rob.d.sensor("left_pdms_force").data[2]), delta=0.5)
        self.assertGreater(frame.net_margin(), 0.0)

    def test_controller_loop_lifts_the_box(self):
        class Lift(Controller):  # close, then lift 5 cm with speedl
            rate_hz = 50.0

            def reset(self, obs):
                self.z0 = obs.tcp_pose[2]

            def act(self, obs):
                if obs.gripper_object != 2:
                    return Command(np.zeros(6), gripper_position=255)
                if obs.tcp_pose[2] - self.z0 < 0.05:
                    return Command(np.array([0, 0, 0.05, 0, 0, 0]))
                return Command(done=True)

        rob = self.rob
        sensors = {s: SkinSensor(PDMSSkin(rob.m, s), rate_hz=20.0) for s in ("left", "right")}
        for s in sensors.values():
            rob.callbacks.append(s.tick)
        rob.sleep(0.3)
        z0 = rob.d.xpos[rob.m.body("box").id][2]
        run(Lift(), rob, self.gripper, skins=sensors, max_time=10.0)
        rob.sleep(0.5)
        self.assertGreater(rob.d.xpos[rob.m.body("box").id][2] - z0, 0.04)
        self.assertGreater(sensors["left"].latest.total_normal, 5.0)


if __name__ == "__main__":
    unittest.main()
