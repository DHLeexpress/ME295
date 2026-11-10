"""Embodiment cards vs the MuJoCo assembly, retargeting arm, and the scene builder.

    python -m unittest discover -s baseline/tests -t .      (from the ME295 root; ~10 s, no GPU)
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from baseline.embodiment import EndEffector, FixedTCPArm, se3  # noqa: E402
from baseline.embodiment.check import run_check  # noqa: E402


class TestEndEffector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ee = EndEffector.load()

    def test_fk_matches_mujoco(self):
        res = run_check(self.ee, n=25, verbose=False)
        self.assertTrue(res["ok"], res)

    def test_tcp_and_gap(self):
        ee = self.ee
        self.assertAlmostEqual(ee.width(0.0), 0.1609, places=3)          # assembly/README.md: open 160.9 mm
        self.assertAlmostEqual(ee.width(ee.gripper.q_closed), 0.0322, places=3)   # closed 32.2 mm
        q = ee.q_for_width(0.040)
        self.assertAlmostEqual(ee.width(q), 0.040, places=6)
        self.assertAlmostEqual(ee.tcp(q)[2, 3], 0.292, delta=0.001)      # run_librarian.py TCP (0, 0, 0.292)
        np.testing.assert_allclose(ee.tcp(q)[:3, :3], np.eye(3))          # flange axes by default
        travel = ee.tcp(ee.gripper.q_closed)[2, 3] - ee.tcp(0.0)[2, 3]   # fingertips swing ~24 mm toward the tip
        self.assertAlmostEqual(travel, 0.0237, delta=0.0005)

    def test_jaw_axes(self):
        ee, q = self.ee, 0.3
        J = ee.tcp(q, "jaw")
        y = ee.jaw_axis(q)
        np.testing.assert_allclose(J[:3, 1], y, atol=1e-12)
        np.testing.assert_allclose(J[:3, 2], [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(abs(y[0]), 1.0, delta=1e-3)                  # jaw ~ flange x (0.5 deg tilt)

    def test_ur_ik_roundtrip(self):
        ee = self.ee
        q0 = np.array(ee.arm.q_home) + 0.2
        T = ee.fk(q0, 0.4)["tcp"]
        q, err = ee.ik_tcp(T, np.array(ee.arm.q_home), 0.4)
        self.assertLess(err, 1e-8)
        np.testing.assert_allclose(ee.fk(q, 0.4)["tcp"], T, atol=1e-7)


class TestVirtualFranka(unittest.TestCase):
    def test_panda_zero_pose(self):
        fr = FixedTCPArm.load("virtual_franka_droid.toml")
        T = fr.arm.fk(np.zeros(7))
        np.testing.assert_allclose(T[:3, 3], [0.088, 0.0, 0.926], atol=1e-9)   # Franka docs: flange at q = 0
        np.testing.assert_allclose(T[:3, 2], [0, 0, -1], atol=1e-9)

    def test_jacobian(self):
        fr = FixedTCPArm.load("virtual_franka_droid.toml")
        q = np.array(fr.arm.q_home) + 0.1
        J = fr.arm.jacobian(q, fr.T_flange_tcp)
        eps, Jn = 1e-6, np.zeros((6, 7))
        for i in range(7):
            d = np.zeros(7)
            d[i] = eps
            Tp, Tm = fr.tcp(q + d), fr.tcp(q - d)
            Jn[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
            Jn[3:, i] = se3.mat_to_rotvec(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * eps)
        np.testing.assert_allclose(J, Jn, atol=1e-7)


class TestScene(unittest.TestCase):
    def test_nominal_is_the_simulator_scene(self):
        from baseline.sim import scene
        m, info = scene.build(scene.nominal())
        self.assertEqual(info.designated, "book_target")
        np.testing.assert_allclose(info.center, [0.0, 0.905, 0.535])
        self.assertAlmostEqual(info.mass, 0.8)
        self.assertEqual(info.instruction, "pick up the dark blue book")

    def test_random_gaps(self):
        from baseline.sim import scene
        import mujoco
        v = scene.sample(np.random.default_rng(7))
        m, info = scene.build(v)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        i = v.order.index(v.designated)
        x = lambda n: d.xpos[m.body(n).id][0]
        h = lambda n: m.geom_size[m.geom(n).id][0]
        gl = (x(v.designated) - h(v.designated)) - (x(v.order[i - 1]) + h(v.order[i - 1]))
        gr = (x(v.order[i + 1]) - h(v.order[i + 1])) - (x(v.designated) + h(v.designated))
        self.assertAlmostEqual(gl, v.gap_left, places=9)
        self.assertAlmostEqual(gr, v.gap_right, places=9)
        self.assertGreaterEqual(2 * info.half[0], scene.MIN_GRASP)


if __name__ == "__main__":
    unittest.main()
