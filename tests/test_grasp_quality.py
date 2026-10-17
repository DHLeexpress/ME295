import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from shear_agent import (  # noqa: E402
    friction_margin,
    gentlest_closure,
    grasp_margin,
    required_grip_force,
    shear_magnitude,
)


class TestFrictionMargin(unittest.TestCase):
    def test_reference_points(self):
        self.assertAlmostEqual(float(friction_margin(0.0, 10.0, 0.5)), 1.0)   # no shear
        self.assertAlmostEqual(float(friction_margin(5.0, 10.0, 0.5)), 0.0)   # on the cone
        self.assertLess(float(friction_margin(6.0, 10.0, 0.5)), 0.0)          # slipping

    def test_no_normal_force(self):
        self.assertEqual(float(friction_margin(1.0, 0.0, 0.5)), -np.inf)
        self.assertEqual(float(friction_margin(0.0, 0.0, 0.5)), 0.0)

    def test_grasp_margin_is_weakest_contact(self):
        q = grasp_margin([1.0, 4.0], [10.0, 10.0], [0.5, 0.5])
        self.assertAlmostEqual(float(q), 0.2)

    def test_shear_magnitude(self):
        self.assertAlmostEqual(float(shear_magnitude([3.0, 4.0])), 5.0)

    def test_required_force_hits_target_margin(self):
        for q_star in (0.0, 0.2, 0.5):
            f_n = required_grip_force(3.0, 0.4, q_star)
            self.assertAlmostEqual(float(friction_margin(3.0, f_n, 0.4)), q_star)
        with self.assertRaises(ValueError):
            required_grip_force(1.0, 0.5, 1.0)

    def test_gentlest_closure(self):
        # closure -> grip force map f_n = 125 * alpha (2F-140 max force), shear 5 N, mu 0.5
        margin_at = lambda a: float(friction_margin(5.0, 125.0 * a, 0.5))  # noqa: E731
        alpha = gentlest_closure(margin_at, q_star=0.2)
        self.assertAlmostEqual(alpha, 0.1, places=6)  # 5 / (0.5 * 125 * 0.8)
        self.assertIsNone(gentlest_closure(lambda a: -1.0))


if __name__ == "__main__":
    unittest.main()
