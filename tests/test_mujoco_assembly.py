import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_sim"))

try:
    import mujoco  # noqa: F401
except ImportError:  # the simulator is optional for the rest of the repo
    mujoco = None

if mujoco is not None:
    import sim_gripper as sim  # noqa: E402


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestAssembly(unittest.TestCase):
    def setUp(self):
        self.m, self.d = sim.load()

    def test_no_contacts_at_home(self):
        sim.reset(self.m, self.d)
        self.assertEqual(self.d.ncon, 0)

    def test_closed_gaps(self):
        # blade bolted through the shared AGC lug hole: at full closure the closest blade
        # surfaces are ~16.8 mm apart and the PDMS faces ~32.2 mm
        sim.reset(self.m, self.d)
        sim.set_fingers_kinematic(self.m, self.d, 0.7)
        g = sim.exact_gaps(self.m, self.d)
        self.assertAlmostEqual(g["blade"], 0.0168, delta=0.0005)
        self.assertAlmostEqual(g["pdms"], 0.0322, delta=0.0005)

    def test_open_close_without_self_contact(self):
        m, d = self.m, self.d
        sim.reset(m, d)
        sim.park_box(m, d)
        sim.gripper_ramp(m, d, 255, 2.0)
        self.assertGreater(d.joint("left_driver_joint").qpos[0], 0.69)
        self.assertEqual(sim.gripper_self_contacts(m, d), [])
        sim.gripper_ramp(m, d, 0, 2.0)
        self.assertLess(d.joint("left_driver_joint").qpos[0], 0.05)

    def test_pdms_holds_box(self):
        m, d = self.m, self.d
        sim.place_box_between_pdms(m, d)
        sim.step(m, d, 0.3)
        sim.gripper_ramp(m, d, 255, 1.5)
        f = sim.finger_forces(m, d, m.geom("box").id)
        self.assertEqual(set(f), {("left", "pdms"), ("right", "pdms")})  # not the blade body
        self.assertGreater(min(v[0] for v in f.values()), 100.0)
        self.assertAlmostEqual(f[("left", "pdms")][0], f[("right", "pdms")][0], delta=5.0)
        # the bond sensor sees what the contacts push on the layer
        state = sim.pdms_state(m, d)["left"]
        self.assertAlmostEqual(state["normal"], f[("left", "pdms")][0], delta=1.0)
        self.assertAlmostEqual(state["touch"], f[("left", "pdms")][0], delta=1.0)


if __name__ == "__main__":
    unittest.main()
