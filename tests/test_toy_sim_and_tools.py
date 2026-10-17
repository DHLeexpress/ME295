import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from shear_agent.agent import TOOL_NAMES, TOOLS, MockBackend, dispatch  # noqa: E402
from shear_agent.sim import (  # noqa: E402
    FixedForce,
    ShearMargin,
    SimConfig,
    SlipReactive,
    sample_conditions,
    simulate,
)


class TestToySim(unittest.TestCase):
    def setUp(self):
        self.books = sample_conditions(80, seed=0)

    def test_strong_fixed_grip_never_slips(self):
        res = simulate(FixedForce(120.0), *self.books)
        self.assertFalse(res.dropped.any())
        self.assertEqual(float(res.slip_mm.max()), 0.0)

    def test_shear_margin_is_gentler_than_strong_fixed_grip(self):
        ours = simulate(ShearMargin(), *self.books).summary()
        strong = simulate(FixedForce(60.0), *self.books).summary()
        self.assertLess(ours["mean_grip_N"], strong["mean_grip_N"])
        self.assertLessEqual(ours["drop_rate"], 0.05)

    def test_latency_increases_slip_for_reactive_control(self):
        fast = simulate(SlipReactive(), *self.books, cfg=SimConfig(sense_latency=0.0)).summary()
        slow = simulate(SlipReactive(), *self.books, cfg=SimConfig(sense_latency=0.2)).summary()
        self.assertGreater(slow["slip_mm"], fast["slip_mm"])

    def test_record_trace(self):
        res = simulate(ShearMargin(), 0.8, 0.4, 8.0, record=True)
        self.assertEqual(len(res.trace["t"]), int(SimConfig().T / SimConfig().dt))
        self.assertTrue(np.all(np.isfinite(res.trace["f_n"])))


class TestTools(unittest.TestCase):
    def test_schemas(self):
        for tool in TOOLS:
            self.assertEqual(set(tool), {"name", "description", "input_schema"})
            self.assertEqual(tool["input_schema"]["type"], "object")

    def test_every_tool_dispatches(self):
        be = MockBackend()
        args = {"set_margin": {"q_star": 0.3}, "execute_skill": {"skill": "pinch"}}
        for name in TOOL_NAMES:
            out = dispatch(be, name, args.get(name))
            self.assertIsInstance(out, dict)
        self.assertFalse(dispatch(be, "not_a_tool")["ok"])

    def test_regrasp_with_higher_margin_removes_slip(self):
        be = MockBackend()
        for skill in ("pinch", "pull"):
            be.execute_skill(skill)
        self.assertTrue(be.incipient_slip()["incipient_slip"])
        be.set_margin(0.3)
        be.execute_skill("regrasp")
        self.assertFalse(be.incipient_slip()["incipient_slip"])


if __name__ == "__main__":
    unittest.main()
