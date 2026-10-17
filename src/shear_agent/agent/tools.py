"""Tactile tools exposed to the LLM agent (layer C).

The agent never sees raw sensor frames. It calls a few typed tools whose
answers come from the low-level shear pipeline (layer A). Schemas use the
Anthropic Messages API tool format (name / description / input_schema) and map
1:1 to OpenAI-style function specs.

Backends implement the calls: ``MockBackend`` (offline, scripted book pull) now,
a ROS 2 backend on the real gripper later.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SKILLS = ["tilt", "pinch", "pull", "regrasp", "release"]

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "contact_state",
        "description": "Which fingers touch the object and the contact area seen by the thin tactile skin.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "shear_margin",
        "description": (
            "Friction-cone grasp margin Q = min_i (1 - |f_t,i| / (mu_i f_n,i)) from measured shear. "
            "Q near 0 means about to slip, Q < 0 means slipping. Also returns mu_hat and grip force."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "incipient_slip",
        "description": "True if the shear field shows slip starting at the contact edge (precedes gross slip).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_margin",
        "description": "Set the target margin Q* of the low-level shear-margin controller (higher = squeeze harder).",
        "input_schema": {
            "type": "object",
            "properties": {"q_star": {"type": "number", "minimum": 0.0, "maximum": 0.8}},
            "required": ["q_star"],
        },
    },
    {
        "name": "execute_skill",
        "description": "Run one book-extraction primitive and report the outcome.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "enum": SKILLS},
                "book_id": {"type": "string"},
            },
            "required": ["skill"],
        },
    },
]
TOOL_NAMES = [t["name"] for t in TOOLS]


class MockBackend:
    """Offline stand-in for the gripper: a scripted book pull.

    The book is heavier than the default margin assumes, so the first ``pull``
    at a low Q* shows incipient slip; a ``regrasp`` with a higher Q* fixes it.
    """

    def __init__(self, mu: float = 0.45, book_load_n: float = 9.0):
        self.mu, self.load = mu, book_load_n
        self.q_star, self.grip = 0.15, 10.0
        self.touching = {"left": False, "right": False}
        self.phase = "idle"

    # ---- tools ------------------------------------------------------------
    def contact_state(self) -> Dict[str, Any]:
        return {"left": self.touching["left"], "right": self.touching["right"],
                "contact_area_mm2": 180 if any(self.touching.values()) else 0}

    def shear_margin(self) -> Dict[str, Any]:
        f_t = self._shear()
        q = 1.0 - f_t / (self.mu * self.grip) if self.grip > 0 else 0.0
        return {"Q": round(q, 2), "mu_hat": self.mu, "grip_N": round(self.grip, 1), "shear_N": round(f_t, 1)}

    def incipient_slip(self) -> Dict[str, Any]:
        return {"incipient_slip": self.shear_margin()["Q"] < 0.1}

    def set_margin(self, q_star: float) -> Dict[str, Any]:
        self.q_star = float(q_star)
        return {"ok": True, "q_star": self.q_star}

    def execute_skill(self, skill: str, book_id: Optional[str] = None) -> Dict[str, Any]:
        if skill not in SKILLS:
            return {"ok": False, "error": f"unknown skill {skill!r}"}
        if skill == "tilt":
            self.phase = "tilted"
        elif skill == "pinch":
            self.touching = {"left": True, "right": True}
            self.phase = "pinched"
        elif skill in ("pull", "regrasp"):
            self.phase = "pulling"
            # the low-level controller sizes the grip for an *expected* load; the
            # real load is higher, which a low Q* does not cover
            expected = 0.6 * self.load
            self.grip = max(10.0, 0.5 * expected / (self.mu * (1.0 - self.q_star)))
            if skill == "regrasp":
                self.grip = max(self.grip, 0.5 * self.load / (self.mu * (1.0 - self.q_star)))
        elif skill == "release":
            self.touching = {"left": False, "right": False}
            self.phase, self.grip = "idle", 0.0
        return {"ok": True, "skill": skill, "book_id": book_id, "phase": self.phase}

    # ---- internals --------------------------------------------------------
    def _shear(self) -> float:
        return 0.5 * self.load if self.phase == "pulling" else 0.0


def dispatch(backend: Any, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a tool call coming from the LLM."""
    if name not in TOOL_NAMES:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    return getattr(backend, name)(**(args or {}))
