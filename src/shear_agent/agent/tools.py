"""Tactile tools exposed to the LLM agent (layer C).

The agent never sees raw sensor frames and never commands forces. It calls a
few typed tools whose answers come from the low-level shear pipeline (layer A).
Every call is validated against its JSON schema before it reaches a backend, so
an out-of-range or malformed request from the model becomes an error message for
the model, not a robot command.

Schemas use the Anthropic Messages API tool format (name / description /
input_schema); ``llm.to_openai_tools`` converts them for OpenAI-compatible APIs.
Backends: ``MockBackend`` (offline, scripted book pull) now; ROS 2 on the real
gripper later.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SKILLS = ["insert", "tilt", "pinch", "pull", "regrasp", "release"]
Q_STAR_MAX = 0.8

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
            "properties": {"q_star": {"type": "number", "minimum": 0.0, "maximum": Q_STAR_MAX}},
            "required": ["q_star"],
        },
    },
    {
        "name": "execute_skill",
        "description": (
            "Run one book-extraction primitive: insert (slide the thin fingers into the gaps beside the "
            "book), tilt (tip the book out by its top edge when the gaps are too tight), pinch, pull "
            "(horizontally out of the shelf), regrasp, release."
        ),
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
_SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}


def validate(name: str, args: Any) -> Optional[str]:
    """Return an error message if ``args`` violates the tool's schema, else None."""
    schema = _SCHEMAS[name]
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    unknown = [k for k in args if k not in props]
    if unknown:
        return f"unknown argument(s): {', '.join(unknown)}"
    for key, val in args.items():
        spec = props[key]
        if spec.get("type") == "number":
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return f"{key} must be a number"
            if "minimum" in spec and val < spec["minimum"]:
                return f"{key} must be >= {spec['minimum']}"
            if "maximum" in spec and val > spec["maximum"]:
                return f"{key} must be <= {spec['maximum']}"
        if spec.get("type") == "string" and not isinstance(val, str):
            return f"{key} must be a string"
        if "enum" in spec and val not in spec["enum"]:
            return f"{key} must be one of {spec['enum']}"
    return None


def dispatch(backend: Any, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute one tool call from the LLM; never raises, errors go back to the model."""
    if name not in TOOL_NAMES:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    args = {} if args is None else args
    error = validate(name, args)
    if error:
        return {"ok": False, "error": error}
    try:
        return getattr(backend, name)(**args)
    except Exception as e:  # backend failure is reported, not propagated to the loop
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


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
        if skill in ("insert", "tilt"):
            self.phase = "inserted" if skill == "insert" else "tilted"
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
