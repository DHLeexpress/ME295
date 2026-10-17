"""Shear-aware agentic grasping with a thin tactile skin on a parallel-jaw gripper."""
from .grasp_quality import (
    friction_margin,
    gentlest_closure,
    grasp_margin,
    required_grip_force,
    shear_magnitude,
)

__all__ = [
    "friction_margin",
    "gentlest_closure",
    "grasp_margin",
    "required_grip_force",
    "shear_magnitude",
]
__version__ = "0.0.1"
