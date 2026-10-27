"""Contact-phase controllers: interface (base), rule-based placeholder (margin), future slots (placeholders)."""
from .base import Command, Controller, Observation, apply, observe, run
from .margin import ShearMarginGrasp
from .placeholders import MPCController, MPPIController, VLFAPolicy

CONTROLLERS = {"margin": ShearMarginGrasp, "mpc": MPCController, "mppi": MPPIController, "vlfa": VLFAPolicy}

__all__ = ["CONTROLLERS", "Command", "Controller", "MPCController", "MPPIController", "Observation",
           "ShearMarginGrasp", "VLFAPolicy", "apply", "observe", "run"]
