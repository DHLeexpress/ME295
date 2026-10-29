"""Contact-phase controllers: interface (base), rule-based placeholders (margin), future slots (placeholders)."""
from .base import Command, Controller, Observation, apply, observe, run
from .margin import FixedForceGrasp, ShearMarginGrasp
from .placeholders import MPCController, MPPIController, VLFAPolicy

CONTROLLERS = {"margin": ShearMarginGrasp, "fixed": FixedForceGrasp, "mpc": MPCController,
               "mppi": MPPIController, "vlfa": VLFAPolicy}

__all__ = ["CONTROLLERS", "Command", "Controller", "FixedForceGrasp", "MPCController", "MPPIController",
           "Observation", "ShearMarginGrasp", "VLFAPolicy", "apply", "observe", "run"]
