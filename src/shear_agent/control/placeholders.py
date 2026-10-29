"""Slots for the controllers that will replace the rule-based placeholder (same interface).

Each receives an Observation (joints, TCP pose/wrench, Robotiq registers, the skin's per-cell
normal/shear force and deformation, optional camera images) and returns a Command (speedl twist + Robotiq position/
force) at rate_hz. In simulation, a model-based controller can roll out copies of the MuJoCo
model (SimRobot.m, mujoco.MjData copies, or mujoco.rollout) to predict the skin's response.
"""
from __future__ import annotations

from .base import Command, Controller, Observation


class MPCController(Controller):
    """Model-predictive control over (TCP velocity, grip force): e.g. minimise pull-out time and
    grip force subject to the friction-margin constraint Q >= Q* along the horizon."""
    name = "MPC (placeholder)"

    def act(self, obs: Observation) -> Command:
        raise NotImplementedError("MPC controller: implement act(); see ShearMarginGrasp for the I/O")


class MPPIController(Controller):
    """Sampling-based MPC (MPPI): sample K perturbed (speedl, force) sequences, roll them out in
    MuJoCo (mujoco.rollout), weight by exp(-cost / lambda) with cost terms on the skin's margin,
    book slip and neighbour-book disturbance, execute the first command."""
    name = "MPPI (placeholder)"

    def act(self, obs: Observation) -> Command:
        raise NotImplementedError("MPPI controller: implement act(); see ShearMarginGrasp for the I/O")


class VLFAPolicy(Controller):
    """Learned vision-language-force-action policy: wrist image + instruction + skin (normal/shear and
    deformation maps) + wrench -> Command."""
    name = "VLFA policy (placeholder)"

    def __init__(self, instruction: str = "pull out the blue book"):
        self.instruction = instruction

    def act(self, obs: Observation) -> Command:
        raise NotImplementedError("VLFA policy: load a model and implement act(); see ShearMarginGrasp for the I/O")
