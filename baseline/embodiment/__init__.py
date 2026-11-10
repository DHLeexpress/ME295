"""Modular end-effector description and forward kinematics (arm -> coupling -> gripper -> finger tool -> sensor pad).

One TOML card per hardware component in cards/ (mirroring model/), one assembly card that stacks them. A new blade or
skin is a new card; python -m baseline.embodiment.check verifies an assembly against the MuJoCo model.
"""
from .assembly import SIDES, EndEffector, FixedTCPArm, available_cards
from .components import DHArm, FingerTool, LinkageGripper, Mount, PadSensor, load_card

__all__ = ["SIDES", "DHArm", "EndEffector", "FingerTool", "FixedTCPArm", "LinkageGripper", "Mount", "PadSensor",
           "available_cards", "load_card"]
