"""Grasp controllers for the ablation (same perception, gripper sequence and safety filter; different approach):

    scripted   open loop: one look, planned path (sense-plan-act baseline)
    pbvs       closed-loop position-based visual servoing, align then descend, goal frozen near the object
    mpc        receding-horizon linear MPC (OSQP) with an approach funnel, re-planned on every estimate
"""
from .base import GoalFilter, GraspObservation, TopDownGraspController, run_grasp
from .mpc import MPCGrasp
from .pbvs import PBVSGrasp
from .scripted import ScriptedGrasp

CONTROLLERS = {"scripted": ScriptedGrasp, "pbvs": PBVSGrasp, "mpc": MPCGrasp}

__all__ = ["CONTROLLERS", "GoalFilter", "GraspObservation", "MPCGrasp", "PBVSGrasp", "ScriptedGrasp",
           "TopDownGraspController", "run_grasp"]
