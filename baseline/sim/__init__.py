"""Librarian pick-up environment, scene variations and scripted expert for the VLA baseline."""
import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")   # headless Linux: must be set before mujoco is first imported
