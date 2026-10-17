"""Friction-cone grasp margin from measured shear (layer A).

For contact i with tangential force f_t,i (from the tactile skin's shear field),
normal force f_n,i (from the gripper force/current) and friction coefficient mu_i:

    q_i = 1 - ||f_t,i|| / (mu_i * f_n,i)

    q_i = 1  no shear load
    q_i = 0  on the friction-cone boundary (slip onset)
    q_i < 0  outside the cone (slipping)

The grasp margin is the weakest contact, Q = min_i q_i. The "gentlest stable
closure" is the smallest closure alpha in [0, 1] whose predicted Q >= Q*.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

_EPS = 1e-9


def shear_magnitude(f_t_xy) -> np.ndarray:
    """||f_t|| from in-plane shear components with shape (..., 2)."""
    return np.linalg.norm(np.asarray(f_t_xy, dtype=float), axis=-1)


def friction_margin(ft_norm, f_n, mu) -> np.ndarray:
    """Per-contact margin q_i = 1 - ||f_t|| / (mu f_n). Inputs broadcast.

    A contact with no normal force has margin 0 if unloaded (it cannot resist
    any shear) and -inf if it is already carrying shear.
    """
    ft_norm, f_n, mu = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (ft_norm, f_n, mu))
    )
    cap = mu * f_n
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 1.0 - ft_norm / cap
    no_cap = np.where(ft_norm > _EPS, -np.inf, 0.0)
    return np.where(cap > _EPS, q, no_cap)


def grasp_margin(ft_norm, f_n, mu) -> np.ndarray:
    """Q = min_i q_i, taken over the last axis (contacts)."""
    return np.min(friction_margin(ft_norm, f_n, mu), axis=-1)


def required_grip_force(ft_norm, mu, q_star: float = 0.2) -> np.ndarray:
    """Smallest normal force with q >= q_star:  f_n = ||f_t|| / (mu (1 - q_star))."""
    if not 0.0 <= q_star < 1.0:
        raise ValueError("q_star must be in [0, 1)")
    return np.asarray(ft_norm, dtype=float) / (np.asarray(mu, dtype=float) * (1.0 - q_star))


def gentlest_closure(
    margin_at: Callable[[float], float],
    q_star: float = 0.2,
    alphas: Optional[Sequence[float]] = None,
) -> Optional[float]:
    """Smallest closure alpha in [0, 1] (0 = open, 1 = fully closed) with Q(alpha) >= q_star.

    ``margin_at(alpha)`` predicts the grasp margin for a closure, e.g. from a
    calibrated closure->force map and the current shear estimate. Returns None if
    no candidate closure is stable.
    """
    grid = np.linspace(0.0, 1.0, 101) if alphas is None else np.sort(np.asarray(alphas, float))
    for a in grid:
        if margin_at(float(a)) >= q_star - 1e-9:  # tolerance for float round-off
            return float(a)
    return None
