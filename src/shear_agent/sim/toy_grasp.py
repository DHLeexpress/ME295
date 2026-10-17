"""Toy 1-D grasp/slip simulator: an evaluation harness for layer A.

This is NOT a model of our sensor or of a real book. A book is held by two
Coulomb contacts while it is pulled out of a shelf:

* the tangential load rises as the shelf hands over the book's weight, the pull
  accelerates, neighbouring books resist, and the book briefly snags;
* while the grasp cannot carry the load, the fingers slide along the book at the
  pull speed (kinematic slip); slip beyond ``drop_slip`` counts as a failure;
* the gripper force follows its command with a first-order lag;
* controllers see per-contact shear and grip force sampled at the sensor
  frame rate, with latency and noise.

Purpose: fix metrics and baselines before hardware time, and ask how much the
~50 ms sensing loop limits shear-based control. Numbers from this file are not
evidence about the real system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..grasp_quality import required_grip_force

G = 9.81


@dataclass
class SimConfig:
    dt: float = 1e-3                # integration step (s)
    T: float = 3.0                  # episode length (s)
    sense_period: float = 0.05      # sensor frame period (s), ~20 Hz
    sense_latency: float = 0.05     # acquisition -> strain -> ROS transport (s)
    shear_noise: float = 0.15       # std of per-contact shear estimate (N)
    normal_noise: float = 0.5       # std of grip-force estimate (N)
    grip_tau: float = 0.08          # first-order grip-force response (s), to be measured
    f_min: float = 10.0             # Robotiq 2F-140 grip-force range (N), product sheet
    f_max: float = 125.0
    mu_k_ratio: float = 0.8         # kinetic / static friction
    pull_speed: float = 0.05        # finger-on-book sliding speed during slip (m/s)
    drop_slip: float = 0.02         # accumulated slip counted as a failed extraction (m)
    hold_start: float = 0.5         # metrics are computed from here on (s)


def _smoothstep(t, t0, t1):
    x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _bump(t, t0, t1):
    x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return np.sin(np.pi * x) ** 2


def tangential_load(t, mass, resist_peak):
    """Total tangential load (N) the two contacts carry while pulling a book out."""
    lift = _smoothstep(t, 0.5, 0.7)                  # shelf hands over the weight
    accel = 3.0 * _bump(t, 0.5, 0.9)                 # pull acceleration (m/s^2)
    resist = resist_peak * _bump(t, 0.7, 1.7)        # friction from neighbouring books
    snag = 0.6 * resist_peak * _bump(t, 2.2, 2.45)   # brief snag on the shelf edge
    return mass * (G * lift + accel) + resist + snag


# --------------------------------------------------------------------------- controllers
class Controller:
    name = "controller"

    def reset(self, n: int, cfg: SimConfig) -> None:
        self.n, self.cfg = n, cfg

    def __call__(self, t, f_t, f_n, slipping):
        """Return the commanded grip force (N) for every trial, shape (n,)."""
        raise NotImplementedError


class FixedForce(Controller):
    """Gripper default: close with one force setting and never change it."""

    def __init__(self, force: float):
        self.force = force
        self.name = f"Fixed {force:.0f} N"

    def __call__(self, t, f_t, f_n, slipping):
        return np.full(self.n, self.force)


class SlipReactive(Controller):
    """Classical tactile baseline: close gently, add force after each detected slip
    (slip detection -> regrasp with more force, cf. Dong et al. 2017)."""

    name = "Slip-reactive"

    def __init__(self, f0: float = 10.0, step: float = 10.0, refractory: float = 0.1):
        self.f0, self.step, self.refractory = f0, step, refractory

    def reset(self, n, cfg):
        super().reset(n, cfg)
        self.f = np.full(n, max(self.f0, cfg.f_min))
        self.last = np.full(n, -np.inf)

    def __call__(self, t, f_t, f_n, slipping):
        bump = slipping & (t - self.last >= self.refractory)
        self.f = np.where(bump, self.f + self.step, self.f)
        self.last = np.where(bump, t, self.last)
        return np.clip(self.f, self.cfg.f_min, self.cfg.f_max)


class ShearMargin(Controller):
    """Ours (layer A): keep the friction margin Q >= q_star using measured shear,
    plus a slip reflex.

    f_cmd = f_t_pred / (mu_hat (1 - q_star)). f_t_pred extrapolates the measured
    shear rate over the known loop delay (lead = sensing latency + gripper lag by
    default) to cover sensing + actuation lag. mu_hat starts from a
    conservative prior (e.g. an LLM / DeliGrasp-style guess) and is lowered to the
    observed f_t / f_n ratio whenever slip is seen. During gross slip the measured
    shear saturates at the friction limit and under-reports the load, so a reflex
    (same +step as the slip-reactive baseline) is applied on top.
    """

    name = "Shear-margin (ours)"

    def __init__(self, q_star: float = 0.2, mu0: float = 0.35, lead: Optional[float] = None,
                 reflex_step: float = 10.0):
        self.q_star, self.mu0, self.lead_arg, self.reflex_step = q_star, mu0, lead, reflex_step

    def reset(self, n, cfg):
        super().reset(n, cfg)
        self.lead = cfg.sense_latency + cfg.grip_tau if self.lead_arg is None else self.lead_arg
        self.mu = np.full(n, self.mu0)
        self.prev = None
        self.rate = np.zeros(n)

    def __call__(self, t, f_t, f_n, slipping):
        if self.prev is not None:
            self.rate = 0.5 * self.rate + 0.5 * (f_t - self.prev) / self.cfg.sense_period
        self.prev = f_t.copy()
        observed = f_t / np.maximum(f_n, 1e-6)
        self.mu = np.clip(np.where(slipping, np.minimum(self.mu, 0.95 * observed), self.mu), 0.05, 2.0)
        f_pred = f_t + np.maximum(self.rate, 0.0) * self.lead
        f_cmd = required_grip_force(f_pred, self.mu, self.q_star)
        f_cmd = np.where(slipping, np.maximum(f_cmd, f_n + self.reflex_step), f_cmd)
        return np.clip(f_cmd, self.cfg.f_min, self.cfg.f_max)


def default_controllers() -> List[Controller]:
    return [FixedForce(15.0), FixedForce(60.0), SlipReactive(), ShearMargin()]


# --------------------------------------------------------------------------- simulation
@dataclass
class Result:
    name: str
    dropped: np.ndarray      # failed extraction (slip >= drop_slip)
    slip_mm: np.ndarray      # accumulated finger-on-book slip
    mean_grip: np.ndarray    # mean grip force while holding (N)
    peak_grip: np.ndarray    # peak grip force while holding (N)
    min_margin: np.ndarray   # worst required-vs-available margin while holding
    trace: Optional[Dict[str, np.ndarray]] = None

    def summary(self) -> Dict[str, float]:
        return {
            "drop_rate": float(self.dropped.mean()),
            "slip_mm": float(self.slip_mm.mean()),
            "mean_grip_N": float(np.nanmean(self.mean_grip)),
            "peak_grip_N": float(self.peak_grip.mean()),
        }


def sample_conditions(n: int, seed: int = 0):
    """Random books: mass 0.2-1.2 kg, skin-cover friction 0.3-0.9, neighbour resistance 0-10 N."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 1.2, n), rng.uniform(0.3, 0.9, n), rng.uniform(0.0, 10.0, n)


def simulate(controller: Controller, mass, mu, resist_peak, cfg: Optional[SimConfig] = None,
             seed: int = 0, record: bool = False) -> Result:
    """Run one episode per condition (vectorised over trials)."""
    cfg = cfg or SimConfig()
    mass, mu, resist_peak = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(v, dtype=float)) for v in (mass, mu, resist_peak)))
    n = mass.size
    rng = np.random.default_rng(seed)
    steps = int(round(cfg.T / cfg.dt))
    period = max(1, int(round(cfg.sense_period / cfg.dt)))
    delay = int(round(cfg.sense_latency / cfg.dt))
    mu_k = cfg.mu_k_ratio * mu

    controller.reset(n, cfg)
    f_n = np.full(n, cfg.f_min)               # gripper closed on the book at minimum force
    f_cmd = f_n.copy()
    slipping = np.zeros(n, dtype=bool)
    slip = np.zeros(n)
    dropped = np.zeros(n, dtype=bool)
    hist_ft, hist_fn = np.zeros((steps, n)), np.zeros((steps, n))
    hist_slip = np.zeros((steps, n), dtype=bool)
    grip_sum, grip_cnt = np.zeros(n), np.zeros(n)
    peak, q_min = np.zeros(n), np.full(n, np.inf)
    keys = ("t", "demand", "f_n", "slip_mm", "margin")
    trace = {k: np.zeros(steps) for k in keys} if record else None

    for k in range(steps):
        t = k * cfg.dt
        demand = np.where(dropped, 0.0, 0.5 * tangential_load(t, mass, resist_peak))  # per contact
        cap_s, cap_k = mu * f_n, mu_k * f_n
        slipping = np.where(slipping, demand > cap_k, demand > cap_s) & ~dropped
        slip = slip + np.where(slipping, cfg.pull_speed * cfg.dt, 0.0)
        dropped = dropped | (slip >= cfg.drop_slip)
        f_t = np.where(dropped, 0.0, np.where(slipping, cap_k, demand))
        hist_ft[k], hist_fn[k], hist_slip[k] = f_t, f_n, slipping

        if k % period == 0:                    # a new sensor frame reaches the controller
            j = max(k - delay, 0)
            ft_meas = np.abs(hist_ft[j] + rng.normal(0.0, cfg.shear_noise, n))
            fn_meas = np.maximum(hist_fn[j] + rng.normal(0.0, cfg.normal_noise, n), 0.0)
            f_cmd = controller(t, ft_meas, fn_meas, hist_slip[j])
        f_n = f_n + cfg.dt / cfg.grip_tau * (f_cmd - f_n)

        margin = 1.0 - demand / np.maximum(mu * f_n, 1e-9)
        if t >= cfg.hold_start:
            held = ~dropped
            grip_sum += np.where(held, f_n, 0.0)
            grip_cnt += held
            peak = np.maximum(peak, np.where(held, f_n, 0.0))
            q_min = np.where(held, np.minimum(q_min, margin), q_min)
        if record:
            for key, val in zip(keys, (t, demand[0], f_n[0], 1e3 * slip[0], margin[0])):
                trace[key][k] = val

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grip = np.where(grip_cnt > 0, grip_sum / np.maximum(grip_cnt, 1), np.nan)
    return Result(controller.name, dropped, 1e3 * slip, mean_grip, peak, q_min, trace)
