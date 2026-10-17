"""Numerical integrators for the flight dynamics model.

All integrators operate on a state vector y with dynamics dy/dt = f(t, y).
A `post_step` hook lets the caller project the state back onto its
constraint manifold after each step (e.g. renormalize a quaternion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

Derivative = Callable[[float, np.ndarray], np.ndarray]
PostStep = Callable[[np.ndarray], np.ndarray]
StopCondition = Callable[[float, np.ndarray], bool]


@dataclass
class Trajectory:
    """Time history produced by `integrate`."""

    t: np.ndarray  # shape (N,)
    y: np.ndarray  # shape (N, n_states)

    @property
    def final_state(self) -> np.ndarray:
        return self.y[-1]


def euler_step(f: Derivative, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Explicit Euler step (1st order). Kept for accuracy comparisons."""
    return y + dt * f(t, y)


def rk4_step(f: Derivative, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Classical Runge-Kutta step (4th order)."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


STEPPERS = {
    "euler": euler_step,
    "rk4": rk4_step,
}


def integrate(
    f: Derivative,
    y0: np.ndarray,
    t_span: tuple[float, float],
    dt: float,
    method: str = "rk4",
    post_step: Optional[PostStep] = None,
    stop_condition: Optional[StopCondition] = None,
) -> Trajectory:
    """Integrate dy/dt = f(t, y) from t_span[0] to t_span[1] with fixed step dt.

    Args:
        f: derivative function f(t, y) -> dy/dt.
        y0: initial state vector.
        t_span: (t_start, t_end) in seconds.
        dt: fixed time step in seconds.
        method: "euler" or "rk4".
        post_step: optional function applied to the state after every step.
        stop_condition: optional function (t, y) -> bool; integration stops
            (inclusive) at the first step where it returns True.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if method not in STEPPERS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(STEPPERS)}.")

    t0, t1 = t_span
    if t1 <= t0:
        raise ValueError("t_span must satisfy t_end > t_start.")

    step = STEPPERS[method]
    n_steps = int(np.ceil((t1 - t0) / dt - 1e-12))

    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()
    ts = [t]
    ys = [y.copy()]

    for _ in range(n_steps):
        h = min(dt, t1 - t)
        y = step(f, t, y, h)
        t += h
        if post_step is not None:
            y = post_step(y)
        ts.append(t)
        ys.append(y.copy())
        if stop_condition is not None and stop_condition(t, y):
            break

    return Trajectory(t=np.array(ts), y=np.array(ys))
