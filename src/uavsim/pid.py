"""Discrete PID controller with derivative filtering and anti-windup.

Features:
    - derivative on measurement (no kick on setpoint changes)
    - first-order low-pass filter on the derivative term
    - output saturation with conditional-integration anti-windup
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PID:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = float("-inf")
    out_max: float = float("inf")
    tau_d: float = 0.05          # derivative filter time constant (s)

    _integral: float = field(default=0.0, init=False, repr=False)
    _d_filtered: float = field(default=0.0, init=False, repr=False)
    _prev_meas: Optional[float] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._d_filtered = 0.0
        self._prev_meas = None

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Advance the controller by dt seconds and return the control output."""
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.out_min > self.out_max:
            raise ValueError("out_min must not exceed out_max.")

        error = setpoint - measurement

        # Derivative on measurement, low-pass filtered
        if self._prev_meas is None:
            raw_d = 0.0
        else:
            raw_d = -(measurement - self._prev_meas) / dt
        alpha = dt / (self.tau_d + dt)
        self._d_filtered += alpha * (raw_d - self._d_filtered)
        self._prev_meas = measurement

        p_term = self.kp * error
        d_term = self.kd * self._d_filtered

        # Tentative output with the integral advanced by this step
        i_candidate = self._integral + self.ki * error * dt
        unsat = p_term + i_candidate + d_term
        out = min(max(unsat, self.out_min), self.out_max)

        # Conditional integration: only accept the integral if not pushing
        # further into saturation.
        if unsat == out or (unsat > self.out_max and error < 0.0) or (unsat < self.out_min and error > 0.0):
            self._integral = i_candidate
            out = min(max(p_term + self._integral + d_term, self.out_min), self.out_max)

        return out
