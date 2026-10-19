"""Control-surface actuator model: first-order lag with rate and position limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Actuator:
    tau: float = 0.05                    # lag time constant (s)
    rate_limit: float = float("inf")     # max |rate| (units per second)
    pos_limit: float = float("inf")      # max |position|
    x: float = 0.0                       # current position

    def reset(self, x0: float = 0.0) -> None:
        self.x = float(x0)

    def update(self, command: float, dt: float) -> float:
        """Advance the actuator by dt seconds and return its position."""
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        command = float(np.clip(command, -self.pos_limit, self.pos_limit))
        rate = (command - self.x) / self.tau if self.tau > 0.0 else (command - self.x) / dt
        rate = float(np.clip(rate, -self.rate_limit, self.rate_limit))
        self.x = float(np.clip(self.x + rate * dt, -self.pos_limit, self.pos_limit))
        return self.x
