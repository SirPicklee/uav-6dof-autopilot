"""Lateral autopilot: heading hold with roll-angle inner loop and yaw damper.

Loops:
    heading error -> bank angle command   (P, limited to +-30 deg)
    bank error    -> aileron              (PID, derivative on measurement)
    yaw rate      -> rudder               (washout yaw damper)

Sign convention: positive aileron rolls right (positive phi), a right bank
turns the aircraft right (increasing heading). Positive rudder produces a
negative yaw moment, so a positive washed-out yaw rate commands positive rudder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uavsim import quaternion as quat
from uavsim.pid import PID

MAX_BANK = np.radians(30.0)
MAX_AILERON = np.radians(20.0)
MAX_RUDDER = np.radians(15.0)
K_HEADING = 1.0        # rad bank per rad heading error
K_YAW_DAMPER = 0.5     # rad rudder per rad/s washed-out yaw rate
TAU_WASHOUT = 2.0      # s


def wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class LateralAutopilot:
    target_heading: float  # rad
    bank_cmd: float = field(default=0.0, init=False)
    _r_lp: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.roll_pid = PID(
            kp=0.6, ki=0.0, kd=0.05,
            out_min=-MAX_AILERON, out_max=MAX_AILERON, tau_d=0.05,
        )

    def reset(self) -> None:
        self.roll_pid.reset()
        self._r_lp = 0.0
        self.bank_cmd = 0.0

    def update(self, y: np.ndarray, dt: float) -> tuple[float, float]:
        """Return (aileron, rudder) in rad from the 13-element state."""
        phi, _, psi = quat.to_euler(y[6:10])
        r = float(y[12])

        heading_error = wrap_pi(self.target_heading - psi)
        self.bank_cmd = float(np.clip(K_HEADING * heading_error, -MAX_BANK, MAX_BANK))
        aileron = self.roll_pid.update(self.bank_cmd, phi, dt)

        self._r_lp += dt / (TAU_WASHOUT + dt) * (r - self._r_lp)
        r_washed = r - self._r_lp
        rudder = float(np.clip(K_YAW_DAMPER * r_washed, -MAX_RUDDER, MAX_RUDDER))

        return float(aileron), rudder
