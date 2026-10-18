"""Altitude-hold and airspeed-hold autopilot for the fixed-wing UAV.

Loops:
    altitude error -> pitch command   (outer PID)
    pitch error    -> elevator        (inner PID, derivative gives pitch damping)
    airspeed error -> throttle        (PID around the trim throttle)

Sign convention: in this model a NEGATIVE elevator deflection pitches the
nose up, so the pitch PID output is subtracted from the trim elevator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
from uavsim.dynamics import TrimResult
from uavsim.pid import PID

MAX_PITCH_CMD = np.radians(11.0)
MAX_ELEVATOR = np.radians(20.0)


@dataclass
class Autopilot:
    trim: TrimResult
    target_altitude: float
    target_airspeed: float

    def __post_init__(self) -> None:
        self.alt_pid = PID(
            kp=0.004, ki=0.0002, kd=0.02, i_zone=5.0,
            out_min=-MAX_PITCH_CMD, out_max=MAX_PITCH_CMD, tau_d=0.5,
        )
        self.pitch_pid = PID(
            kp=1.0, ki=0.0, kd=0.3,
            out_min=-MAX_ELEVATOR, out_max=MAX_ELEVATOR, tau_d=0.05,
        )
        self.speed_pid = PID(
            kp=0.05, ki=0.01, kd=0.0,
            out_min=-self.trim.throttle, out_max=1.0 - self.trim.throttle,
        )

    def reset(self) -> None:
        self.alt_pid.reset()
        self.pitch_pid.reset()
        self.speed_pid.reset()

    def update(self, y: np.ndarray, dt: float, airspeed: float | None = None) -> Controls:
        """Compute controls from the 13-element state for one control step."""
        altitude = -y[2]
        airspeed = float(np.linalg.norm(y[3:6])) if airspeed is None else float(airspeed)
        pitch = quat.to_euler(y[6:10])[1]

        pitch_cmd = self.trim.pitch + self.alt_pid.update(self.target_altitude, altitude, dt)
        pitch_cmd = float(np.clip(pitch_cmd, -MAX_PITCH_CMD, MAX_PITCH_CMD))

        elevator_delta = self.pitch_pid.update(pitch_cmd, pitch, dt)
        elevator = self.trim.elevator - elevator_delta

        throttle = self.trim.throttle + self.speed_pid.update(self.target_airspeed, airspeed, dt)
        throttle = float(np.clip(throttle, 0.0, 1.0))

        return Controls(elevator=float(elevator), throttle=throttle)




