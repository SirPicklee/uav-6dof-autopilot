"""Mahony complementary attitude filter (accelerometer-only correction).

Quaternion q rotates body -> NED. The accelerometer measures specific force,
which equals -g (pointing up) in unaccelerated flight. During turns the
measured direction is biased by centripetal acceleration; this version does
NOT compensate for that. Yaw is not corrected (gyro integration only).
"""

from __future__ import annotations

import numpy as np

from uavsim import quaternion as quat


class MahonyFilter:
    def __init__(self, q0, kp=1.0, ki=0.0):
        self.q = quat.normalize(np.asarray(q0, dtype=float))
        self.kp = kp
        self.ki = ki
        self._integral = np.zeros(3)

    def update(self, gyro, accel, dt):
        norm = float(np.linalg.norm(accel))
        omega = np.asarray(gyro, dtype=float).copy()
        if norm > 1e-6:
            a = np.asarray(accel) / norm                    # measured "up" in body
            v = quat.to_dcm(self.q).T @ np.array([0.0, 0.0, -1.0])  # estimated "up"
            e = np.cross(a, v)
            self._integral += self.ki * e * dt
            omega = omega + self.kp * e + self._integral
        self.q = quat.normalize(self.q + quat.derivative(self.q, omega) * dt)
        return self.q
