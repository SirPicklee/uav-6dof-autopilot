"""Extended Kalman filter for position, velocity and accelerometer bias.

State x = [pn, pe, pd, vn, ve, vd, bn, be, bd]  (NED). Attitude is supplied
(ground truth in this version); the bias is modelled in the NED frame, which is
an approximation that degrades when the aircraft turns.
"""

from __future__ import annotations

import numpy as np

G_NED = np.array([0.0, 0.0, 9.80665])


class EKF:
    def __init__(self, pos0, vel0, pos_sigma=5.0, vel_sigma=1.0,
                 accel_noise=0.05, bias_walk=0.001):
        self.x = np.concatenate([pos0, vel0, np.zeros(3)])
        self.P = np.diag([pos_sigma**2] * 3 + [vel_sigma**2] * 3 + [0.05**2] * 3)
        self.accel_noise = accel_noise
        self.bias_walk = bias_walk

    def predict(self, accel_body, dcm_body_to_ned, dt):
        a_ned = dcm_body_to_ned @ accel_body + G_NED - self.x[6:9]
        self.x[0:3] += self.x[3:6] * dt + 0.5 * a_ned * dt**2
        self.x[3:6] += a_ned * dt

        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -np.eye(3) * dt
        Q = np.zeros((9, 9))
        Q[3:6, 3:6] = np.eye(3) * (self.accel_noise**2) * dt
        Q[6:9, 6:9] = np.eye(3) * (self.bias_walk**2) * dt
        self.P = F @ self.P @ F.T + Q

    def _update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_gps(self, pos, vel, pos_noise, vel_noise):
        H = np.zeros((6, 9))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        R = np.diag([pos_noise**2] * 3 + [vel_noise**2] * 3)
        self._update(np.concatenate([pos, vel]), H, R)

    def update_baro(self, altitude, noise):
        H = np.zeros((1, 9))
        H[0, 2] = -1.0  # altitude = -pd
        self._update(np.array([altitude]), H, np.array([[noise**2]]))
