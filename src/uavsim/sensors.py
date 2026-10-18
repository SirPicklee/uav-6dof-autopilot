"""Sensor models: IMU, GPS and barometer with noise and bias.

Noise levels are representative of small-UAV sensors; they are estimates, not
taken from a specific datasheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IMU:
    """Accelerometer and gyro: white noise plus a slowly varying (random-walk) bias."""

    rate_hz: float = 100.0
    accel_noise: float = 0.05        # m/s^2, white noise (1 sigma)
    gyro_noise: float = 0.002        # rad/s, white noise (1 sigma)
    accel_bias_walk: float = 0.001   # m/s^2 per sqrt(s)
    gyro_bias_walk: float = 0.00005  # rad/s per sqrt(s)
    seed: int = 1
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def measure(self, accel_true: np.ndarray, gyro_true: np.ndarray):
        dt = 1.0 / self.rate_hz
        self.accel_bias = self.accel_bias + self.rng.normal(0.0, self.accel_bias_walk * np.sqrt(dt), 3)
        self.gyro_bias = self.gyro_bias + self.rng.normal(0.0, self.gyro_bias_walk * np.sqrt(dt), 3)
        accel = accel_true + self.accel_bias + self.rng.normal(0.0, self.accel_noise, 3)
        gyro = gyro_true + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise, 3)
        return accel, gyro


@dataclass
class GPS:
    """Position (NED, m) and velocity (NED, m/s) with white noise."""

    rate_hz: float = 5.0
    pos_noise: float = 2.5
    vel_noise: float = 0.1
    seed: int = 2

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def measure(self, pos_true: np.ndarray, vel_true: np.ndarray):
        return (pos_true + self.rng.normal(0.0, self.pos_noise, 3),
                vel_true + self.rng.normal(0.0, self.vel_noise, 3))


@dataclass
class Barometer:
    """Altitude (m) with white noise."""

    rate_hz: float = 25.0
    noise: float = 1.5
    seed: int = 3

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def measure(self, altitude_true: float) -> float:
        return float(altitude_true + self.rng.normal(0.0, self.noise))
