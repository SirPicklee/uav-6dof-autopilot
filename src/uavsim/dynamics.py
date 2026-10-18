"""6-DoF rigid-body equations of motion for a fixed-wing UAV.

State vector (13): [pn, pe, pd, u, v, w, qw, qx, qy, qz, p, q, r]
    pn, pe, pd : position in NED frame (m), pd positive down
    u, v, w    : body-axis velocity relative to the ground (m/s)
    qw..qz     : attitude quaternion, body -> NED
    p, q, r    : body angular rates (rad/s)

Wind is specified in the NED frame (m/s) and affects aerodynamics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import least_squares

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls, aero_forces_moments
from uavsim.aircraft import AircraftParams, DEFAULT_AIRCRAFT
from uavsim.atmosphere import G0

IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_QUAT = slice(6, 10)
IDX_RATE = slice(10, 13)

ControlLaw = Callable[[float, np.ndarray], Controls]
WindModel = Callable[[float, np.ndarray], np.ndarray]


def constant_controls(controls: Controls) -> ControlLaw:
    return lambda t, y: controls


def no_wind(t: float, y: np.ndarray) -> np.ndarray:
    return np.zeros(3)


def state_derivative(
    y: np.ndarray,
    controls: Controls,
    params: AircraftParams = DEFAULT_AIRCRAFT,
    wind_ned: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Time derivative of the 13-element state vector."""
    v_b = y[IDX_VEL]
    q = y[IDX_QUAT]
    omega = y[IDX_RATE]

    dcm = quat.to_dcm(q)  # body -> NED
    altitude = -y[2]

    wind_b = dcm.T @ (np.zeros(3) if wind_ned is None else wind_ned)
    v_air_b = v_b - wind_b

    res = aero_forces_moments(params, v_air_b, omega, controls, altitude)

    gravity_b = dcm.T @ np.array([0.0, 0.0, params.mass * G0])
    accel_b = (res.force_body + gravity_b) / params.mass - np.cross(omega, v_b)

    inertia = params.inertia
    omega_dot = np.linalg.solve(inertia, res.moment_body - np.cross(omega, inertia @ omega))

    pos_dot = dcm @ v_b
    q_dot = quat.derivative(q, omega)

    return np.concatenate([pos_dot, accel_b, q_dot, omega_dot])


def make_dynamics(
    control_law: ControlLaw,
    params: AircraftParams = DEFAULT_AIRCRAFT,
    wind_model: WindModel = no_wind,
):
    """Build f(t, y) suitable for uavsim.integrators.integrate."""

    def f(t: float, y: np.ndarray) -> np.ndarray:
        return state_derivative(y, control_law(t, y), params, wind_model(t, y))

    return f


def post_step(y: np.ndarray) -> np.ndarray:
    """Renormalize the quaternion after every integration step."""
    y = y.copy()
    y[IDX_QUAT] = quat.normalize(y[IDX_QUAT])
    return y


def make_state(
    airspeed: float,
    alpha: float,
    pitch: float,
    altitude: float,
    roll: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """Build a state with zero sideslip and zero angular rates."""
    u = airspeed * np.cos(alpha)
    w = airspeed * np.sin(alpha)
    q = quat.from_euler(roll, pitch, yaw)
    return np.concatenate([[0.0, 0.0, -altitude], [u, 0.0, w], q, [0.0, 0.0, 0.0]])


@dataclass(frozen=True)
class TrimResult:
    airspeed: float
    altitude: float
    alpha: float
    pitch: float
    elevator: float
    throttle: float
    residual: float

    @property
    def controls(self) -> Controls:
        return Controls(elevator=self.elevator, throttle=self.throttle)

    def state(self) -> np.ndarray:
        return make_state(self.airspeed, self.alpha, self.pitch, self.altitude)


def trim_level_flight(
    airspeed: float,
    altitude: float,
    params: AircraftParams = DEFAULT_AIRCRAFT,
) -> TrimResult:
    """Find alpha, elevator and throttle for steady level flight (gamma = 0)."""

    def residual(x):
        alpha, elevator, throttle = x
        y = make_state(airspeed, alpha, alpha, altitude)  # pitch = alpha when gamma = 0
        d = state_derivative(y, Controls(elevator=elevator, throttle=throttle), params)
        # u_dot, w_dot (scaled by mass) and q_dot must all vanish
        return np.array([d[3], d[5], d[11]])

    sol = least_squares(
        residual,
        x0=np.array([0.05, 0.0, 0.5]),
        bounds=([-0.2, -0.5, 0.0], [0.3, 0.5, 1.0]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    alpha, elevator, throttle = sol.x
    return TrimResult(
        airspeed=airspeed,
        altitude=altitude,
        alpha=float(alpha),
        pitch=float(alpha),
        elevator=float(elevator),
        throttle=float(throttle),
        residual=float(np.linalg.norm(sol.fun)),
    )

