"""Aerodynamic forces and moments for a fixed-wing UAV (body axes).

Body axes: x forward, y right, z down.
Wind axes are related to body axes by angle of attack (alpha) and
sideslip (beta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavsim.aircraft import AircraftParams
from uavsim.atmosphere import atmosphere, dynamic_pressure


@dataclass(frozen=True)
class Controls:
    """Control inputs. Surface deflections in rad, throttle in [0, 1]."""

    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0


@dataclass(frozen=True)
class AeroResult:
    """Output of the aerodynamic model."""

    force_body: np.ndarray    # N, [Fx, Fy, Fz] incl. thrust, excl. gravity
    moment_body: np.ndarray   # N m, [L, M, N]
    alpha: float              # rad
    beta: float               # rad
    airspeed: float           # m/s
    dynamic_pressure: float   # Pa
    lift: float               # N
    drag: float               # N


def air_data(v_body_rel_air: np.ndarray) -> tuple[float, float, float]:
    """Return (airspeed, alpha, beta) from body-axis air-relative velocity."""
    u, v, w = v_body_rel_air
    vt = float(np.sqrt(u * u + v * v + w * w))
    if vt < 1e-6:
        return 0.0, 0.0, 0.0
    alpha = float(np.arctan2(w, u))
    beta = float(np.arcsin(np.clip(v / vt, -1.0, 1.0)))
    return vt, alpha, beta


def thrust(params: AircraftParams, throttle: float, airspeed: float, density: float) -> float:
    """Simple propeller thrust model: decays with airspeed, scales with density."""
    throttle = float(np.clip(throttle, 0.0, 1.0))
    decay = max(0.0, 1.0 - params.thrust_decay * airspeed)
    return params.max_thrust_sl * throttle * decay * (density / 1.225)


def aero_forces_moments(
    params: AircraftParams,
    v_body_rel_air: np.ndarray,
    omega_body: np.ndarray,
    controls: Controls,
    altitude_m: float,
) -> AeroResult:
    """Compute total aerodynamic + thrust force and moment in body axes."""
    atm = atmosphere(float(np.clip(altitude_m, 0.0, 20000.0)))
    vt, alpha, beta = air_data(v_body_rel_air)
    qbar = dynamic_pressure(atm.density, vt)
    p, q, r = omega_body

    S, b, c = params.wing_area, params.wing_span, params.mean_chord

    if vt > 1e-3:
        p_hat = p * b / (2.0 * vt)
        q_hat = q * c / (2.0 * vt)
        r_hat = r * b / (2.0 * vt)
    else:
        p_hat = q_hat = r_hat = 0.0

    # Lift with soft stall: linear region clipped at +-cl_max
    cl_lin = (
        params.cl0 + params.cl_alpha * alpha
        + params.cl_q * q_hat + params.cl_de * controls.elevator
    )
    cl = float(np.clip(cl_lin, -params.cl_max, params.cl_max))

    # Drag polar plus a simple post-stall penalty
    cd = params.cd0 + params.induced_drag_k * cl**2
    if abs(cl_lin) > params.cl_max:
        cd += 0.5 * (abs(cl_lin) - params.cl_max) ** 2

    cy = (
        params.cy_beta * beta + params.cy_dr * controls.rudder
    )

    lift = qbar * S * cl
    drag = qbar * S * cd
    side = qbar * S * cy

    # Wind axes -> body axes (lift/drag in the x-z plane, side force along y)
    ca, sa = np.cos(alpha), np.sin(alpha)
    fx_aero = -drag * ca + lift * sa
    fz_aero = -drag * sa - lift * ca
    fy_aero = side

    t = thrust(params, controls.throttle, vt, atm.density)

    force = np.array([fx_aero + t, fy_aero, fz_aero])

    cl_roll = (
        params.cl_beta * beta + params.cl_p * p_hat + params.cl_r * r_hat
        + params.cl_da * controls.aileron
    )
    cm = (
        params.cm0 + params.cm_alpha * alpha + params.cm_q * q_hat
        + params.cm_de * controls.elevator
    )
    cn = (
        params.cn_beta * beta + params.cn_p * p_hat + params.cn_r * r_hat
        + params.cn_da * controls.aileron + params.cn_dr * controls.rudder
    )

    moment = np.array([
        qbar * S * b * cl_roll,
        qbar * S * c * cm,
        qbar * S * b * cn,
    ])

    return AeroResult(
        force_body=force,
        moment_body=moment,
        alpha=alpha,
        beta=beta,
        airspeed=vt,
        dynamic_pressure=qbar,
        lift=lift,
        drag=drag,
    )
