"""Numerical linearization of the 6-DoF model about a trim point.

Reduced 12-state vector (Euler angles instead of quaternion, so there is no
redundant state):
    x = [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
Inputs:
    u = [elevator, aileron, rudder, throttle]
"""

from __future__ import annotations

import numpy as np

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
from uavsim.aircraft import AircraftParams, DEFAULT_AIRCRAFT
from uavsim.dynamics import TrimResult, state_derivative

LON_IDX = [3, 5, 10, 7]     # u, w, q, theta
LAT_IDX = [4, 9, 11, 6]     # v, p, r, phi
LON_NAMES = ["u", "w", "q", "theta"]
LAT_NAMES = ["v", "p", "r", "phi"]


def to_reduced(y13: np.ndarray) -> np.ndarray:
    phi, theta, psi = quat.to_euler(y13[6:10])
    return np.concatenate([y13[0:6], [phi, theta, psi], y13[10:13]])


def f_reduced(x: np.ndarray, u: np.ndarray, params: AircraftParams) -> np.ndarray:
    phi, theta, psi = x[6:9]
    p, q, r = x[9:12]
    y13 = np.concatenate([x[0:6], quat.from_euler(phi, theta, psi), x[9:12]])
    controls = Controls(elevator=u[0], aileron=u[1], rudder=u[2], throttle=u[3])
    d = state_derivative(y13, controls, params)

    s, c = np.sin(phi), np.cos(phi)
    phi_dot = p + np.tan(theta) * (q * s + r * c)
    theta_dot = q * c - r * s
    psi_dot = (q * s + r * c) / np.cos(theta)
    return np.concatenate([d[0:6], [phi_dot, theta_dot, psi_dot], d[10:13]])


def linearize(trim: TrimResult, params: AircraftParams = DEFAULT_AIRCRAFT):
    """Return (A, B, x0, u0) with central-difference Jacobians."""
    x0 = to_reduced(trim.state())
    u0 = np.array([trim.elevator, 0.0, 0.0, trim.throttle])

    n, m = len(x0), len(u0)
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    for i in range(n):
        h = 1e-6 * max(1.0, abs(x0[i]))
        dx = np.zeros(n)
        dx[i] = h
        A[:, i] = (f_reduced(x0 + dx, u0, params) - f_reduced(x0 - dx, u0, params)) / (2 * h)
    for j in range(m):
        h = 1e-6
        du = np.zeros(m)
        du[j] = h
        B[:, j] = (f_reduced(x0, u0 + du, params) - f_reduced(x0, u0 - du, params)) / (2 * h)
    return A, B, x0, u0


def longitudinal(A: np.ndarray, B: np.ndarray):
    """Longitudinal subsystem: states [u, w, q, theta], inputs [elevator, throttle]."""
    return A[np.ix_(LON_IDX, LON_IDX)], B[np.ix_(LON_IDX, [0, 3])]


def lateral(A: np.ndarray, B: np.ndarray):
    """Lateral subsystem: states [v, p, r, phi], inputs [aileron, rudder]."""
    return A[np.ix_(LAT_IDX, LAT_IDX)], B[np.ix_(LAT_IDX, [1, 2])]


def describe_eigenvalues(A: np.ndarray) -> list[dict]:
    """Describe each eigenvalue (one entry per complex pair)."""
    out = []
    for lam in np.linalg.eigvals(A):
        if abs(lam.imag) < 1e-9:
            out.append({"lambda": lam, "type": "real",
                        "tau": (-1.0 / lam.real) if lam.real != 0 else np.inf})
        elif lam.imag > 0:
            wn = abs(lam)
            out.append({"lambda": lam, "type": "oscillatory", "wn": wn,
                        "zeta": -lam.real / wn, "period": 2 * np.pi / lam.imag})
    return out
