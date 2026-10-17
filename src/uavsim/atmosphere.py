"""International Standard Atmosphere (ISA) model, 0 to 20 km geopotential altitude.

Layers:
    troposphere   0 - 11 km : linear temperature lapse rate (-6.5 K/km)
    tropopause   11 - 20 km : isothermal at 216.65 K
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sea-level standard constants
T0 = 288.15        # K
P0 = 101325.0      # Pa
RHO0 = 1.225       # kg/m^3
G0 = 9.80665       # m/s^2
R_AIR = 287.05287  # J/(kg K)
GAMMA = 1.4
LAPSE = -0.0065    # K/m, troposphere
H_TROPOPAUSE = 11000.0
H_MAX = 20000.0
T_TROPOPAUSE = T0 + LAPSE * H_TROPOPAUSE
P_TROPOPAUSE = P0 * (T_TROPOPAUSE / T0) ** (-G0 / (LAPSE * R_AIR))


@dataclass(frozen=True)
class AtmosphereState:
    """Atmospheric properties at a given altitude."""

    altitude: float     # m
    temperature: float  # K
    pressure: float     # Pa
    density: float      # kg/m^3
    speed_of_sound: float  # m/s


def atmosphere(altitude_m: float, delta_isa: float = 0.0) -> AtmosphereState:
    """Return ISA properties at `altitude_m` (m).

    Args:
        altitude_m: altitude above sea level in meters (0 to 20000).
        delta_isa: temperature offset from standard day in K (hot/cold day).
            Pressure follows the standard profile; density is corrected
            using the offset temperature.
    """
    h = float(altitude_m)
    if h < 0.0 or h > H_MAX:
        raise ValueError(f"Altitude {h} m is outside the supported range 0-{H_MAX:.0f} m.")

    if h <= H_TROPOPAUSE:
        t_std = T0 + LAPSE * h
        p = P0 * (t_std / T0) ** (-G0 / (LAPSE * R_AIR))
    else:
        t_std = T_TROPOPAUSE
        p = P_TROPOPAUSE * np.exp(-G0 * (h - H_TROPOPAUSE) / (R_AIR * T_TROPOPAUSE))

    t = t_std + delta_isa
    rho = p / (R_AIR * t)
    a = np.sqrt(GAMMA * R_AIR * t)
    return AtmosphereState(altitude=h, temperature=t, pressure=p, density=rho, speed_of_sound=a)


def dynamic_pressure(density: float, airspeed: float) -> float:
    """Dynamic pressure q = 0.5 * rho * V^2 (Pa)."""
    return 0.5 * density * airspeed**2


def mach_number(airspeed: float, speed_of_sound: float) -> float:
    """Mach number from true airspeed and local speed of sound."""
    return airspeed / speed_of_sound
