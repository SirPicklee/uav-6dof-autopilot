"""Aircraft parameter set for a medium-altitude fixed-wing UAV.

NOTE: Values are representative of a TB2-class UAV (open-source figures for
span, wing area and mass; aerodynamic coefficients are estimated). They are
NOT official manufacturer data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AircraftParams:
    """Geometry, mass properties and aerodynamic coefficients (SI units)."""

    name: str = "TB2-class UAV (representative)"

    # Geometry
    wing_area: float = 11.0       # S, m^2
    wing_span: float = 12.0       # b, m
    mean_chord: float = 0.92      # c_bar, m

    # Mass properties
    mass: float = 650.0           # kg
    inertia: np.ndarray = field(
        default_factory=lambda: np.array([
            [1200.0, 0.0, 0.0],
            [0.0, 1500.0, 0.0],
            [0.0, 0.0, 2400.0],
        ])
    )  # kg m^2, body axes (Ixx, Iyy, Izz)

    # Lift
    cl0: float = 0.30
    cl_alpha: float = 5.2         # per rad
    cl_max: float = 1.4
    cl_q: float = 6.0             # per rad (pitch rate, nondimensional)
    cl_de: float = 0.4            # per rad (elevator)

    # Drag
    cd0: float = 0.030
    oswald: float = 0.80
    # Induced drag factor k = 1 / (pi * e * AR) is derived below.

    # Pitch moment
    cm0: float = 0.05
    cm_alpha: float = -1.2        # per rad (static stability)
    cm_q: float = -18.0           # per rad (pitch damping)
    cm_de: float = -1.5           # per rad (elevator)

    # Side force and lateral-directional
    cy_beta: float = -0.40
    cy_dr: float = 0.15
    cl_beta: float = -0.10        # dihedral effect
    cl_p: float = -0.45           # roll damping
    cl_r: float = 0.10
    cl_da: float = 0.20           # aileron
    cn_beta: float = 0.10         # weathercock stability
    cn_p: float = -0.05
    cn_r: float = -0.15           # yaw damping
    cn_da: float = -0.01
    cn_dr: float = -0.08

    # Propulsion (simple propeller model)
    max_thrust_sl: float = 1800.0  # N at sea level, zero airspeed
    thrust_decay: float = 0.012    # per m/s of airspeed

    @property
    def aspect_ratio(self) -> float:
        return self.wing_span**2 / self.wing_area

    @property
    def induced_drag_k(self) -> float:
        return 1.0 / (np.pi * self.oswald * self.aspect_ratio)


DEFAULT_AIRCRAFT = AircraftParams()
