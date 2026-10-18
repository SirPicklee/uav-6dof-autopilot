# uav-6dof-autopilot

[![tests](https://github.com/SirPicklee/uav-6dof-autopilot/actions/workflows/tests.yml/badge.svg)](https://github.com/SirPicklee/uav-6dof-autopilot/actions/workflows/tests.yml)

[![tests](https://github.com/SirPicklee/uav-6dof-autopilot/actions/workflows/tests.yml/badge.svg)](https://github.com/SirPicklee/uav-6dof-autopilot/actions/workflows/tests.yml)

A 6-DoF fixed-wing UAV flight dynamics simulator with a PID autopilot, written in Python.
Built from first principles: quaternion kinematics, RK4 integration, ISA atmosphere,
aerodynamic force/moment model, trim solver, and closed-loop altitude/airspeed hold.

> **Disclaimer:** The aircraft is a *TB2-class representative UAV*. Span, wing area and mass
> are in the range of open-source figures; aerodynamic coefficients are **estimated**.
> They are not official manufacturer data.

## Features

- Quaternion attitude library (Hamilton convention, body -> NED)
- Fixed-step Euler and RK4 integrators with post-step hook and stop conditions
- International Standard Atmosphere model (0-20 km, with temperature offset)
- Aerodynamic model: lift with stall saturation, drag polar, side force, roll/pitch/yaw moments, propeller thrust
- 13-state 6-DoF equations of motion with wind input
- Level-flight trim solver (alpha, elevator, throttle)
- PID controller with derivative-on-measurement, derivative filtering and anti-windup
- Altitude-hold and airspeed-hold autopilot (cascaded loops)
- 56 unit tests

## Validation

| Check | Result |
|---|---|
| Torque-free rotation, kinetic energy drift (30 s) | ~7e-13 |
| Torque-free rotation, angular momentum drift (30 s) | ~4e-13 |
| RK4 convergence order (harmonic oscillator) | ~4 |
| ISA atmosphere vs. standard table values | within 0.1% |
| Trim: thrust vs. drag balance at 50 m/s, 1000 m | converged |
| Autopilot: 50 m climb step, overshoot | 0.6 m (1.2%) |
| Autopilot: final altitude error after 120 s | -0.06 m |

### Torque-free rotation (Dzhanibekov effect)

![Torque-free rotation](examples/torque_free_rotation.png)

### Open-loop response: short period and phugoid

![First flight](examples/first_flight.png)

### Closed-loop altitude hold

![Autopilot climb](examples/autopilot_climb.png)

## Installation

```bash
git clone https://github.com/SirPicklee/uav-6dof-autopilot.git
cd uav-6dof-autopilot
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Usage

Run the tests:

```bash
python -m pytest
```

Run the examples (Windows PowerShell):

```powershell
$env:PYTHONPATH="src"
python examples\torque_free_rotation.py
python examples\first_flight.py
python examples\autopilot_climb.py
```

## Project structure

## Roadmap

- [ ] Rejection of wind and turbulence (re-tune integral action)
- [ ] Linearization and modal analysis (short period, phugoid, Dutch roll)
- [ ] LQR controller and comparison with PID
- [ ] Sensor models and Extended Kalman Filter (IMU + GPS + barometer)
- [ ] Waypoint navigation and lateral autopilot
- [ ] Cross-validation against JSBSim / PX4 SITL
- [ ] GitHub Actions CI
- [ ] C++ core

## License

MIT


