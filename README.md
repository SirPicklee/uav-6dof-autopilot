# uav-6dof-autopilot

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




## PID vs. LQI (nonlinear 6-DoF, 50 m climb)

Both controllers start 50 m below the target at trim airspeed (50 m/s, 1000 m) and are
evaluated in the full nonlinear model. The LQI is a longitudinal LQR augmented with an
airspeed integrator (the altitude state is the altitude error itself, not its integral).
Weights: `Q = diag(0.1, 1, 1, 10, Q_h, 0.5)`, `R = diag(100, 1)`.

| Controller | Settling (2 m) | Min airspeed | Max alpha |
|---|---|---|---|
| PID (`ki = 0`) | 39.9 s | 46.7 m/s | 4.65 deg |
| LQI, Q_h = 0.02 | 56.3 s | 48.8 m/s | 2.49 deg |
| LQI, Q_h = 0.05 | 57.5 s | 48.2 m/s | 3.16 deg |
| LQI, Q_h = 0.10 | 58.9 s | 47.4 m/s | 3.89 deg |
| LQI, Q_h = 0.20 | 59.0 s | 46.4 m/s | 4.87 deg |
| LQI, Q_h = 0.50 | 48.4 s | 44.5 m/s | 6.61 deg |
| LQI, Q_h = 1.00 | 43.5 s | 43.1 m/s | 8.30 deg |
| LQI, Q_h = 2.00 | 32.8 s | 42.2 m/s | 10.26 deg |

![LQI weight sweep](examples/lqi_weight_sweep.png)

**Finding:** in this scenario and with these weights, PID gives a better trade-off than LQI.
LQI with a similar speed dip and alpha (Q_h = 0.2) is much slower (59 s); LQI with a similar
settling time (Q_h = 1.0) has a deeper speed dip and higher alpha. This is not a general
statement about LQR: R, the other Q entries and the altitude-error clipping (10 m) were not
explored systematically. Settling time is not monotonic in Q_h for small weights; the cause
was not investigated.

Linear analysis at trim (50 m/s, 1000 m): phugoid period 26.2 s (zeta 0.10), short period
wn 3.85 rad/s (zeta 0.52), Dutch roll wn 3.0 rad/s (zeta 0.26), roll mode tau 0.12 s,
spiral tau 69 s (stable). The simulated phugoid period matches the eigenvalue analysis.
