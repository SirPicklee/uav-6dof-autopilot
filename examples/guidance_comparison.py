"""Compare waypoint guidance laws on the same closed-square mission."""

import numpy as np

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
from uavsim.autopilot import Autopilot
from uavsim.dynamics import (
    constant_controls, make_dynamics, make_state, post_step, trim_level_flight,
)
from uavsim.guidance import L1Navigator, WaypointNavigator
from uavsim.integrators import rk4_step
from uavsim.lateral import LateralAutopilot

ALT, SPD, DT = 1000.0, 50.0, 0.02
WAYPOINTS = np.array([[2000.0, 0.0], [2000.0, 2000.0], [0.0, 2000.0], [0.0, 0.0]])


def fly(nav):
    trim = trim_level_flight(SPD, ALT)
    lon = Autopilot(trim, ALT, SPD)
    lat = LateralAutopilot(target_heading=0.0)
    y = make_state(SPD, trim.alpha, trim.pitch, ALT)
    pts, ts = [], []
    for i in range(int(400.0 / DT)):
        t = i * DT
        if nav.done:
            break
        lat.target_heading = nav.heading_command(y[0], y[1], t)
        c_lon = lon.update(y, DT)
        aileron, rudder = lat.update(y, DT)
        c = Controls(elevator=c_lon.elevator, throttle=c_lon.throttle, aileron=aileron, rudder=rudder)
        pts.append([y[0], y[1]])
        ts.append(t)
        y = post_step(rk4_step(make_dynamics(constant_controls(c)), t, y, DT))
    return np.array(pts), np.array(ts)


def leg_cross_track(pts, ts, nav, k):
    """Max |cross-track| over the points between the leg's start and end projection."""
    a = np.zeros(2) if k == 0 else WAYPOINTS[k - 1]
    b = WAYPOINTS[k]
    u = (b - a) / np.hypot(*(b - a))
    rel = pts - a
    along = rel @ u
    xt = rel[:, 0] * u[1] - rel[:, 1] * u[0]
    t0 = 0.0 if k == 0 else nav.captures[k - 1][1]
    t1 = nav.captures[k][1]
    mask = (ts >= t0) & (ts <= t1) & (along > 0.5 * np.hypot(*(b - a)))
    return float(np.abs(xt[mask]).max()) if mask.any() else float("nan")


def main():
    for name, nav in (
        ("Radius capture (300 m)", WaypointNavigator(WAYPOINTS, capture_radius=300.0)),
        ("L1 leg following", L1Navigator(WAYPOINTS)),
    ):
        pts, ts = fly(nav)
        print(name, f"| completed: {nav.done} | time {ts[-1]:.1f} s")
        for k in range(len(WAYPOINTS)):
            d = float(np.hypot(*(pts - WAYPOINTS[k]).T).min())
            print(f"  WP{k + 1} closest approach {d:6.1f} m | max cross-track on leg {leg_cross_track(pts, ts, nav, k):6.1f} m")


if __name__ == "__main__":
    main()


