"""Autonomous waypoint mission: closed square, 2 km legs, altitude and airspeed hold."""

import numpy as np
import matplotlib.pyplot as plt

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
from uavsim.autopilot import Autopilot
from uavsim.dynamics import (
    constant_controls, make_dynamics, make_state, post_step, trim_level_flight,
)
from uavsim.guidance import WaypointNavigator
from uavsim.integrators import rk4_step
from uavsim.lateral import LateralAutopilot

ALT = 1000.0
SPD = 50.0
DT = 0.02
T_MAX = 400.0; T_EXTRA = 40.0
WAYPOINTS = np.array([[2000.0, 0.0], [2000.0, 2000.0], [0.0, 2000.0], [0.0, 0.0]])
CAPTURE = 300.0


def main():
    trim = trim_level_flight(SPD, ALT)
    lon = Autopilot(trim, ALT, SPD)
    lat = LateralAutopilot(target_heading=0.0)
    nav = WaypointNavigator(WAYPOINTS, capture_radius=CAPTURE)
    y = make_state(SPD, trim.alpha, trim.pitch, ALT)

    keys = ("t", "pn", "pe", "alt", "spd", "phi", "psi", "cmd")
    log = {k: [] for k in keys}

    for i in range(int(T_MAX / DT)):
        t = i * DT
        if nav.done and t > nav.captures[-1][1] + T_EXTRA:
            break
        phi, _, psi = quat.to_euler(y[6:10])
        lat.target_heading = nav.heading_command(y[0], y[1], t)
        c_lon = lon.update(y, DT)
        aileron, rudder = lat.update(y, DT)
        c = Controls(elevator=c_lon.elevator, throttle=c_lon.throttle,
                     aileron=aileron, rudder=rudder)

        log["t"].append(t)
        log["pn"].append(y[0])
        log["pe"].append(y[1])
        log["alt"].append(-y[2])
        log["spd"].append(float(np.linalg.norm(y[3:6])))
        log["phi"].append(np.degrees(phi))
        log["psi"].append(psi)
        log["cmd"].append(lat.target_heading)

        f = make_dynamics(constant_controls(c))
        y = post_step(rk4_step(f, t, y, DT))

    d = {k: np.array(v) for k, v in log.items()}

    print(f"Mission {'completed' if nav.done else 'NOT completed'} in {nav.captures[-1][1]:.1f} s")
    for idx, tc, miss in nav.captures:
        print(f"  WP{idx + 1} {WAYPOINTS[idx].astype(int).tolist()} | captured at {tc:6.1f} s | closest approach {miss:6.1f} m")
    pts = np.column_stack([d["pn"], d["pe"]])
    starts = np.vstack([[0.0, 0.0], WAYPOINTS[:-1]])
    print("Waypoint accuracy (true closest approach, cross-track on leg):")
    for k, (wp, s) in enumerate(zip(WAYPOINTS, starts)):
        t0 = 0.0 if k == 0 else nav.captures[k - 1][1]
        t1 = nav.captures[k][1]
        after = d["t"] >= t0
        dist = np.hypot(pts[after, 0] - wp[0], pts[after, 1] - wp[1])
        seg = (d["t"] >= t0) & (d["t"] <= t1)
        leg = wp - s
        length = np.hypot(leg[0], leg[1])
        rel = pts[seg] - s
        xt = (rel[:, 0] * leg[1] - rel[:, 1] * leg[0]) / length
        print(f"  WP{k + 1} | true closest approach {dist.min():6.1f} m | max cross-track on leg {np.abs(xt).max():6.1f} m")
    print(f"Max bank angle:  {np.abs(d['phi']).max():.1f} deg")
    print(f"Altitude range:  {d['alt'].min():.1f} .. {d['alt'].max():.1f} m")
    print(f"Airspeed range: {d['spd'].min():.1f} .. {d['spd'].max():.1f} m/s")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2])
    track = fig.add_subplot(gs[:, 1])
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    a2 = fig.add_subplot(gs[2, 0], sharex=a0)

    sc = track.scatter(d["pe"], d["pn"], c=d["t"], s=4, cmap="viridis")
    start = np.array([[0.0, 0.0]])
    path = np.vstack([start, WAYPOINTS])
    track.plot(path[:, 1], path[:, 0], "k--", lw=1, label="planned route")
    for k, (wn, we) in enumerate(WAYPOINTS):
        track.add_patch(plt.Circle((we, wn), CAPTURE, fill=False, color="tab:red", alpha=0.5))
        track.annotate(f"WP{k + 1}", (we, wn), textcoords="offset points", xytext=(6, 6))
    track.plot(0, 0, "go", label="start")
    track.set_aspect("equal")
    track.set_xlabel("East (m)")
    track.set_ylabel("North (m)")
    track.set_title("Autonomous waypoint mission (ground track)")
    track.grid(True)
    track.legend(loc="upper right")
    fig.colorbar(sc, ax=track, label="Time (s)", shrink=0.7)

    a0.plot(d["t"], np.degrees(np.unwrap(d["psi"])), label="heading")
    a0.plot(d["t"], np.degrees(np.unwrap(d["cmd"])), "k--", label="guidance command")
    a0.set_ylabel("Heading (deg)")
    a0.set_title("Guidance and control")
    a1.plot(d["t"], d["phi"])
    a1.set_ylabel("Bank (deg)")
    a2.plot(d["t"], d["alt"] - ALT, label="altitude error (m)")
    a2.plot(d["t"], d["spd"] - SPD, label="airspeed error (m/s)")
    a2.set_ylabel("Error")
    a2.set_xlabel("Time (s)")
    for a in (a0, a1, a2):
        a.grid(True)
    a0.legend(fontsize=8)
    a2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("examples/waypoint_mission.png", dpi=150)
    print("Saved examples/waypoint_mission.png")
    plt.show()


if __name__ == "__main__":
    main()

