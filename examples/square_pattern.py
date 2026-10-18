"""Square pattern: heading hold + altitude/airspeed hold in the nonlinear 6-DoF model."""

import numpy as np
import matplotlib.pyplot as plt

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
from uavsim.autopilot import Autopilot
from uavsim.dynamics import (
    constant_controls, make_dynamics, make_state, post_step, trim_level_flight,
)
from uavsim.integrators import rk4_step
from uavsim.lateral import LateralAutopilot

ALT = 1000.0
SPD = 50.0
DT = 0.02
T_END = 210.0
CMD_TIMES = np.array([0.0, 10.0, 60.0, 110.0, 160.0])
CMD_HEADINGS = np.radians([0.0, 90.0, 180.0, 270.0, 360.0])


def heading_command(t):
    return float(CMD_HEADINGS[np.searchsorted(CMD_TIMES, t, side="right") - 1])


def main():
    trim = trim_level_flight(SPD, ALT)
    lon = Autopilot(trim, ALT, SPD)
    lat = LateralAutopilot(target_heading=0.0)
    y = make_state(SPD, trim.alpha, trim.pitch, ALT)

    keys = ("t", "pn", "pe", "alt", "spd", "phi", "psi", "beta", "ail", "rud", "cmd", "bank")
    log = {k: [] for k in keys}

    for i in range(int(T_END / DT)):
        t = i * DT
        lat.target_heading = heading_command(t)
        c_lon = lon.update(y, DT)
        aileron, rudder = lat.update(y, DT)
        c = Controls(elevator=c_lon.elevator, throttle=c_lon.throttle,
                     aileron=aileron, rudder=rudder)

        phi, _, psi = quat.to_euler(y[6:10])
        vt = float(np.linalg.norm(y[3:6]))
        log["t"].append(t)
        log["pn"].append(y[0])
        log["pe"].append(y[1])
        log["alt"].append(-y[2])
        log["spd"].append(vt)
        log["phi"].append(np.degrees(phi))
        log["psi"].append(psi)
        log["beta"].append(np.degrees(np.arcsin(np.clip(y[4] / vt, -1.0, 1.0))))
        log["ail"].append(np.degrees(aileron))
        log["rud"].append(np.degrees(rudder))
        log["cmd"].append(lat.target_heading)
        log["bank"].append(np.degrees(lat.bank_cmd))

        f = make_dynamics(constant_controls(c))
        y = post_step(rk4_step(f, t, y, DT))

    d = {k: np.array(v) for k, v in log.items()}
    psi_u = np.degrees(np.unwrap(d["psi"]))
    cmd_deg = np.degrees(d["cmd"])
    err = cmd_deg - psi_u

    print("Heading step response (target -> time to stay within 2 deg, overshoot):")
    for k in range(1, len(CMD_TIMES)):
        t0 = CMD_TIMES[k]
        t1 = CMD_TIMES[k + 1] if k + 1 < len(CMD_TIMES) else T_END
        seg = (d["t"] >= t0) & (d["t"] < t1)
        e = err[seg]
        tt = d["t"][seg]
        outside = np.where(np.abs(e) > 2.0)[0]
        settle = (tt[outside[-1]] - t0) if len(outside) else 0.0
        overshoot = max(0.0, -e.min())
        print(f"  {np.degrees(CMD_HEADINGS[k]):5.0f} deg | settle {settle:5.1f} s | overshoot {overshoot:5.2f} deg")

    print(f"Max bank angle:        {np.abs(d['phi']).max():.1f} deg")
    print(f"Max |sideslip|:        {np.abs(d['beta']).max():.2f} deg")
    print(f"Altitude range:        {d['alt'].min():.1f} .. {d['alt'].max():.1f} m")
    print(f"Airspeed range:        {d['spd'].min():.1f} .. {d['spd'].max():.1f} m/s")
    print(f"Final position error:  north {d['pn'][-1]:.0f} m, east {d['pe'][-1]:.0f} m")

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.4, 1])
    ax = [fig.add_subplot(gs[0, 0])]
    for r in range(1, 4):
        ax.append(fig.add_subplot(gs[r, 0], sharex=ax[0]))
    track = fig.add_subplot(gs[:, 1])

    ax[0].plot(d["t"], cmd_deg, "k--", label="command")
    ax[0].plot(d["t"], psi_u, label="heading")
    ax[0].set_ylabel("Heading (deg)")
    ax[0].set_title("Lateral autopilot: square pattern with altitude and airspeed hold")
    ax[1].plot(d["t"], d["bank"], "k--", label="bank command")
    ax[1].plot(d["t"], d["phi"], label="roll")
    ax[1].set_ylabel("Bank (deg)")
    ax[2].plot(d["t"], d["alt"] - ALT, label="altitude error (m)")
    ax[2].plot(d["t"], d["spd"] - SPD, label="airspeed error (m/s)")
    ax[2].set_ylabel("Error")
    ax[3].plot(d["t"], d["ail"], label="aileron (deg)")
    ax[3].plot(d["t"], d["rud"], label="rudder (deg)")
    ax[3].plot(d["t"], d["beta"], label="sideslip (deg)")
    ax[3].set_ylabel("Surfaces / sideslip")
    ax[3].set_xlabel("Time (s)")
    for a in ax:
        a.grid(True)
        a.legend(loc="upper left", fontsize=8)

    sc = track.scatter(d["pe"], d["pn"], c=d["t"], s=3, cmap="viridis")
    track.plot(d["pe"][0], d["pn"][0], "go", label="start")
    track.plot(d["pe"][-1], d["pn"][-1], "rs", label="end")
    track.set_aspect("equal")
    track.set_xlabel("East (m)")
    track.set_ylabel("North (m)")
    track.set_title("Ground track")
    track.grid(True)
    track.legend()
    fig.colorbar(sc, ax=track, label="Time (s)", shrink=0.7)

    fig.tight_layout()
    fig.savefig("examples/square_pattern.png", dpi=150)
    print("Saved examples/square_pattern.png")
    plt.show()


if __name__ == "__main__":
    main()
