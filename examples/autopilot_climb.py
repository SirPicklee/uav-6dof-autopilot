"""Closed-loop altitude hold: UAV starts 50 m below target and climbs to it."""

import numpy as np
import matplotlib.pyplot as plt

from uavsim import quaternion as quat
from uavsim.autopilot import Autopilot
from uavsim.dynamics import (
    constant_controls,
    make_dynamics,
    make_state,
    post_step,
    trim_level_flight,
)
from uavsim.integrators import rk4_step

TARGET_ALT = 1000.0
TARGET_SPEED = 50.0
START_ALT = 950.0
DT = 0.02
T_END = 120.0


def main():
    trim = trim_level_flight(TARGET_SPEED, TARGET_ALT)
    ap = Autopilot(trim, TARGET_ALT, TARGET_SPEED)
    y = make_state(TARGET_SPEED, trim.alpha, trim.pitch, START_ALT)

    n = int(T_END / DT)
    t_log, y_log, elev_log, thr_log = [], [], [], []

    for i in range(n):
        t = i * DT
        controls = ap.update(y, DT)
        f = make_dynamics(constant_controls(controls))
        t_log.append(t)
        y_log.append(y.copy())
        elev_log.append(np.degrees(controls.elevator))
        thr_log.append(controls.throttle)
        y = post_step(rk4_step(f, t, y, DT))

    t = np.array(t_log)
    Y = np.array(y_log)
    altitude = -Y[:, 2]
    airspeed = np.linalg.norm(Y[:, 3:6], axis=1)
    pitch = np.degrees([quat.to_euler(q)[1] for q in Y[:, 6:10]])

    print(f"Max altitude:  {altitude.max():.1f} m (overshoot {altitude.max() - TARGET_ALT:.1f} m)")
    print(f"Final altitude: {altitude[-1]:.1f} m (error {altitude[-1] - TARGET_ALT:.2f} m)")
    print(f"Final airspeed: {airspeed[-1]:.2f} m/s")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(t, altitude)
    axes[0].axhline(TARGET_ALT, color="k", linestyle="--", label="target")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("Autopilot: climb from 950 m to 1000 m and hold")
    axes[0].legend()
    axes[1].plot(t, airspeed)
    axes[1].axhline(TARGET_SPEED, color="k", linestyle="--")
    axes[1].set_ylabel("Airspeed (m/s)")
    axes[2].plot(t, pitch)
    axes[2].set_ylabel("Pitch (deg)")
    axes[3].plot(t, elev_log, label="elevator (deg)")
    axes[3].plot(t, np.array(thr_log) * 10.0, label="throttle x10")
    axes[3].set_ylabel("Controls")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig("examples/autopilot_climb.png", dpi=150)
    print("Saved examples/autopilot_climb.png")
    plt.show()


if __name__ == "__main__":
    main()
