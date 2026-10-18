"""Nonlinear closed-loop comparison: PID autopilot vs. LQI controller.

Scenario: start 50 m below the target altitude at trim airspeed and climb.
"""

import numpy as np
import matplotlib.pyplot as plt

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
from uavsim.autopilot import Autopilot
from uavsim.dynamics import (
    constant_controls,
    make_dynamics,
    make_state,
    post_step,
    trim_level_flight,
)
from uavsim.integrators import rk4_step
from uavsim.linearize import linearize, longitudinal
from uavsim.lqr import lqr
from lqi_longitudinal import augment

TARGET_ALT = 1000.0
TARGET_SPEED = 50.0
START_ALT = 950.0
DT = 0.02
T_END = 150.0

E_H_MAX = 10.0                  # altitude error limit fed to the controller (m)
XI_V_MAX = 30.0                 # speed-integrator clamp (anti-windup)
MAX_ELEVATOR = np.radians(20.0)


class LQIController:
    """u = u_trim - K [u, w, q, theta, e_h, xi_v] (deviations from trim)."""

    def __init__(self, trim, K):
        self.trim = trim
        self.K = K
        self.x_trim = trim.state()
        self.xi_v = 0.0

    def update(self, y, dt, airspeed=None):
        altitude = -y[2]
        airspeed = float(np.linalg.norm(y[3:6])) if airspeed is None else float(airspeed)
        theta = quat.to_euler(y[6:10])[1]

        e_h = float(np.clip(TARGET_ALT - altitude, -E_H_MAX, E_H_MAX))
        self.xi_v = float(np.clip(self.xi_v + (TARGET_SPEED - airspeed) * dt, -XI_V_MAX, XI_V_MAX))

        x = np.array([
            y[3] - self.x_trim[3],
            y[5] - self.x_trim[5],
            y[11],
            theta - self.trim.pitch,
            e_h,
            self.xi_v,
        ])
        du = -self.K @ x
        elevator = float(np.clip(self.trim.elevator + du[0], -MAX_ELEVATOR, MAX_ELEVATOR))
        throttle = float(np.clip(self.trim.throttle + du[1], 0.0, 1.0))
        return Controls(elevator=elevator, throttle=throttle)


def simulate(controller, trim):
    y = make_state(TARGET_SPEED, trim.alpha, trim.pitch, START_ALT)
    n = int(T_END / DT)
    log = {k: [] for k in ("t", "alt", "spd", "pitch", "alpha", "elev", "thr")}
    for i in range(n):
        t = i * DT
        c = controller.update(y, DT)
        log["t"].append(t)
        log["alt"].append(-y[2])
        log["spd"].append(float(np.linalg.norm(y[3:6])))
        log["pitch"].append(np.degrees(quat.to_euler(y[6:10])[1]))
        log["alpha"].append(np.degrees(np.arctan2(y[5], y[3])))
        log["elev"].append(np.degrees(c.elevator))
        log["thr"].append(c.throttle)
        f = make_dynamics(constant_controls(c))
        y = post_step(rk4_step(f, t, y, DT))
    return {k: np.array(v) for k, v in log.items()}


def metrics(name, d):
    err = d["alt"] - TARGET_ALT
    outside = np.where(np.abs(err) > 2.0)[0]
    settle = d["t"][outside[-1] + 1] if len(outside) and outside[-1] + 1 < len(d["t"]) else (0.0 if not len(outside) else float("nan"))
    print(f"{name:4s} | overshoot {max(0.0, err.max()):6.2f} m | settling(2 m) {settle:6.1f} s | "
          f"final alt err {err[-1]:7.3f} m | final speed err {d['spd'][-1] - TARGET_SPEED:6.3f} m/s | "
          f"min speed {d['spd'].min():5.1f} m/s | max alpha {d['alpha'].max():5.2f} deg")


def main():
    trim = trim_level_flight(TARGET_SPEED, TARGET_ALT)

    A, B, _, _ = linearize(trim)
    A_lon, B_lon = longitudinal(A, B)
    A_aug, B_aug = augment(A_lon, B_lon, TARGET_SPEED)
    Q = np.diag([0.1, 1.0, 1.0, 10.0, 0.5, 0.5])
    R = np.diag([100.0, 1.0])
    K, _, _ = lqr(A_aug, B_aug, Q, R)

    pid = simulate(Autopilot(trim, TARGET_ALT, TARGET_SPEED), trim)
    lqi = simulate(LQIController(trim, K), trim)

    metrics("PID", pid)
    metrics("LQI", lqi)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for d, label in ((pid, "PID"), (lqi, "LQI")):
        axes[0].plot(d["t"], d["alt"], label=label)
        axes[1].plot(d["t"], d["spd"], label=label)
        axes[2].plot(d["t"], d["pitch"], label=label)
        axes[3].plot(d["t"], d["elev"], label=label)
    axes[0].axhline(TARGET_ALT, color="k", linestyle="--")
    axes[1].axhline(TARGET_SPEED, color="k", linestyle="--")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("PID vs. LQI: 50 m climb in the nonlinear 6-DoF model")
    axes[1].set_ylabel("Airspeed (m/s)")
    axes[2].set_ylabel("Pitch (deg)")
    axes[3].set_ylabel("Elevator (deg)")
    axes[3].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig("examples/lqi_vs_pid.png", dpi=150)
    print("Saved examples/lqi_vs_pid.png")
    plt.show()


if __name__ == "__main__":
    main()




