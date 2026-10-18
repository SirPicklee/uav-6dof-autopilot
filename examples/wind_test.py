"""Wind rejection test: PID autopilot vs. LQI controller (nonlinear 6-DoF).

Wind steps on at t = 10 s. Controllers see ground-relative speed (y[3:6]).
"""

import numpy as np

from uavsim import quaternion as quat
from uavsim.autopilot import Autopilot
from uavsim.dynamics import (
    constant_controls, make_dynamics, make_state, post_step, trim_level_flight,
)
from uavsim.integrators import rk4_step
from uavsim.linearize import linearize, longitudinal
from uavsim.lqr import lqr
from lqi_longitudinal import augment
from lqi_vs_pid import LQIController, TARGET_ALT, TARGET_SPEED

DT = 0.02
T_END = 120.0
T_WIND = 10.0

CASES = {
    "downdraft 0.5 m/s": np.array([0.0, 0.0, 0.5]),
    "north wind 5 m/s": np.array([5.0, 0.0, 0.0]),
}


def run(controller, trim, wind_vec):
    def wind(t, y):
        return wind_vec if t >= T_WIND else np.zeros(3)

    y = make_state(TARGET_SPEED, trim.alpha, trim.pitch, TARGET_ALT)
    alt_err = []
    for i in range(int(T_END / DT)):
        t = i * DT
        c = controller.update(y, DT, float(np.linalg.norm(y[3:6] - quat.to_dcm(y[6:10]).T @ (wind_vec if t >= T_WIND else np.zeros(3)))))
        f = make_dynamics(constant_controls(c), wind_model=wind)
        alt_err.append(-y[2] - TARGET_ALT)
        y = post_step(rk4_step(f, t, y, DT))
    ground_speed = float(np.linalg.norm(y[3:6]))
    air = y[3:6] - quat.to_dcm(y[6:10]).T @ wind_vec
    return np.array(alt_err), ground_speed, float(np.linalg.norm(air))


def main():
    trim = trim_level_flight(TARGET_SPEED, TARGET_ALT)
    A, B, _, _ = linearize(trim)
    A_lon, B_lon = longitudinal(A, B)
    A_aug, B_aug = augment(A_lon, B_lon, TARGET_SPEED)
    Q = np.diag([0.1, 1.0, 1.0, 10.0, 0.2, 0.5])
    R = np.diag([100.0, 1.0])
    K, _, _ = lqr(A_aug, B_aug, Q, R)

    for name, wind_vec in CASES.items():
        print(name)
        controllers = {
            "PID": Autopilot(trim, TARGET_ALT, TARGET_SPEED),
            "LQI": LQIController(trim, K),
        }
        for label, ctrl in controllers.items():
            err, gs, tas = run(ctrl, trim, wind_vec)
            print(f"  {label} | max |alt err| {np.abs(err).max():6.2f} m | "
                  f"final alt err {err[-1]:7.3f} m | ground speed {gs:6.2f} m/s | airspeed {tas:6.2f} m/s")


if __name__ == "__main__":
    main()


