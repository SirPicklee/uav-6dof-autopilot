"""PID vs. LQI with and without actuator dynamics (50 m climb, nonlinear 6-DoF)."""

import numpy as np

from uavsim import quaternion as quat
from uavsim.actuators import Actuator
from uavsim.aerodynamics import Controls
from uavsim.autopilot import Autopilot
from uavsim.dynamics import (
    constant_controls, make_dynamics, make_state, post_step, trim_level_flight,
)
from uavsim.integrators import rk4_step
from uavsim.linearize import linearize, longitudinal
from uavsim.lqr import lqr
from lqi_longitudinal import augment
from lqi_vs_pid import LQIController, TARGET_ALT, TARGET_SPEED, START_ALT

DT = 0.02
T_END = 150.0

CASES = {
    "no actuator": None,
    "nominal servo (tau 0.05 s, 60 deg/s)": dict(tau=0.05, rate=np.radians(60.0), thr_tau=0.3),
    "slow servo (tau 0.15 s, 30 deg/s)": dict(tau=0.15, rate=np.radians(30.0), thr_tau=0.5),
}


def simulate(controller, trim, act):
    y = make_state(TARGET_SPEED, trim.alpha, trim.pitch, START_ALT)
    if act:
        elev = Actuator(tau=act["tau"], rate_limit=act["rate"], pos_limit=np.radians(20.0), x=trim.elevator)
        thr = Actuator(tau=act["thr_tau"], rate_limit=1.0, pos_limit=1.0, x=trim.throttle)
    alt, spd, alpha, elev_cmd = [], [], [], []
    for i in range(int(T_END / DT)):
        t = i * DT
        c = controller.update(y, DT)
        if act:
            c = Controls(elevator=elev.update(c.elevator, DT), throttle=thr.update(c.throttle, DT))
        alt.append(-y[2])
        spd.append(float(np.linalg.norm(y[3:6])))
        alpha.append(np.degrees(np.arctan2(y[5], y[3])))
        elev_cmd.append(np.degrees(c.elevator))
        y = post_step(rk4_step(make_dynamics(constant_controls(c)), t, y, DT))
    return np.array(alt), np.array(spd), np.array(alpha), np.array(elev_cmd)


def metrics(label, alt, spd, alpha):
    err = alt - TARGET_ALT
    outside = np.where(np.abs(err) > 2.0)[0]
    settle = (outside[-1] + 1) * DT if len(outside) else 0.0
    if len(outside) and outside[-1] + 1 >= len(alt):
        settle = float("nan")
    print(f"  {label:4s} | overshoot {max(0.0, err.max()):6.2f} m | settling(2 m) {settle:6.1f} s | "
          f"min speed {spd.min():5.1f} m/s | max alpha {alpha.max():5.2f} deg | final err {err[-1]:7.3f} m")


def main():
    trim = trim_level_flight(TARGET_SPEED, TARGET_ALT)
    A, B, _, _ = linearize(trim)
    A_lon, B_lon = longitudinal(A, B)
    A_aug, B_aug = augment(A_lon, B_lon, TARGET_SPEED)
    Q = np.diag([0.1, 1.0, 1.0, 10.0, 0.5, 0.5])
    R = np.diag([100.0, 1.0])
    K, _, _ = lqr(A_aug, B_aug, Q, R)

    for name, act in CASES.items():
        print(name)
        a, s, al, _ = simulate(Autopilot(trim, TARGET_ALT, TARGET_SPEED), trim, act)
        metrics("PID", a, s, al)
        a, s, al, _ = simulate(LQIController(trim, K), trim, act)
        metrics("LQI", a, s, al)


if __name__ == "__main__":
    main()
