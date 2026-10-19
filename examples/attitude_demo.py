"""Attitude estimation error of the Mahony filter during the square mission."""

import numpy as np

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
from uavsim.atmosphere import G0
from uavsim.attitude import MahonyFilter
from uavsim.autopilot import Autopilot
from uavsim.dynamics import (
    constant_controls, make_dynamics, make_state, post_step,
    state_derivative, trim_level_flight,
)
from uavsim.guidance import L1Navigator
from uavsim.integrators import rk4_step
from uavsim.lateral import LateralAutopilot
from uavsim.sensors import IMU

ALT, SPD, DT = 1000.0, 50.0, 0.01
WAYPOINTS = np.array([[2000.0, 0.0], [2000.0, 2000.0], [0.0, 2000.0], [0.0, 0.0]])
KP = 1.0


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def main():
    trim = trim_level_flight(SPD, ALT)
    lon = Autopilot(trim, ALT, SPD)
    lat = LateralAutopilot(target_heading=0.0)
    nav = L1Navigator(WAYPOINTS)
    y = make_state(SPD, trim.alpha, trim.pitch, ALT)
    imu = IMU()
    est = MahonyFilter(y[6:10].copy(), kp=KP)

    rows = []
    for i in range(int(150.0 / DT)):
        t = i * DT
        if nav.done:
            break
        lat.target_heading = nav.heading_command(y[0], y[1], t)
        c_lon = lon.update(y, DT)
        aileron, rudder = lat.update(y, DT)
        c = Controls(elevator=c_lon.elevator, throttle=c_lon.throttle, aileron=aileron, rudder=rudder)
        dcm = quat.to_dcm(y[6:10])
        d = state_derivative(y, c)
        accel_true = d[3:6] + np.cross(y[10:13], y[3:6]) - dcm.T @ np.array([0.0, 0.0, G0])
        accel_meas, gyro_meas = imu.measure(accel_true, y[10:13])
        q_est = est.update(gyro_meas, accel_meas, DT, vel_body=np.array([SPD, 0.0, 0.0]))

        dq = quat.multiply(quat.conjugate(y[6:10]), q_est)
        total = np.degrees(2.0 * np.arccos(np.clip(abs(dq[0]), 0.0, 1.0)))
        e_true = np.array(quat.to_euler(y[6:10]))
        e_est = np.array(quat.to_euler(q_est))
        err = np.degrees(wrap(e_est - e_true))
        rows.append([t, np.degrees(e_true[0]), total, err[0], err[1], err[2]])
        y = post_step(rk4_step(make_dynamics(constant_controls(c)), t, y, DT))

    r = np.array(rows)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")
    straight = np.abs(r[:, 1]) < 5.0
    turning = np.abs(r[:, 1]) > 20.0
    print(f"Mahony kp={KP}, samples {len(r)}, duration {r[-1, 0]:.1f} s (starts from true attitude)")
    print(f"Total attitude error RMS: {rms(r[:, 2]):.2f} deg | max {r[:, 2].max():.2f} deg")
    print(f"  roll {rms(r[:, 3]):.2f} | pitch {rms(r[:, 4]):.2f} | yaw {rms(r[:, 5]):.2f} deg RMS")
    print(f"Wings-level (|bank|<5 deg, {straight.sum()} samples):  total RMS {rms(r[straight, 2]):.2f} deg")
    print(f"Turning (|bank|>20 deg, {turning.sum()} samples): total RMS {rms(r[turning, 2]):.2f} deg")


if __name__ == "__main__":
    main()

