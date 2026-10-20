"""Mahony yaw drift over 600 s: without and with GPS-course yaw correction."""

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
from uavsim.sensors import GPS, IMU

ALT, SPD, DT = 1000.0, 50.0, 0.02
T_END = 600.0
SQUARE = np.array([[2000.0, 0.0], [2000.0, 2000.0], [0.0, 2000.0], [0.0, 0.0]])


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def run(kp_yaw):
    trim = trim_level_flight(SPD, ALT)
    lon = Autopilot(trim, ALT, SPD)
    lat = LateralAutopilot(target_heading=0.0)
    nav = L1Navigator(SQUARE)
    y = make_state(SPD, trim.alpha, trim.pitch, ALT)
    imu, gps = IMU(), GPS()
    gps_every = int(round(1.0 / (gps.rate_hz * DT)))
    est = MahonyFilter(y[6:10].copy(), kp=1.0)

    ts, yaw_err, tot_err = [], [], []
    for i in range(int(T_END / DT)):
        t = i * DT
        if nav.done:
            nav = L1Navigator(SQUARE, start=y[0:2].copy())
        lat.target_heading = nav.heading_command(y[0], y[1], t)
        c_lon = lon.update(y, DT)
        aileron, rudder = lat.update(y, DT)
        c = Controls(elevator=c_lon.elevator, throttle=c_lon.throttle, aileron=aileron, rudder=rudder)
        dcm = quat.to_dcm(y[6:10])
        d = state_derivative(y, c)
        accel_true = d[3:6] + np.cross(y[10:13], y[3:6]) - dcm.T @ np.array([0.0, 0.0, G0])
        accel_meas, gyro_meas = imu.measure(accel_true, y[10:13])
        q_est = est.update(gyro_meas, accel_meas, DT, vel_body=np.array([SPD, 0.0, 0.0]))
        if i % gps_every == 0:
            _, v_gps = gps.measure(y[0:3], dcm @ y[3:6])
            if kp_yaw > 0.0:
                q_est = est.correct_yaw(float(np.arctan2(v_gps[1], v_gps[0])), kp_yaw, DT * gps_every)

        e_true = np.array(quat.to_euler(y[6:10]))
        e_est = np.array(quat.to_euler(q_est))
        ts.append(t)
        yaw_err.append(np.degrees(wrap(e_est[2] - e_true[2])))
        dq = quat.multiply(quat.conjugate(y[6:10]), q_est)
        tot_err.append(np.degrees(2.0 * np.arccos(np.clip(abs(dq[0]), 0.0, 1.0))))
        y = post_step(rk4_step(make_dynamics(constant_controls(c)), t, y, DT))

    return np.array(ts), np.array(yaw_err), np.array(tot_err)


def report(label, ts, yaw_err, tot_err):
    print(label)
    print("window (s)   | yaw error mean | yaw |error| max | total RMS")
    for a in range(0, int(T_END), 100):
        m = (ts >= a) & (ts < a + 100)
        print(f"  {a:3d}-{a + 100:3d}   | {yaw_err[m].mean():8.2f} deg | {np.abs(yaw_err[m]).max():8.2f} deg | "
              f"{np.sqrt(np.mean(tot_err[m] ** 2)):6.2f} deg")


def main():
    for label, kp in (("No yaw correction", 0.0), ("GPS course yaw correction, kp_yaw=0.2", 0.2)):
        report(label, *run(kp))


if __name__ == "__main__":
    main()
