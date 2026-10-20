"""Yaw correction with a constant wind: course != yaw, so the GPS-course fix is biased."""

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
T_END = 300.0
SQUARE = np.array([[2000.0, 0.0], [2000.0, 2000.0], [0.0, 2000.0], [0.0, 0.0]])
WIND = np.array([0.0, 5.0, 0.0])   # 5 m/s toward east


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
    wind = lambda t, s: WIND

    yaw_err = []
    for i in range(int(T_END / DT)):
        t = i * DT
        if nav.done:
            nav = L1Navigator(SQUARE, start=y[0:2].copy())
        lat.target_heading = nav.heading_command(y[0], y[1], t)
        airspeed = float(np.linalg.norm(y[3:6] - quat.to_dcm(y[6:10]).T @ WIND))
        c_lon = lon.update(y, DT, airspeed)
        aileron, rudder = lat.update(y, DT)
        c = Controls(elevator=c_lon.elevator, throttle=c_lon.throttle, aileron=aileron, rudder=rudder)
        dcm = quat.to_dcm(y[6:10])
        d = state_derivative(y, c, wind_ned=WIND)
        accel_true = d[3:6] + np.cross(y[10:13], y[3:6]) - dcm.T @ np.array([0.0, 0.0, G0])
        accel_meas, gyro_meas = imu.measure(accel_true, y[10:13])
        q_est = est.update(gyro_meas, accel_meas, DT, vel_body=np.array([SPD, 0.0, 0.0]))
        if i % gps_every == 0:
            _, v_gps = gps.measure(y[0:3], dcm @ y[3:6])
            if kp_yaw > 0.0:
                q_est = est.correct_yaw(float(np.arctan2(v_gps[1], v_gps[0])), kp_yaw, DT * gps_every)
        e_true = quat.to_euler(y[6:10])
        e_est = quat.to_euler(q_est)
        yaw_err.append(np.degrees(wrap(e_est[2] - e_true[2])))
        y = post_step(rk4_step(make_dynamics(constant_controls(c), wind_model=wind), t, y, DT))
    return np.array(yaw_err)


def main():
    print("Constant 5 m/s east wind, 300 s")
    for label, kp in (("No yaw correction", 0.0), ("GPS course correction", 0.2)):
        e = run(kp)
        print(f"  {label:22s} | yaw error mean {e.mean():6.2f} deg | RMS {np.sqrt(np.mean(e**2)):6.2f} deg | max |err| {np.abs(e).max():6.2f} deg")


if __name__ == "__main__":
    main()
