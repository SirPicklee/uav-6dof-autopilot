"""Effect of attitude error on EKF accuracy (attitude is an input to the EKF)."""

import numpy as np

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
from uavsim.atmosphere import G0
from uavsim.autopilot import Autopilot
from uavsim.dynamics import (
    constant_controls, make_dynamics, make_state, post_step,
    state_derivative, trim_level_flight,
)
from uavsim.ekf import EKF
from uavsim.guidance import L1Navigator
from uavsim.integrators import rk4_step
from uavsim.lateral import LateralAutopilot
from uavsim.sensors import GPS, IMU, Barometer

ALT, SPD, DT = 1000.0, 50.0, 0.01
WAYPOINTS = np.array([[2000.0, 0.0], [2000.0, 2000.0], [0.0, 2000.0], [0.0, 0.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def run(att_err_deg):
    trim = trim_level_flight(SPD, ALT)
    lon = Autopilot(trim, ALT, SPD)
    lat = LateralAutopilot(target_heading=0.0)
    nav = L1Navigator(WAYPOINTS)
    y = make_state(SPD, trim.alpha, trim.pitch, ALT)
    imu, gps, baro = IMU(), GPS(), Barometer()
    ekf = EKF(y[0:3].copy() + np.array([3.0, -3.0, 2.0]), np.array([SPD, 0.0, 0.0]))
    gps_every = int(round(1.0 / (gps.rate_hz * DT)))
    baro_every = int(round(1.0 / (baro.rate_hz * DT)))
    err_rot = rot_x(np.radians(att_err_deg))

    pos_err, vel_err = [], []
    ts = []
    for i in range(int(150.0 / DT)):
        t = i * DT
        if nav.done:
            break
        lat.target_heading = nav.heading_command(y[0], y[1], t)
        c_lon = lon.update(y, DT)
        aileron, rudder = lat.update(y, DT)
        c = Controls(elevator=c_lon.elevator, throttle=c_lon.throttle, aileron=aileron, rudder=rudder)
        f = make_dynamics(constant_controls(c))
        dcm = quat.to_dcm(y[6:10])
        d = state_derivative(y, c)
        accel_true = d[3:6] + np.cross(y[10:13], y[3:6]) - dcm.T @ np.array([0.0, 0.0, G0])
        accel_meas, _ = imu.measure(accel_true, y[10:13])
        ekf.predict(accel_meas, dcm @ err_rot, DT)   # EKF uses the wrong attitude
        if i % gps_every == 0:
            p_meas, v_meas = gps.measure(y[0:3], dcm @ y[3:6])
            ekf.update_gps(p_meas, v_meas, gps.pos_noise, gps.vel_noise)
        if i % baro_every == 0:
            ekf.update_baro(baro.measure(-y[2]), baro.noise)
        pos_err.append(np.linalg.norm(ekf.x[0:3] - y[0:3]))
        vel_err.append(np.linalg.norm(ekf.x[3:6] - dcm @ y[3:6]))
        ts.append(t)
        y = post_step(rk4_step(f, t, y, DT))

    ts = np.array(ts)
    m = ts > 10.0
    rms = lambda a: float(np.sqrt(np.mean(np.array(a)[m] ** 2)))
    return rms(pos_err), rms(vel_err), float(np.max(np.array(pos_err)[m]))


def main():
    print("Roll attitude error given to EKF | position RMS | velocity RMS | max position error")
    for e in (0.0, 1.0, 3.0, 10.0):
        p, v, mx = run(e)
        print(f"  {e:4.1f} deg | {p:6.2f} m | {v:6.3f} m/s | {mx:6.2f} m")


if __name__ == "__main__":
    main()
