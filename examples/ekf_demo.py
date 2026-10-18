"""EKF demo: closed-square mission, estimate vs. raw sensor error."""

import numpy as np
import matplotlib.pyplot as plt

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
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
from uavsim.atmosphere import G0

ALT, SPD, DT = 1000.0, 50.0, 0.01
WAYPOINTS = np.array([[2000.0, 0.0], [2000.0, 2000.0], [0.0, 2000.0], [0.0, 0.0]])


def main():
    trim = trim_level_flight(SPD, ALT)
    lon = Autopilot(trim, ALT, SPD)
    lat = LateralAutopilot(target_heading=0.0)
    nav = L1Navigator(WAYPOINTS)
    y = make_state(SPD, trim.alpha, trim.pitch, ALT)

    imu, gps, baro = IMU(), GPS(), Barometer()
    ekf = EKF(y[0:3].copy() + np.array([3.0, -3.0, 2.0]), np.array([SPD, 0.0, 0.0]))
    gps_every = int(round(1.0 / (gps.rate_hz * DT)))
    baro_every = int(round(1.0 / (baro.rate_hz * DT)))

    rows = []
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
        # Specific force in body axes = body acceleration + omega x v - gravity (body)
        accel_body_true = d[3:6] + np.cross(y[10:13], y[3:6]) - dcm.T @ np.array([0.0, 0.0, G0])
        accel_meas, _ = imu.measure(accel_body_true, y[10:13])
        ekf.predict(accel_meas, dcm, DT)

        if i % gps_every == 0:
            p_meas, v_meas = gps.measure(y[0:3], dcm @ y[3:6])
            ekf.update_gps(p_meas, v_meas, gps.pos_noise, gps.vel_noise)
            gps_err = np.linalg.norm(p_meas - y[0:3])
        if i % baro_every == 0:
            a_meas = baro.measure(-y[2])
            ekf.update_baro(a_meas, baro.noise)
            baro_err = abs(a_meas - (-y[2]))

        v_ned_true = dcm @ y[3:6]
        rows.append([t, np.linalg.norm(ekf.x[0:3] - y[0:3]), abs(ekf.x[2] - y[2]),
                     np.linalg.norm(ekf.x[3:6] - v_ned_true), gps_err, baro_err])
        y = post_step(rk4_step(f, t, y, DT))

    r = np.array(rows)
    skip = r[:, 0] > 10.0  # ignore initial convergence
    print(f"Samples: {len(r)}, duration {r[-1, 0]:.1f} s (first 10 s excluded from RMS)")
    print(f"Position RMS error | EKF {np.sqrt(np.mean(r[skip, 1]**2)):6.2f} m | raw GPS {np.sqrt(np.mean(r[skip, 4]**2)):6.2f} m")
    print(f"Altitude RMS error | EKF {np.sqrt(np.mean(r[skip, 2]**2)):6.2f} m | raw baro {np.sqrt(np.mean(r[skip, 5]**2)):6.2f} m")
    print(f"Velocity RMS error | EKF {np.sqrt(np.mean(r[skip, 3]**2)):6.3f} m/s | raw GPS ~{np.sqrt(3) * gps.vel_noise:.3f} m/s (theory)")

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(r[:, 0], r[:, 1], label="EKF")
    ax[0].plot(r[:, 0], r[:, 4], alpha=0.4, label="raw GPS")
    ax[0].set_ylabel("Position error (m)")
    ax[0].set_title("EKF state estimation: error vs. ground truth")
    ax[1].plot(r[:, 0], r[:, 2], label="EKF")
    ax[1].plot(r[:, 0], r[:, 5], alpha=0.4, label="raw baro")
    ax[1].set_ylabel("Altitude error (m)")
    ax[2].plot(r[:, 0], r[:, 3], label="EKF")
    ax[2].set_ylabel("Velocity error (m/s)")
    ax[2].set_xlabel("Time (s)")
    for a in ax:
        a.grid(True)
        a.legend()
    fig.tight_layout()
    fig.savefig("examples/ekf_estimation.png", dpi=150)
    print("Saved examples/ekf_estimation.png")
    plt.show()


if __name__ == "__main__":
    main()
