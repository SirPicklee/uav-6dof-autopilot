"""First flight: trimmed UAV with an elevator pulse and a throttle step."""

import numpy as np
import matplotlib.pyplot as plt

from uavsim import quaternion as quat
from uavsim.aerodynamics import Controls
from uavsim.dynamics import make_dynamics, post_step, trim_level_flight
from uavsim.integrators import integrate

AIRSPEED = 50.0
ALTITUDE = 1000.0


def main():
    trim = trim_level_flight(AIRSPEED, ALTITUDE)
    print(f"Trim: alpha={np.degrees(trim.alpha):.2f} deg, "
          f"elevator={np.degrees(trim.elevator):.2f} deg, throttle={trim.throttle:.3f}")

    def control_law(t, y):
        elevator = trim.elevator
        throttle = trim.throttle
        if 2.0 <= t < 3.0:
            elevator -= np.radians(2.0)  # nose-up pulse
        if t >= 30.0:
            throttle = min(1.0, trim.throttle + 0.2)
        return Controls(elevator=elevator, throttle=throttle)

    f = make_dynamics(control_law)
    traj = integrate(f, trim.state(), (0.0, 80.0), dt=0.01, method="rk4", post_step=post_step)

    t = traj.t
    altitude = -traj.y[:, 2]
    airspeed = np.linalg.norm(traj.y[:, 3:6], axis=1)
    alpha = np.degrees(np.arctan2(traj.y[:, 5], traj.y[:, 3]))
    pitch = np.degrees([quat.to_euler(q)[1] for q in traj.y[:, 6:10]])

    print(f"Final altitude: {altitude[-1]:.1f} m, final airspeed: {airspeed[-1]:.1f} m/s")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, altitude)
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("Trimmed UAV: elevator pulse at 2 s, throttle step at 30 s")
    axes[0].grid(True)
    axes[1].plot(t, airspeed)
    axes[1].set_ylabel("Airspeed (m/s)")
    axes[1].grid(True)
    axes[2].plot(t, pitch, label="pitch")
    axes[2].plot(t, alpha, label="alpha")
    axes[2].set_ylabel("Angle (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True)
    fig.tight_layout()
    fig.savefig("examples/first_flight.png", dpi=150)
    print("Saved examples/first_flight.png")
    plt.show()


if __name__ == "__main__":
    main()
