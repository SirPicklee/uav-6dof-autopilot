"""Torque-free rigid body rotation (Euler's equations) with quaternion attitude.

Validates the quaternion + RK4 stack: for a torque-free body, rotational
kinetic energy and the inertial-frame angular momentum must stay constant.
"""

import numpy as np
import matplotlib.pyplot as plt

from uavsim import quaternion as quat
from uavsim.integrators import integrate

# Asymmetric inertia tensor (kg m^2), principal axes
J = np.diag([0.8, 1.2, 1.5])
J_inv = np.linalg.inv(J)


def dynamics(t, y):
    """State y = [qw, qx, qy, qz, p, q, r]."""
    q = y[:4]
    w = y[4:]
    q_dot = quat.derivative(q, w)
    w_dot = J_inv @ (-np.cross(w, J @ w))
    return np.concatenate([q_dot, w_dot])


def post_step(y):
    y = y.copy()
    y[:4] = quat.normalize(y[:4])
    return y


def main():
    q0 = quat.from_euler(0.0, 0.0, 0.0)
    w0 = np.array([0.1, 3.0, 0.1])  # spin near the unstable intermediate axis
    y0 = np.concatenate([q0, w0])

    traj = integrate(dynamics, y0, (0.0, 30.0), dt=0.005, method="rk4", post_step=post_step)

    t = traj.t
    q = traj.y[:, :4]
    w = traj.y[:, 4:]

    energy = 0.5 * np.einsum("ij,jk,ik->i", w, J, w)
    h_body = w @ J.T
    h_inertial = np.array([quat.rotate(qi, hi) for qi, hi in zip(q, h_body)])
    h_norm = np.linalg.norm(h_inertial, axis=1)
    euler = np.array([quat.to_euler(qi) for qi in q])

    print(f"Energy drift:           {abs(energy[-1] / energy[0] - 1):.2e}")
    print(f"Momentum magnitude drift: {abs(h_norm[-1] / h_norm[0] - 1):.2e}")
    print(f"Max inertial momentum direction change: "
          f"{np.max(np.linalg.norm(h_inertial - h_inertial[0], axis=1)):.2e}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, w[:, 0], label="p")
    axes[0].plot(t, w[:, 1], label="q")
    axes[0].plot(t, w[:, 2], label="r")
    axes[0].set_ylabel("Body rates (rad/s)")
    axes[0].set_title("Torque-free rotation of an asymmetric body (Dzhanibekov effect)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, np.degrees(euler[:, 0]), label="roll")
    axes[1].plot(t, np.degrees(euler[:, 1]), label="pitch")
    axes[1].plot(t, np.degrees(euler[:, 2]), label="yaw")
    axes[1].set_ylabel("Euler angles (deg)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, energy / energy[0] - 1, label="Kinetic energy error")
    axes[2].plot(t, h_norm / h_norm[0] - 1, label="Angular momentum error")
    axes[2].set_ylabel("Relative error")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig("examples/torque_free_rotation.png", dpi=150)
    print("Saved examples/torque_free_rotation.png")
    plt.show()


if __name__ == "__main__":
    main()
