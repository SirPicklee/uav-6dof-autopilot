"""LQI design: longitudinal LQR augmented with altitude and airspeed integrators."""

import numpy as np

from uavsim.dynamics import trim_level_flight
from uavsim.linearize import describe_eigenvalues, linearize, longitudinal
from uavsim.lqr import controllability_rank, lqr

AIRSPEED = 50.0
ALTITUDE = 1000.0


def augment(A_lon, B_lon, v0):
    """Augment [u, w, q, theta] with integrals of altitude and airspeed error.

    Linearized outputs (small angles about level trim):
        h_dot ~ -w + v0 * theta     (altitude rate)
        V_dot ~ u                   (airspeed deviation)
    Integrator states: xi_h' = h_ref - h, xi_v' = V_ref - V.
    """
    n = A_lon.shape[0]
    C = np.zeros((2, n))
    C[0, 1] = 1.0      # -(h_dot) contribution from w ->  xi_h' = -( -w + v0*theta ) = w - v0*theta
    C[0, 3] = -v0
    C[1, 0] = -1.0     # xi_v' = -u
    A_aug = np.block([
        [A_lon, np.zeros((n, 2))],
        [C, np.zeros((2, 2))],
    ])
    B_aug = np.vstack([B_lon, np.zeros((2, B_lon.shape[1]))])
    return A_aug, B_aug


def main():
    trim = trim_level_flight(AIRSPEED, ALTITUDE)
    A, B, _, _ = linearize(trim)
    A_lon, B_lon = longitudinal(A, B)
    A_aug, B_aug = augment(A_lon, B_lon, AIRSPEED)

    print("Augmented controllability rank:", controllability_rank(A_aug, B_aug), "of", A_aug.shape[0])

    # States [u, w, q, theta, xi_h, xi_v]; inputs [elevator, throttle]
    Q = np.diag([0.1, 1.0, 1.0, 10.0, 0.05, 0.5])
    R = np.diag([100.0, 1.0])
    K, _, eig = lqr(A_aug, B_aug, Q, R)

    np.set_printoptions(precision=4, suppress=True)
    print("K =\n", K)
    print("Closed-loop eigenvalues:")
    for m in describe_eigenvalues(A_aug - B_aug @ K):
        if m["type"] == "oscillatory":
            print(f"  oscillatory: wn={m['wn']:.3f} rad/s, zeta={m['zeta']:.3f}, period={m['period']:.2f} s")
        else:
            print(f"  real:        lambda={m['lambda'].real:.4f} 1/s, tau={m['tau']:.2f} s")


if __name__ == "__main__":
    main()
