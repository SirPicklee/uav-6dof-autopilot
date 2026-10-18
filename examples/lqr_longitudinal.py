"""Longitudinal LQR design: compare open-loop and closed-loop modes."""

import numpy as np

from uavsim.dynamics import trim_level_flight
from uavsim.linearize import linearize, longitudinal
from uavsim.lqr import controllability_rank, lqr


def main():
    trim = trim_level_flight(50.0, 1000.0)
    A, B, _, _ = linearize(trim)
    A_lon, B_lon = longitudinal(A, B)

    print("Controllability rank:", controllability_rank(A_lon, B_lon), "of", A_lon.shape[0])

    # States [u, w, q, theta], inputs [elevator, throttle]
    Q = np.diag([0.1, 1.0, 1.0, 10.0])
    R = np.diag([10.0, 1.0])
    K, _, eig = lqr(A_lon, B_lon, Q, R)

    np.set_printoptions(precision=4, suppress=True)
    print("K =\n", K)
    print("Open-loop eigenvalues:  ", np.linalg.eigvals(A_lon))
    print("Closed-loop eigenvalues:", eig)


if __name__ == "__main__":
    main()
