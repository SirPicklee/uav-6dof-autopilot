"""LQR design for the linearized longitudinal dynamics."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


def controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    """Rank of the controllability matrix [B, AB, A^2 B, ...]."""
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return int(np.linalg.matrix_rank(np.hstack(blocks)))


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Continuous-time LQR. Returns (K, P, closed_loop_eigenvalues).

    Control law: u = -K x, minimizing the integral of x'Qx + u'Ru.
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    eig = np.linalg.eigvals(A - B @ K)
    return K, P, eig
