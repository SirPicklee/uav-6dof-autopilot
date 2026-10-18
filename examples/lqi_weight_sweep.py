"""Sweep the altitude weight Q_h of the LQI design and show the trade-off.

Metrics per weight: settling time (2 m band), minimum airspeed, maximum
angle of attack, in the nonlinear 6-DoF model (50 m climb).
"""

import numpy as np
import matplotlib.pyplot as plt

from uavsim.dynamics import trim_level_flight
from uavsim.linearize import linearize, longitudinal
from uavsim.lqr import lqr
from lqi_longitudinal import augment
from lqi_vs_pid import (
    LQIController, TARGET_ALT, TARGET_SPEED, simulate,
)

WEIGHTS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def settle_time(d, band=2.0):
    err = np.abs(d["alt"] - TARGET_ALT)
    outside = np.where(err > band)[0]
    if len(outside) == 0:
        return 0.0
    if outside[-1] + 1 >= len(d["t"]):
        return float("nan")
    return float(d["t"][outside[-1] + 1])


def main():
    trim = trim_level_flight(TARGET_SPEED, TARGET_ALT)
    A, B, _, _ = linearize(trim)
    A_lon, B_lon = longitudinal(A, B)
    A_aug, B_aug = augment(A_lon, B_lon, TARGET_SPEED)
    R = np.diag([100.0, 1.0])

    rows = []
    for qh in WEIGHTS:
        Q = np.diag([0.1, 1.0, 1.0, 10.0, qh, 0.5])
        K, _, _ = lqr(A_aug, B_aug, Q, R)
        d = simulate(LQIController(trim, K), trim)
        row = (qh, settle_time(d), float(d["spd"].min()), float(d["alpha"].max()))
        rows.append(row)
        print(f"Q_h={row[0]:5.2f} | settling {row[1]:6.1f} s | min speed {row[2]:5.1f} m/s | max alpha {row[3]:5.2f} deg")

    rows = np.array(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["Settling time, 2 m band (s)", "Minimum airspeed (m/s)", "Max angle of attack (deg)"]
    for i, ax in enumerate(axes):
        ax.semilogx(rows[:, 0], rows[:, i + 1], "o-")
        ax.set_xlabel("Altitude weight Q_h")
        ax.set_ylabel(labels[i])
        ax.grid(True, which="both")
    fig.suptitle("LQI design trade-off vs. altitude weight (nonlinear 6-DoF, 50 m climb)")
    fig.tight_layout()
    fig.savefig("examples/lqi_weight_sweep.png", dpi=150)
    print("Saved examples/lqi_weight_sweep.png")
    plt.show()


if __name__ == "__main__":
    main()
