"""Modal analysis of the UAV at a level-flight trim point."""

import numpy as np

from uavsim.dynamics import trim_level_flight
from uavsim.linearize import (
    describe_eigenvalues, lateral, linearize, longitudinal,
)

AIRSPEED = 50.0
ALTITUDE = 1000.0


def show(title, modes):
    print(title)
    for m in modes:
        if m["type"] == "oscillatory":
            print(f"  oscillatory: wn={m['wn']:.3f} rad/s, zeta={m['zeta']:.3f}, "
                  f"period={m['period']:.2f} s, lambda={m['lambda']:.4f}")
        else:
            print(f"  real:        lambda={m['lambda'].real:.4f} 1/s, tau={m['tau']:.2f} s")


def main():
    trim = trim_level_flight(AIRSPEED, ALTITUDE)
    A, B, x0, u0 = linearize(trim)

    A_lon, _ = longitudinal(A, B)
    A_lat, _ = lateral(A, B)

    lon = sorted(describe_eigenvalues(A_lon), key=lambda m: m.get("wn", 0.0))
    lat = describe_eigenvalues(A_lat)

    show("Longitudinal modes (slow = phugoid, fast = short period):", lon)
    show("Lateral modes (oscillatory = Dutch roll, real = roll / spiral):", lat)


if __name__ == "__main__":
    main()
