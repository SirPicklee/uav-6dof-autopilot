import numpy as np

from uavsim.dynamics import trim_level_flight
from uavsim.linearize import (
    describe_eigenvalues,
    f_reduced,
    lateral,
    linearize,
    longitudinal,
)
from uavsim.aircraft import DEFAULT_AIRCRAFT as P


def setup():
    trim = trim_level_flight(50.0, 1000.0)
    A, B, x0, u0 = linearize(trim)
    return trim, A, B, x0, u0


def test_matrix_shapes():
    _, A, B, x0, u0 = setup()
    assert A.shape == (12, 12)
    assert B.shape == (12, 4)
    assert x0.shape == (12,)
    assert u0.shape == (4,)


def test_trim_is_equilibrium_of_reduced_model():
    _, _, _, x0, u0 = setup()
    d = f_reduced(x0, u0, P)
    # Position rates are the NED velocity (nonzero for pn); everything else ~ 0
    assert np.isclose(d[2], 0.0, atol=1e-6)
    assert np.allclose(d[3:], 0.0, atol=1e-6)


def test_subsystem_shapes():
    _, A, B, _, _ = setup()
    A_lon, B_lon = longitudinal(A, B)
    A_lat, B_lat = lateral(A, B)
    assert A_lon.shape == (4, 4) and B_lon.shape == (4, 2)
    assert A_lat.shape == (4, 4) and B_lat.shape == (4, 2)


def test_longitudinal_modes_are_stable():
    _, A, B, _, _ = setup()
    A_lon, _ = longitudinal(A, B)
    assert np.all(np.linalg.eigvals(A_lon).real < 0.0)


def test_lateral_modes_are_stable():
    _, A, B, _, _ = setup()
    A_lat, _ = lateral(A, B)
    assert np.all(np.linalg.eigvals(A_lat).real < 0.0)


def test_phugoid_slow_and_short_period_fast():
    _, A, B, _, _ = setup()
    A_lon, _ = longitudinal(A, B)
    osc = sorted(
        [m for m in describe_eigenvalues(A_lon) if m["type"] == "oscillatory"],
        key=lambda m: m["wn"],
    )
    assert len(osc) == 2
    phugoid, short_period = osc
    assert phugoid["wn"] < 1.0 < short_period["wn"]
    assert 20.0 < phugoid["period"] < 35.0
    assert short_period["period"] < 5.0
    assert 0.0 < phugoid["zeta"] < short_period["zeta"] < 1.0


def test_lateral_has_dutch_roll_roll_and_spiral():
    _, A, B, _, _ = setup()
    A_lat, _ = lateral(A, B)
    modes = describe_eigenvalues(A_lat)
    osc = [m for m in modes if m["type"] == "oscillatory"]
    real = sorted([m for m in modes if m["type"] == "real"], key=lambda m: m["tau"])
    assert len(osc) == 1 and len(real) == 2
    assert real[0]["tau"] < 1.0      # roll subsidence
    assert real[1]["tau"] > 10.0     # spiral


def test_elevator_pitch_moment_sign_in_B():
    _, A, B, _, _ = setup()
    # Positive elevator -> negative q_dot (nose down) in this model
    assert B[10, 0] < 0.0
    # Throttle increases u_dot
    assert B[3, 3] > 0.0
