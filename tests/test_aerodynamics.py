import numpy as np

from uavsim.aircraft import DEFAULT_AIRCRAFT as P
from uavsim.aerodynamics import (
    Controls,
    aero_forces_moments,
    air_data,
    thrust,
)


def vel(alpha, vt=50.0):
    return np.array([vt * np.cos(alpha), 0.0, vt * np.sin(alpha)])


def run(v, omega=(0.0, 0.0, 0.0), controls=Controls(), alt=1000.0):
    return aero_forces_moments(P, v, np.array(omega), controls, alt)


def test_air_data_straight_flight():
    vt, alpha, beta = air_data(np.array([50.0, 0.0, 0.0]))
    assert np.isclose(vt, 50.0)
    assert np.isclose(alpha, 0.0)
    assert np.isclose(beta, 0.0)


def test_air_data_alpha_and_beta():
    vt, alpha, beta = air_data(np.array([50.0, 0.0, 5.0]))
    assert np.isclose(alpha, np.arctan2(5.0, 50.0))
    assert np.isclose(beta, 0.0)
    vt, alpha, beta = air_data(np.array([50.0, 5.0, 0.0]))
    assert np.isclose(beta, np.arcsin(5.0 / vt))
    assert np.isclose(alpha, 0.0)


def test_zero_airspeed_gives_zero_force_and_moment():
    res = run(np.zeros(3))
    assert np.allclose(res.force_body, 0.0)
    assert np.allclose(res.moment_body, 0.0)


def test_lift_increases_with_alpha():
    low = run(vel(0.0))
    high = run(vel(0.05))
    assert high.lift > low.lift
    # Lift acts upward: negative body z force (z is down)
    assert high.force_body[2] < 0.0


def test_lift_saturates_at_cl_max():
    a = run(vel(0.5))
    b = run(vel(0.6))
    expected = a.dynamic_pressure * P.wing_area * P.cl_max
    assert np.isclose(a.lift, expected)
    assert np.isclose(b.lift, expected)
    # Post-stall drag penalty grows with depth of stall
    assert b.drag > a.drag


def test_drag_matches_polar():
    res = run(vel(0.0))
    cl = P.cl0
    cd = P.cd0 + P.induced_drag_k * cl**2
    assert np.isclose(res.drag, res.dynamic_pressure * P.wing_area * cd)


def test_static_pitch_stability():
    m0 = run(vel(0.0)).moment_body[1]
    m1 = run(vel(0.1)).moment_body[1]
    assert m1 < m0


def test_pitch_damping_opposes_pitch_rate():
    m0 = run(vel(0.0), omega=(0.0, 0.0, 0.0)).moment_body[1]
    m1 = run(vel(0.0), omega=(0.0, 0.5, 0.0)).moment_body[1]
    assert m1 < m0


def test_roll_damping_opposes_roll_rate():
    res = run(np.array([50.0, 0.0, 0.0]), omega=(0.5, 0.0, 0.0))
    assert res.moment_body[0] < 0.0


def test_sideslip_restoring_signs():
    res = run(np.array([50.0, 5.0, 0.0]))
    assert res.beta > 0.0
    assert res.force_body[1] < 0.0     # side force opposes sideslip
    assert res.moment_body[0] < 0.0    # dihedral effect
    assert res.moment_body[2] > 0.0    # weathercock stability


def test_elevator_gives_nose_down_moment():
    m0 = run(vel(0.0)).moment_body[1]
    m1 = run(vel(0.0), controls=Controls(elevator=0.1)).moment_body[1]
    assert m1 < m0


def test_thrust_model():
    assert np.isclose(thrust(P, 1.0, 0.0, 1.225), P.max_thrust_sl)
    assert np.isclose(thrust(P, 0.0, 0.0, 1.225), 0.0)
    assert np.isclose(thrust(P, 1.0, 50.0, 1.225), P.max_thrust_sl * 0.4)
    assert np.isclose(thrust(P, 1.0, 0.0, 0.6125), P.max_thrust_sl * 0.5)
    assert np.isclose(thrust(P, 2.0, 0.0, 1.225), P.max_thrust_sl)
    assert np.isclose(thrust(P, 1.0, 100.0, 1.225), 0.0)
