import numpy as np
import pytest

from uavsim.aerodynamics import Controls
from uavsim.atmosphere import G0
from uavsim.dynamics import (
    IDX_QUAT,
    constant_controls,
    make_dynamics,
    make_state,
    post_step,
    state_derivative,
    trim_level_flight,
)
from uavsim.integrators import integrate


def test_zero_airspeed_gives_gravity_only():
    y = make_state(airspeed=0.0, alpha=0.0, pitch=0.0, altitude=1000.0)
    d = state_derivative(y, Controls())
    assert np.isclose(d[3], 0.0)
    assert np.isclose(d[4], 0.0)
    assert np.isclose(d[5], G0)
    assert np.allclose(d[10:13], 0.0)


def test_position_rate_is_ned_velocity():
    y = make_state(airspeed=50.0, alpha=0.0, pitch=0.0, altitude=1000.0)
    d = state_derivative(y, Controls())
    assert np.allclose(d[0:3], [50.0, 0.0, 0.0])


def test_wind_equal_to_ground_velocity_removes_aero():
    y = make_state(airspeed=50.0, alpha=0.0, pitch=0.0, altitude=1000.0)
    d = state_derivative(y, Controls(), wind_ned=np.array([50.0, 0.0, 0.0]))
    assert np.isclose(d[3], 0.0, atol=1e-9)
    assert np.isclose(d[5], G0)
    assert np.allclose(d[10:13], 0.0, atol=1e-9)


@pytest.mark.parametrize("airspeed", [40.0, 45.0, 50.0])
def test_trim_converges(airspeed):
    trim = trim_level_flight(airspeed, 1000.0)
    assert trim.residual < 1e-6
    assert 0.0 < trim.throttle < 1.0
    assert 0.0 < trim.alpha < 0.2


def test_trim_state_has_zero_derivatives_and_no_climb():
    trim = trim_level_flight(50.0, 1000.0)
    d = state_derivative(trim.state(), trim.controls)
    assert np.isclose(d[2], 0.0, atol=1e-6)    # no climb rate
    assert np.isclose(d[3], 0.0, atol=1e-6)    # u_dot
    assert np.isclose(d[5], 0.0, atol=1e-6)    # w_dot
    assert np.isclose(d[11], 0.0, atol=1e-6)   # q_dot


def test_trim_holds_steady_flight_for_10s():
    trim = trim_level_flight(50.0, 1000.0)
    f = make_dynamics(constant_controls(trim.controls))
    traj = integrate(f, trim.state(), (0.0, 10.0), dt=0.01, method="rk4", post_step=post_step)
    final = traj.final_state
    assert abs(final[2] - trim.state()[2]) < 1.0
    airspeed = np.linalg.norm(final[3:6])
    assert abs(airspeed - 50.0) < 0.5


def test_quaternion_norm_stays_unit_with_rk4():
    trim = trim_level_flight(50.0, 1000.0)
    y0 = trim.state()
    y0[10] = 0.5  # initial roll rate disturbance
    f = make_dynamics(constant_controls(trim.controls))
    traj = integrate(f, y0, (0.0, 10.0), dt=0.01, method="rk4")
    norms = np.linalg.norm(traj.y[:, IDX_QUAT], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
