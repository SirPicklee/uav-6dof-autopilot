import numpy as np
import pytest

from uavsim.integrators import integrate
from uavsim import quaternion as quat


def free_fall(t, y):
    # y = [z, vz], z positive up, gravity only
    return np.array([y[1], -9.81])


def oscillator(t, y):
    # y = [x, v], unit mass and unit stiffness: x'' = -x
    return np.array([y[1], -y[0]])


def test_rk4_free_fall_matches_analytic():
    y0 = np.array([100.0, 0.0])
    traj = integrate(free_fall, y0, (0.0, 3.0), dt=0.01, method="rk4")
    z_exact = 100.0 - 0.5 * 9.81 * 3.0**2
    v_exact = -9.81 * 3.0
    assert np.isclose(traj.final_state[0], z_exact, atol=1e-9)
    assert np.isclose(traj.final_state[1], v_exact, atol=1e-9)


def test_rk4_oscillator_matches_analytic():
    y0 = np.array([1.0, 0.0])
    traj = integrate(oscillator, y0, (0.0, 10.0), dt=0.01, method="rk4")
    assert np.isclose(traj.final_state[0], np.cos(10.0), atol=1e-8)
    assert np.isclose(traj.final_state[1], -np.sin(10.0), atol=1e-8)


def test_rk4_is_more_accurate_than_euler():
    y0 = np.array([1.0, 0.0])
    exact = np.cos(10.0)
    err_euler = abs(integrate(oscillator, y0, (0.0, 10.0), 0.01, "euler").final_state[0] - exact)
    err_rk4 = abs(integrate(oscillator, y0, (0.0, 10.0), 0.01, "rk4").final_state[0] - exact)
    assert err_rk4 < err_euler * 1e-3


def test_rk4_has_fourth_order_convergence():
    y0 = np.array([1.0, 0.0])
    exact = np.cos(5.0)
    err_coarse = abs(integrate(oscillator, y0, (0.0, 5.0), 0.1, "rk4").final_state[0] - exact)
    err_fine = abs(integrate(oscillator, y0, (0.0, 5.0), 0.05, "rk4").final_state[0] - exact)
    order = np.log2(err_coarse / err_fine)
    assert 3.5 < order < 4.5


def test_stop_condition_halts_at_ground():
    y0 = np.array([100.0, 0.0])
    traj = integrate(
        free_fall, y0, (0.0, 20.0), dt=0.01,
        stop_condition=lambda t, y: y[0] <= 0.0,
    )
    assert traj.final_state[0] <= 0.0
    assert traj.t[-1] < 20.0


def test_post_step_keeps_quaternion_normalized():
    omega = np.array([0.5, -0.3, 0.8])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    traj = integrate(lambda t, y: quat.derivative(y, omega), q0, (0.0, 20.0), dt=0.05, method="euler", post_step=quat.normalize)
    norms = np.linalg.norm(traj.y, axis=1)
    assert np.allclose(norms, 1.0)


def test_invalid_inputs_raise():
    y0 = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        integrate(free_fall, y0, (0.0, 1.0), dt=-0.1)
        with pytest.raises(ValueError):
            integrate(free_fall, y0, (1.0, 0.0), dt=0.1)
            with pytest.raises(ValueError):
                integrate(free_fall, y0, (0.0, 1.0), dt=0.1, method="bogus")