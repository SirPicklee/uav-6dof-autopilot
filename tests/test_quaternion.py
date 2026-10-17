import numpy as np
import pytest

from uavsim import quaternion as quat


def test_normalize_gives_unit_norm():
    q = quat.normalize([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_normalize_zero_raises():
    with pytest.raises(ValueError):
        quat.normalize([0.0, 0.0, 0.0, 0.0])


def test_identity_rotation_leaves_vector_unchanged():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(quat.rotate(q, v), v)


def test_euler_round_trip():
    roll, pitch, yaw = 0.3, -0.5, 1.2
    q = quat.from_euler(roll, pitch, yaw)
    r, p, y = quat.to_euler(q)
    assert np.allclose([r, p, y], [roll, pitch, yaw])


def test_90deg_yaw_rotates_x_to_y():
    q = quat.from_euler(0.0, 0.0, np.pi / 2)
    v = quat.rotate(q, [1.0, 0.0, 0.0])
    assert np.allclose(v, [0.0, 1.0, 0.0])


def test_rotate_matches_dcm():
    q = quat.from_euler(0.4, -0.2, 0.9)
    v = np.array([1.0, -2.0, 0.5])
    assert np.allclose(quat.rotate(q, v), quat.to_dcm(q) @ v)


def test_conjugate_is_inverse_rotation():
    q = quat.from_euler(0.4, -0.2, 0.9)
    v = np.array([1.0, -2.0, 0.5])
    v_inertial = quat.rotate(q, v)
    assert np.allclose(quat.rotate(quat.conjugate(q), v_inertial), v)


def test_multiply_by_conjugate_gives_identity():
    q = quat.from_euler(0.4, -0.2, 0.9)
    result = quat.multiply(q, quat.conjugate(q))
    assert np.allclose(result, [1.0, 0.0, 0.0, 0.0])


def test_dcm_is_orthonormal_with_unit_determinant():
    R = quat.to_dcm(quat.from_euler(0.4, -0.2, 0.9))
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_derivative_matches_small_rotation():
    # Constant roll rate of 1 rad/s for a tiny dt should match from_euler.
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    dt = 1e-6
    q1 = quat.normalize(q0 + quat.derivative(q0, [1.0, 0.0, 0.0]) * dt)
    assert np.allclose(q1, quat.from_euler(dt, 0.0, 0.0), atol=1e-9)