"""Quaternion utilities for rigid-body attitude representation.

Convention: q = [w, x, y, z] (scalar-first), Hamilton product.
A unit quaternion q rotates a vector from the body frame to the
inertial (NED) frame: v_ned = q * v_body * q^-1.
"""

import numpy as np


def normalize(q):
    """Return the unit-norm version of quaternion q."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("Cannot normalize a zero quaternion.")
    return q / n


def conjugate(q):
    """Return the conjugate of q (inverse rotation for a unit quaternion)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def multiply(q1, q2):
    """Hamilton product q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotate(q, v):
    """Rotate vector v (body frame) into the inertial frame using q."""
    qv = np.array([0.0, *v])
    return multiply(multiply(q, qv), conjugate(q))[1:]


def from_euler(roll, pitch, yaw):
    """Build a quaternion from Euler angles (rad), 3-2-1 (yaw-pitch-roll) sequence."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def to_euler(q):
    """Convert a quaternion to Euler angles (roll, pitch, yaw) in rad."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def to_dcm(q):
    """Return the 3x3 rotation matrix (body -> inertial) for quaternion q."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def derivative(q, omega):
    """Quaternion kinematics: dq/dt for body angular rate omega = [p, q, r] (rad/s)."""
    p, qr, r = omega
    return 0.5 * multiply(q, np.array([0.0, p, qr, r]))