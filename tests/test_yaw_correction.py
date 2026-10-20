import numpy as np

from uavsim import quaternion as quat
from uavsim.attitude import MahonyFilter


def test_correct_yaw_no_change_when_course_matches():
    q0 = quat.from_euler(0.1, 0.05, 0.7)
    f = MahonyFilter(q0, kp=1.0)
    f.correct_yaw(0.7, 0.2, 0.2)
    assert np.allclose(f.q, quat.normalize(q0), atol=1e-9)


def test_correct_yaw_pulls_toward_course():
    f = MahonyFilter(quat.from_euler(0.0, 0.0, 0.0), kp=1.0)
    course = np.radians(20.0)
    before = abs(course - quat.to_euler(f.q)[2])
    for _ in range(50):
        f.correct_yaw(course, 0.2, 0.2)
    after = abs(course - quat.to_euler(f.q)[2])
    assert after < before * 0.15


def test_correct_yaw_wraps_through_pi():
    # Estimated yaw 170 deg, course -170 deg: shortest path is +20 deg through 180
    f = MahonyFilter(quat.from_euler(0.0, 0.0, np.radians(170.0)), kp=1.0)
    f.correct_yaw(np.radians(-170.0), 0.2, 0.2)
    yaw = np.degrees(quat.to_euler(f.q)[2])
    assert np.isclose(yaw, 170.8, atol=0.05)


def test_correct_yaw_keeps_quaternion_normalized():
    f = MahonyFilter(quat.from_euler(0.2, -0.1, 0.3), kp=1.0)
    for _ in range(200):
        f.correct_yaw(1.5, 0.2, 0.2)
    assert np.isclose(np.linalg.norm(f.q), 1.0)

