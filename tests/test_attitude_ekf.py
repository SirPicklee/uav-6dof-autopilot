import numpy as np

from uavsim import quaternion as quat
from uavsim.attitude import MahonyFilter
from uavsim.ekf import EKF, G_NED

G = 9.80665
UP_BODY = np.array([0.0, 0.0, -G])   # specific force in level, unaccelerated flight


def test_mahony_holds_level_attitude():
    f = MahonyFilter(quat.from_euler(0.0, 0.0, 0.0), kp=1.0)
    for _ in range(1000):
        f.update(np.zeros(3), UP_BODY, 0.01)
    assert np.allclose(quat.to_euler(f.q), 0.0, atol=1e-9)


def test_mahony_integrates_gyro():
    f = MahonyFilter(quat.from_euler(0.0, 0.0, 0.0), kp=0.0)
    for _ in range(100):
        f.update(np.array([0.0, 0.0, 0.5]), UP_BODY, 0.01)   # yaw rate 0.5 rad/s for 1 s
    assert np.isclose(quat.to_euler(f.q)[2], 0.5, atol=1e-3)


def test_mahony_corrects_wrong_initial_roll():
    f = MahonyFilter(quat.from_euler(np.radians(20.0), 0.0, 0.0), kp=2.0)
    for _ in range(1000):
        f.update(np.zeros(3), UP_BODY, 0.01)
    assert abs(quat.to_euler(f.q)[0]) < np.radians(1.0)


def test_mahony_quaternion_stays_normalized():
    f = MahonyFilter(quat.from_euler(0.1, 0.2, 0.3), kp=1.0)
    for _ in range(500):
        f.update(np.array([0.1, -0.2, 0.3]), UP_BODY, 0.01)
    assert np.isclose(np.linalg.norm(f.q), 1.0)


def test_mahony_centripetal_compensation_in_steady_turn():
    # Coordinated level turn: bank phi, yaw rate r, speed V. Accelerometer reads
    # specific force = [0, 0, -g/cos(phi)] in body axes for a perfectly coordinated turn.
    v, phi = 50.0, np.radians(30.0)
    turn_rate = G * np.tan(phi) / v
    omega = np.array([0.0, np.sin(phi) * turn_rate, np.cos(phi) * turn_rate])
    accel = np.array([0.0, 0.0, -G / np.cos(phi)])
    vel_body = np.array([v, 0.0, 0.0])

    def final_roll(comp):
        f = MahonyFilter(quat.from_euler(phi, 0.0, 0.0), kp=1.0)
        for _ in range(500):
            f.update(omega, accel, 0.01, vel_body=vel_body if comp else None)
        return abs(quat.to_euler(f.q)[0] - phi)

    assert final_roll(True) < final_roll(False)


def test_ekf_gps_update_reduces_position_covariance():
    ekf = EKF(np.zeros(3), np.zeros(3), pos_sigma=5.0)
    before = ekf.P[0, 0]
    ekf.update_gps(np.zeros(3), np.zeros(3), 2.5, 0.1)
    assert ekf.P[0, 0] < before


def test_ekf_measurement_equal_to_estimate_leaves_state_unchanged():
    ekf = EKF(np.array([10.0, 20.0, -1000.0]), np.array([50.0, 0.0, 0.0]))
    x0 = ekf.x.copy()
    ekf.update_gps(x0[0:3], x0[3:6], 2.5, 0.1)
    ekf.update_baro(1000.0, 1.5)
    assert np.allclose(ekf.x, x0)


def test_ekf_predict_constant_velocity():
    ekf = EKF(np.zeros(3), np.array([50.0, 0.0, 0.0]))
    dcm = np.eye(3)
    accel_level = -G_NED          # specific force that cancels gravity in NED
    for _ in range(1000):
        ekf.predict(accel_level, dcm, 0.01)
    assert np.allclose(ekf.x[0:3], [500.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(ekf.x[3:6], [50.0, 0.0, 0.0], atol=1e-9)


def test_ekf_corrects_initial_position_error():
    ekf = EKF(np.array([30.0, -30.0, -1020.0]), np.zeros(3), pos_sigma=5.0)
    truth = np.array([0.0, 0.0, -1000.0])
    for _ in range(50):
        ekf.update_gps(truth, np.zeros(3), 2.5, 0.1)
        ekf.update_baro(1000.0, 1.5)
    assert np.linalg.norm(ekf.x[0:3] - truth) < 2.0
