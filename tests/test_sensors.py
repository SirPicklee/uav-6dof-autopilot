import numpy as np

from uavsim.sensors import GPS, IMU, Barometer


def test_sensors_are_reproducible_with_seed():
    a = GPS(seed=5).measure(np.zeros(3), np.zeros(3))
    b = GPS(seed=5).measure(np.zeros(3), np.zeros(3))
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_gps_noise_statistics():
    gps = GPS(seed=7)
    errs = np.array([gps.measure(np.zeros(3), np.zeros(3))[0] for _ in range(5000)])
    assert abs(errs.mean()) < 0.2
    assert np.isclose(errs.std(), gps.pos_noise, rtol=0.05)


def test_baro_noise_statistics():
    baro = Barometer(seed=7)
    errs = np.array([baro.measure(1000.0) - 1000.0 for _ in range(5000)])
    assert abs(errs.mean()) < 0.1
    assert np.isclose(errs.std(), baro.noise, rtol=0.05)


def test_imu_bias_random_walks():
    imu = IMU(seed=7)
    for _ in range(10000):
        imu.measure(np.zeros(3), np.zeros(3))
    assert np.linalg.norm(imu.gyro_bias) > 0.0
    assert np.linalg.norm(imu.accel_bias) > 0.0


def test_imu_noise_level():
    imu = IMU(seed=7, accel_bias_walk=0.0, gyro_bias_walk=0.0)
    gyro = np.array([imu.measure(np.zeros(3), np.zeros(3))[1] for _ in range(5000)])
    assert np.isclose(gyro.std(), imu.gyro_noise, rtol=0.05)
