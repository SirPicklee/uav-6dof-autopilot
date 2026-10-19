import numpy as np
import pytest

from uavsim.actuators import Actuator


def test_actuator_reaches_63_percent_at_one_time_constant():
    a = Actuator(tau=0.05)
    for _ in range(50):
        a.update(1.0, 0.001)
    assert np.isclose(a.x, 0.632, atol=0.01)


def test_actuator_converges_to_command():
    a = Actuator(tau=0.05)
    for _ in range(1000):
        a.update(0.2, 0.001)
    assert np.isclose(a.x, 0.2, atol=1e-3)


def test_actuator_rate_limit():
    a = Actuator(tau=0.05, rate_limit=1.0)
    for _ in range(500):
        a.update(10.0, 0.001)      # 0.5 s at max 1 unit/s
    assert np.isclose(a.x, 0.5, atol=1e-6)


def test_actuator_position_limit():
    a = Actuator(tau=0.05, pos_limit=0.3)
    for _ in range(2000):
        a.update(1.0, 0.001)
    assert np.isclose(a.x, 0.3, atol=1e-9)


def test_actuator_reset_and_invalid_dt():
    a = Actuator()
    a.reset(0.7)
    assert a.x == 0.7
    with pytest.raises(ValueError):
        a.update(1.0, 0.0)
