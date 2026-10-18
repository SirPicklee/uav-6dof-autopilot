import numpy as np
import pytest

from uavsim.pid import PID


def test_proportional_only():
    pid = PID(kp=2.0)
    assert np.isclose(pid.update(setpoint=10.0, measurement=4.0, dt=0.1), 12.0)


def test_integral_accumulates():
    pid = PID(kp=0.0, ki=1.0)
    out = 0.0
    for _ in range(10):
        out = pid.update(setpoint=1.0, measurement=0.0, dt=0.1)
    assert np.isclose(out, 1.0)


def test_output_is_saturated():
    pid = PID(kp=10.0, out_min=-1.0, out_max=1.0)
    assert pid.update(5.0, 0.0, 0.1) == 1.0
    assert pid.update(-5.0, 0.0, 0.1) == -1.0


def test_anti_windup_limits_integral():
    pid = PID(kp=1.0, ki=5.0, out_min=-1.0, out_max=1.0)
    for _ in range(1000):
        pid.update(10.0, 0.0, 0.01)
    # Integral must not have grown far beyond what saturation needs
    assert pid._integral < 20.0


def test_anti_windup_recovers_quickly():
    pid = PID(kp=1.0, ki=5.0, out_min=-1.0, out_max=1.0)
    for _ in range(1000):
        pid.update(10.0, 0.0, 0.01)
    # Setpoint now below measurement: output should leave +1 within a few steps
    outs = [pid.update(0.0, 5.0, 0.01) for _ in range(5)]
    assert min(outs) < 1.0


def test_no_derivative_kick_on_setpoint_change():
    pid = PID(kp=0.0, kd=1.0)
    pid.update(0.0, 0.0, 0.01)
    out = pid.update(100.0, 0.0, 0.01)
    assert np.isclose(out, 0.0)


def test_derivative_opposes_measurement_increase():
    pid = PID(kp=0.0, kd=1.0, tau_d=0.0)
    pid.update(0.0, 0.0, 0.01)
    out = pid.update(0.0, 1.0, 0.01)
    assert out < 0.0
    assert np.isclose(out, -100.0)


def test_reset_clears_state():
    pid = PID(kp=1.0, ki=1.0, kd=1.0)
    for _ in range(5):
        pid.update(1.0, 0.0, 0.1)
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_meas is None


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        PID(kp=1.0).update(1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        PID(kp=1.0, out_min=1.0, out_max=-1.0).update(1.0, 0.0, 0.1)
