import numpy as np
import pytest

from uavsim.atmosphere import (
    atmosphere,
    dynamic_pressure,
    mach_number,
    T0,
    P0,
    RHO0,
)


def test_sea_level_values():
    s = atmosphere(0.0)
    assert np.isclose(s.temperature, 288.15)
    assert np.isclose(s.pressure, 101325.0)
    assert np.isclose(s.density, RHO0, rtol=1e-3)
    assert np.isclose(s.speed_of_sound, 340.294, rtol=1e-4)


def test_5000m_matches_isa_table():
    s = atmosphere(5000.0)
    assert np.isclose(s.temperature, 255.65, atol=1e-2)
    assert np.isclose(s.pressure, 54019.9, rtol=1e-3)
    assert np.isclose(s.density, 0.7364, rtol=1e-3)


def test_tropopause_11km_matches_isa_table():
    s = atmosphere(11000.0)
    assert np.isclose(s.temperature, 216.65, atol=1e-2)
    assert np.isclose(s.pressure, 22632.1, rtol=1e-3)
    assert np.isclose(s.density, 0.3639, rtol=1e-3)
    assert np.isclose(s.speed_of_sound, 295.07, rtol=1e-3)


def test_20km_matches_isa_table():
    s = atmosphere(20000.0)
    assert np.isclose(s.temperature, 216.65, atol=1e-2)
    assert np.isclose(s.pressure, 5474.89, rtol=1e-3)
    assert np.isclose(s.density, 0.08803, rtol=1e-3)


def test_density_decreases_with_altitude():
    heights = np.linspace(0.0, 20000.0, 50)
    rho = [atmosphere(h).density for h in heights]
    assert all(a > b for a, b in zip(rho, rho[1:]))


def test_pressure_is_continuous_at_tropopause():
    below = atmosphere(10999.99).pressure
    above = atmosphere(11000.01).pressure
    assert np.isclose(below, above, rtol=1e-5)


def test_hot_day_lowers_density():
    std = atmosphere(3000.0)
    hot = atmosphere(3000.0, delta_isa=15.0)
    assert hot.density < std.density
    assert np.isclose(hot.pressure, std.pressure)


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        atmosphere(-1.0)
    with pytest.raises(ValueError):
        atmosphere(25000.0)


def test_dynamic_pressure_and_mach():
    assert np.isclose(dynamic_pressure(1.225, 100.0), 6125.0)
    assert np.isclose(mach_number(340.294, 340.294), 1.0)
