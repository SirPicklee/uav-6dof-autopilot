import numpy as np

from uavsim import quaternion as quat
from uavsim.guidance import L1Navigator, WaypointNavigator
from uavsim.lateral import MAX_BANK, LateralAutopilot, wrap_pi


def level_state(heading=0.0, yaw_rate=0.0):
    q = quat.from_euler(0.0, 0.0, heading)
    return np.concatenate([[0.0, 0.0, -1000.0], [50.0, 0.0, 0.0], q, [0.0, 0.0, yaw_rate]])


def test_wrap_pi_boundaries():
    assert np.isclose(wrap_pi(0.0), 0.0)
    assert np.isclose(wrap_pi(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_pi(-3 * np.pi / 2), np.pi / 2)
    assert np.isclose(abs(wrap_pi(np.pi)), np.pi)


def test_right_heading_error_commands_positive_bank():
    ap = LateralAutopilot(target_heading=np.radians(20.0))
    ap.update(level_state(0.0), 0.02)
    assert ap.bank_cmd > 0.0


def test_left_heading_error_commands_negative_bank():
    ap = LateralAutopilot(target_heading=np.radians(-20.0))
    ap.update(level_state(0.0), 0.02)
    assert ap.bank_cmd < 0.0


def test_bank_command_is_limited():
    ap = LateralAutopilot(target_heading=np.radians(170.0))
    ap.update(level_state(0.0), 0.02)
    assert np.isclose(ap.bank_cmd, MAX_BANK)


def test_heading_error_wraps_through_north():
    # Heading 350 deg, target 10 deg: shortest turn is right (positive bank)
    ap = LateralAutopilot(target_heading=np.radians(10.0))
    ap.update(level_state(np.radians(-10.0)), 0.02)
    assert ap.bank_cmd > 0.0


def test_yaw_damper_washes_out_constant_rate():
    ap = LateralAutopilot(target_heading=0.0)
    y = level_state(0.0, yaw_rate=0.1)
    rud = 0.0
    for _ in range(int(30.0 / 0.02)):
        _, rud = ap.update(y, 0.02)
    assert abs(rud) < 1e-3


def test_waypoint_navigator_bearing_and_switch():
    nav = WaypointNavigator(np.array([[1000.0, 0.0], [1000.0, 1000.0]]), capture_radius=300.0)
    assert np.isclose(nav.heading_command(0.0, 0.0, 0.0), 0.0)
    assert nav.index == 0
    nav.heading_command(800.0, 0.0, 10.0)     # 200 m from WP1 -> captured
    assert nav.index == 1
    assert len(nav.captures) == 1
    assert np.isclose(nav.heading_command(800.0, 0.0, 10.0), np.arctan2(1000.0, 200.0))


def test_waypoint_navigator_finishes():
    nav = WaypointNavigator(np.array([[100.0, 0.0]]), capture_radius=300.0)
    nav.heading_command(0.0, 0.0, 0.0)
    assert nav.done


def test_l1_heads_along_leg_when_on_track():
    nav = L1Navigator(np.array([[2000.0, 0.0]]))
    assert np.isclose(nav.heading_command(500.0, 0.0, 0.0), 0.0)


def test_l1_steers_back_toward_line():
    nav = L1Navigator(np.array([[2000.0, 0.0]]))
    # Aircraft 100 m east of the line: must steer west of north (negative heading)
    assert nav.heading_command(500.0, 100.0, 0.0) < 0.0


def test_l1_switches_leg_at_turn_lead():
    nav = L1Navigator(np.array([[2000.0, 0.0], [2000.0, 2000.0]]), turn_lead=441.0)
    nav.heading_command(1500.0, 0.0, 0.0)       # 500 m before WP1: still leg 1
    assert nav.index == 0
    nav.heading_command(1600.0, 0.0, 5.0)       # 400 m before WP1: switch
    assert nav.index == 1
    assert len(nav.captures) == 1
