"""Waypoint guidance: direct-to-waypoint bearing with a capture radius."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WaypointNavigator:
    """Sequentially fly to (north, east) waypoints in meters."""

    waypoints: np.ndarray
    capture_radius: float = 300.0
    index: int = field(default=0, init=False)
    captures: list = field(default_factory=list, init=False)
    _min_dist: float = field(default=float("inf"), init=False, repr=False)
    _last_heading: float = field(default=0.0, init=False, repr=False)

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def heading_command(self, pn: float, pe: float, t: float) -> float:
        """Return the heading (rad) toward the active waypoint."""
        while not self.done:
            wn, we = self.waypoints[self.index]
            dn, de = wn - pn, we - pe
            dist = float(np.hypot(dn, de))
            self._min_dist = min(self._min_dist, dist)
            if dist >= self.capture_radius:
                self._last_heading = float(np.arctan2(de, dn))
                return self._last_heading
            self.captures.append((self.index, t, self._min_dist))
            self.index += 1
            self._min_dist = float("inf")
        return self._last_heading

@dataclass
class L1Navigator:
    """Leg-following guidance: aim at a point L1 ahead on the current leg.

    The leg switches when the aircraft is within `turn_lead` meters (along the
    leg) of the waypoint, so the turn starts early and the corner overshoot
    stays small. Start position defines the beginning of the first leg.
    """

    waypoints: np.ndarray
    start: np.ndarray = field(default_factory=lambda: np.zeros(2))
    l1: float = 400.0
    turn_lead: float = 441.0
    index: int = field(default=0, init=False)
    captures: list = field(default_factory=list, init=False)
    _last_heading: float = field(default=0.0, init=False, repr=False)

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def _leg(self) -> tuple[np.ndarray, np.ndarray]:
        a = self.start if self.index == 0 else self.waypoints[self.index - 1]
        return np.asarray(a, dtype=float), np.asarray(self.waypoints[self.index], dtype=float)

    def heading_command(self, pn: float, pe: float, t: float) -> float:
        pos = np.array([pn, pe])
        while not self.done:
            a, b = self._leg()
            leg = b - a
            length = float(np.hypot(leg[0], leg[1]))
            u = leg / length
            along = float((pos - a) @ u)
            if along >= length - self.turn_lead:
                self.captures.append((self.index, t, float(np.hypot(*(b - pos)))))
                self.index += 1
                continue
            # Cross-track (right of track positive) and rabbit point
            rel = pos - a
            xt = float(rel[0] * u[1] - rel[1] * u[0])
            ahead = min(max(along, 0.0) + self.l1, length)
            target = a + u * ahead
            # Pull toward the leg line when far off, never aim behind
            target = target - np.array([u[1], -u[0]]) * 0.0 * xt
            self._last_heading = float(np.arctan2(target[1] - pe, target[0] - pn))
            return self._last_heading
        return self._last_heading
