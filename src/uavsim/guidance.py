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
