"""
Utility helpers used across the app.
"""

import time
from collections import deque
from typing import Deque


class RollingAverage:
    """Rolling average over the last N values."""

    def __init__(self, maxlen: int = 30) -> None:
        self._values: Deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self._values.append(value)

    @property
    def average(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def count(self) -> int:
        return len(self._values)

    def clear(self) -> None:
        self._values.clear()


class FPSCounter:
    """Simple FPS and per-frame latency tracking."""

    def __init__(self, rolling_size: int = 30) -> None:
        self._last_time: float | None = None
        self._frame_times: Deque[float] = deque(maxlen=rolling_size)
        self._rolling = RollingAverage(maxlen=rolling_size)

    def tick(self) -> tuple[float, float]:
        """Call once per frame. Returns (fps, latency_ms)."""
        now = time.perf_counter()
        latency_ms = 0.0
        if self._last_time is not None:
            dt = now - self._last_time
            latency_ms = dt * 1000.0
            self._frame_times.append(dt)
            self._rolling.add(latency_ms)
        self._last_time = now
        fps = 1.0 / self._frame_times[-1] if self._frame_times else 0.0
        return fps, latency_ms

    @property
    def rolling_average_ms(self) -> float:
        return self._rolling.average

    def reset(self) -> None:
        self._last_time = None
        self._frame_times.clear()
        self._rolling.clear()
