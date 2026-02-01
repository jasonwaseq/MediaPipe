"""
Frame processor runner: runs on a worker thread, emits annotated frames and results.
Uses QThread + signals so the UI never blocks.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QThread, Signal

from core.capture import VideoCaptureSource
from core.utils import FPSCounter

if TYPE_CHECKING:
    import numpy as np
    from plugins.base import MediaPipePluginBase


class FrameProcessorRunner(QObject):
    """Worker that grabs frames, runs the active plugin, and emits results."""

    # Emit (annotated_frame_bgr, results_dict, fps, latency_ms, rolling_avg_ms)
    frame_processed = Signal(object, object, float, float, float)
    # Emit error message
    error_occurred = Signal(str)
    # Emit when stopped (e.g. after plugin failure)
    stopped = Signal()

    def __init__(
        self,
        capture: VideoCaptureSource,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._capture = capture
        self._plugin: MediaPipePluginBase | None = None
        self._running = False
        self._thread: QThread | None = None
        self._fps_counter = FPSCounter()

    def set_plugin(self, plugin: MediaPipePluginBase | None) -> None:
        self._plugin = plugin

    def start(self) -> None:
        """Start processing in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run_loop)
        self._thread.start()

    def stop(self) -> None:
        """Request stop; run loop will exit and thread will finish."""
        self._running = False

    def _run_loop(self) -> None:
        """Runs in worker thread: read frame -> process -> emit."""
        self._fps_counter.reset()
        while self._running and self._capture.is_opened():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                break
            timestamp_s = time.perf_counter()
            if self._plugin is None:
                self.frame_processed.emit(frame, {}, 0.0, 0.0, 0.0)
                continue
            try:
                annotated, results = self._plugin.process(frame, timestamp_s)
                fps, latency_ms = self._fps_counter.tick()
                rolling_ms = self._fps_counter.rolling_average_ms
                self.frame_processed.emit(annotated, results, fps, latency_ms, rolling_ms)
            except Exception as e:  # noqa: BLE001
                self.error_occurred.emit(str(e))
                self._running = False
                self.stopped.emit()
                return
        self.stopped.emit()

    def finish_thread(self) -> None:
        """Call after stopped signal: quit and wait for thread."""
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
