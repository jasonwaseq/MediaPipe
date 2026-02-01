"""
Right-side panels: Results (JSON), Logs, Performance.
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def _pretty_json(obj: Any) -> str:
    """Pretty-print dict/list for display."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


class ResultsPanel(QWidget):
    """Shows structured results as pretty-printed JSON, updated in real time."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setPlaceholderText("Results will appear here when a pipeline is running.")
        layout.addWidget(self._text, stretch=1)

    def update_results(self, results: dict[str, Any] | None) -> None:
        if results is None:
            self._text.setPlainText("")
            return
        self._text.setPlainText(_pretty_json(results))

    def get_current_text(self) -> str:
        return self._text.toPlainText()


class LogsPanel(QWidget):
    """Shows application and plugin log messages and errors."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

    def append(self, message: str) -> None:
        self._text.appendPlainText(message)
        # Auto-scroll to bottom
        scrollbar = self._text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self) -> None:
        self._text.clear()


class PerformancePanel(QWidget):
    """Shows FPS, per-frame latency (ms), and rolling average."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._fps_label = QLabel("FPS: —")
        self._latency_label = QLabel("Latency (ms): —")
        self._rolling_label = QLabel("Rolling avg (ms): —")
        for w in (self._fps_label, self._latency_label, self._rolling_label):
            layout.addWidget(w)
        layout.addStretch()

    def update_metrics(self, fps: float, latency_ms: float, rolling_avg_ms: float) -> None:
        self._fps_label.setText(f"FPS: {fps:.1f}")
        self._latency_label.setText(f"Latency (ms): {latency_ms:.1f}")
        self._rolling_label.setText(f"Rolling avg (ms): {rolling_avg_ms:.1f}")

    def reset(self) -> None:
        self._fps_label.setText("FPS: —")
        self._latency_label.setText("Latency (ms): —")
        self._rolling_label.setText("Rolling avg (ms): —")
