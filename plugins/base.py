"""
Base plugin interface that every MediaPipe pipeline plugin must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from PySide6.QtWidgets import QWidget


class MediaPipePluginBase(ABC):
    """Interface for pipeline plugins. Subclass and implement all methods."""

    plugin_id: str = ""
    display_name: str = ""

    @staticmethod
    @abstractmethod
    def default_settings() -> dict[str, Any]:
        """Return default settings dict (e.g. min_detection_confidence)."""
        ...

    @staticmethod
    @abstractmethod
    def build_settings_widget(parent: QWidget | None) -> QWidget:
        """Build and return a Qt widget for editing settings."""
        ...

    @abstractmethod
    def init(self, settings: dict[str, Any]) -> None:
        """Initialize the model with the given settings."""
        ...

    @abstractmethod
    def process(
        self, frame_bgr: np.ndarray, timestamp_s: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Process one frame. Return (annotated_frame_bgr, results_dict).
        results_dict must follow the unified schema:
          pipeline, timestamp_s, detections, landmarks, metadata
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources (e.g. MediaPipe solution instance)."""
        ...
