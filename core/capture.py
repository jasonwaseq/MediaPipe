"""
Video capture: webcam by index or video file. Yields BGR frames and optional path.
"""

from __future__ import annotations

import sys

import cv2
from pathlib import Path
from typing import Generator


class VideoCaptureSource:
    """Unified source for webcam (by index) or video file."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._source_path: str | None = None  # None = webcam
        self._camera_index: int = 0

    def open_camera(self, index: int = 0) -> bool:
        """Open default or specified webcam. Returns True on success."""
        self.close()
        # On Windows, use DirectShow so index order matches enumerated camera list (pygrabber)
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(index)
        self._source_path = None
        self._camera_index = index
        return self._cap.isOpened()

    def open_file(self, path: str | Path) -> bool:
        """Open a video file. Returns True on success."""
        self.close()
        path_str = str(path)
        self._cap = cv2.VideoCapture(path_str)
        self._source_path = path_str
        return self._cap.isOpened()

    def close(self) -> None:
        """Release the current source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._source_path = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """Read next frame. Returns (success, frame_bgr)."""
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        return ok, frame

    def get_fps(self) -> float:
        """Get source FPS (e.g. for recording)."""
        if self._cap is None:
            return 30.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    def get_size(self) -> tuple[int, int]:
        """(width, height) of the stream."""
        if self._cap is None:
            return 0, 0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    @property
    def source_path(self) -> str | None:
        return self._source_path

    @property
    def camera_index(self) -> int:
        return self._camera_index
