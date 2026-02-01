"""
Enumerate cameras with display names. On Windows uses DirectShow (pygrabber) for exact names;
same device order as OpenCV with CAP_DSHOW.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import cv2


def _probe_opencv(max_cameras: int = 16) -> List[Tuple[int, str]]:
    """Probe indices 0..max_cameras-1; return (index, 'Camera N') for each that opens."""
    result: List[Tuple[int, str]] = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            result.append((i, f"Camera {i}"))
            cap.release()
    return result


def get_camera_list() -> List[Tuple[int, str]]:
    """
    Return list of (index, display_name) for available cameras.
    On Windows with pygrabber: uses DirectShow for exact device names (same order as CAP_DSHOW).
    Otherwise: probes with OpenCV and returns "Camera 0", "Camera 1", etc.
    """
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph

            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(enumerate(devices))
        except ImportError:
            pass
    return _probe_opencv()
