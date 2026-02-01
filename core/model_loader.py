"""
Ensures MediaPipe Tasks .task model files exist; downloads from Google storage if missing.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Directory for cached models (next to project root)
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODELS_DIR.mkdir(exist_ok=True)

# Official MediaPipe model URLs (Google storage)
_MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
}


def get_model_path(filename: str) -> Path:
    """Return path to the model file; download if not present."""
    path = _MODELS_DIR / filename
    if path.is_file():
        return path
    url = _MODEL_URLS.get(filename)
    if not url:
        raise FileNotFoundError(f"Unknown model: {filename}. Known: {list(_MODEL_URLS)}")
    urllib.request.urlretrieve(url, path)
    return path
