"""
MediaPipe Face Mesh plugin using the Tasks API (FaceLandmarker).
"""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from core.model_loader import get_model_path
from core.models import unified_results_schema
from plugins.base import MediaPipePluginBase

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QSpinBox, QWidget


class FaceMeshPlugin(MediaPipePluginBase):
    plugin_id = "facemesh"
    display_name = "Face Mesh"

    def __init__(self) -> None:
        self._landmarker: mp.tasks.vision.FaceLandmarker | None = None

    @staticmethod
    def default_settings() -> dict[str, Any]:
        return {
            "num_faces": 1,
            "min_face_detection_confidence": 0.5,
            "min_face_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }

    @staticmethod
    def build_settings_widget(parent: QWidget | None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)
        num_faces = QSpinBox()
        num_faces.setRange(1, 4)
        num_faces.setValue(1)
        num_faces.setObjectName("num_faces")
        layout.addRow("Max faces:", num_faces)
        min_det = QDoubleSpinBox()
        min_det.setRange(0.0, 1.0)
        min_det.setSingleStep(0.05)
        min_det.setValue(0.5)
        min_det.setObjectName("min_face_detection_confidence")
        layout.addRow("Min face detection confidence:", min_det)
        min_presence = QDoubleSpinBox()
        min_presence.setRange(0.0, 1.0)
        min_presence.setSingleStep(0.05)
        min_presence.setValue(0.5)
        min_presence.setObjectName("min_face_presence_confidence")
        layout.addRow("Min face presence confidence:", min_presence)
        min_track = QDoubleSpinBox()
        min_track.setRange(0.0, 1.0)
        min_track.setSingleStep(0.05)
        min_track.setValue(0.5)
        min_track.setObjectName("min_tracking_confidence")
        layout.addRow("Min tracking confidence:", min_track)
        return widget

    def init(self, settings: dict[str, Any]) -> None:
        self.close()
        model_path = str(get_model_path("face_landmarker.task"))
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=int(settings.get("num_faces", 1)),
            min_face_detection_confidence=float(
                settings.get("min_face_detection_confidence", 0.5)
            ),
            min_face_presence_confidence=float(
                settings.get("min_face_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(settings.get("min_tracking_confidence", 0.5)),
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process(
        self, frame_bgr: np.ndarray, timestamp_s: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._landmarker is None:
            return frame_bgr, unified_results_schema(
                self.plugin_id, timestamp_s, metadata={"error": "not initialized"}
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(timestamp_s * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        annotated = frame_bgr.copy()
        drawing_utils = mp.tasks.vision.drawing_utils
        connections = mp.tasks.vision.FaceLandmarksConnections
        all_connections = (
            connections.FACE_LANDMARKS_LIPS
            + connections.FACE_LANDMARKS_LEFT_EYE
            + connections.FACE_LANDMARKS_LEFT_EYEBROW
            + connections.FACE_LANDMARKS_RIGHT_EYE
            + connections.FACE_LANDMARKS_RIGHT_EYEBROW
            + connections.FACE_LANDMARKS_FACE_OVAL
            + connections.FACE_LANDMARKS_NOSE
        )
        landmarks_list: list[list[dict[str, float]]] = []
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                drawing_utils.draw_landmarks(
                    annotated, face_landmarks, all_connections
                )
                points = [
                    {"x": lm.x, "y": lm.y, "z": lm.z or 0.0}
                    for lm in face_landmarks
                ]
                landmarks_list.append(points)
        results = unified_results_schema(
            self.plugin_id,
            timestamp_s,
            detections=[],
            landmarks=landmarks_list,
            metadata={"num_faces": len(landmarks_list)},
        )
        return annotated, results

    def close(self) -> None:
        self._landmarker = None


plugin = FaceMeshPlugin()
