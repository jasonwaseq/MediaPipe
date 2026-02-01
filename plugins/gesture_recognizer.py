"""
MediaPipe Gesture Recognizer plugin: recognizes hand gestures (Thumb_Up, Pointing_Up, etc.) and draws hand landmarks.
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


class GestureRecognizerPlugin(MediaPipePluginBase):
    plugin_id = "gesture_recognizer"
    display_name = "Gesture Recognizer"

    def __init__(self) -> None:
        self._recognizer: mp.tasks.vision.GestureRecognizer | None = None

    @staticmethod
    def default_settings() -> dict[str, Any]:
        return {
            "num_hands": 2,
            "min_hand_detection_confidence": 0.5,
            "min_hand_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }

    @staticmethod
    def build_settings_widget(parent: QWidget | None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)
        num_hands = QSpinBox()
        num_hands.setRange(1, 4)
        num_hands.setValue(2)
        num_hands.setObjectName("num_hands")
        layout.addRow("Max hands:", num_hands)
        min_det = QDoubleSpinBox()
        min_det.setRange(0.0, 1.0)
        min_det.setSingleStep(0.05)
        min_det.setValue(0.5)
        min_det.setObjectName("min_hand_detection_confidence")
        layout.addRow("Min hand detection confidence:", min_det)
        min_presence = QDoubleSpinBox()
        min_presence.setRange(0.0, 1.0)
        min_presence.setSingleStep(0.05)
        min_presence.setValue(0.5)
        min_presence.setObjectName("min_hand_presence_confidence")
        layout.addRow("Min hand presence confidence:", min_presence)
        min_track = QDoubleSpinBox()
        min_track.setRange(0.0, 1.0)
        min_track.setSingleStep(0.05)
        min_track.setValue(0.5)
        min_track.setObjectName("min_tracking_confidence")
        layout.addRow("Min tracking confidence:", min_track)
        return widget

    def init(self, settings: dict[str, Any]) -> None:
        self.close()
        model_path = str(get_model_path("gesture_recognizer.task"))
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=int(settings.get("num_hands", 2)),
            min_hand_detection_confidence=float(
                settings.get("min_hand_detection_confidence", 0.5)
            ),
            min_hand_presence_confidence=float(
                settings.get("min_hand_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(settings.get("min_tracking_confidence", 0.5)),
        )
        self._recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(
            options
        )

    def process(
        self, frame_bgr: np.ndarray, timestamp_s: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._recognizer is None:
            return frame_bgr, unified_results_schema(
                self.plugin_id, timestamp_s, metadata={"error": "not initialized"}
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(timestamp_s * 1000)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
        annotated = frame_bgr.copy()
        drawing_utils = mp.tasks.vision.drawing_utils
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        detections_list: list[dict[str, Any]] = []
        landmarks_list: list[list[dict[str, float]]] = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                drawing_utils.draw_landmarks(
                    annotated, hand_landmarks, connections
                )
                points = [
                    {"x": lm.x, "y": lm.y, "z": lm.z or 0.0}
                    for lm in hand_landmarks
                ]
                landmarks_list.append(points)
        if result.gestures:
            for gesture_cats in result.gestures:
                if gesture_cats:
                    c = gesture_cats[0]
                    detections_list.append({
                        "label": c.category_name or "",
                        "score": c.score or 0.0,
                    })
        if result.handedness:
            for i, handedness_cats in enumerate(result.handedness):
                hand_label = handedness_cats[0].category_name if handedness_cats else ""
                if i < len(detections_list):
                    detections_list[i]["handedness"] = hand_label
        results = unified_results_schema(
            self.plugin_id,
            timestamp_s,
            detections=detections_list,
            landmarks=landmarks_list,
            metadata={"num_hands": len(landmarks_list)},
        )
        return annotated, results

    def close(self) -> None:
        self._recognizer = None


plugin = GestureRecognizerPlugin()
