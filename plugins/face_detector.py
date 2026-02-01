"""
MediaPipe Face Detector plugin: detects faces with bounding boxes (no landmarks).
"""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from core.model_loader import get_model_path
from core.models import unified_results_schema
from plugins.base import MediaPipePluginBase

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QWidget


class FaceDetectorPlugin(MediaPipePluginBase):
    plugin_id = "face_detector"
    display_name = "Face Detector"

    def __init__(self) -> None:
        self._detector: mp.tasks.vision.FaceDetector | None = None

    @staticmethod
    def default_settings() -> dict[str, Any]:
        return {
            "min_detection_confidence": 0.5,
            "min_suppression_threshold": 0.3,
        }

    @staticmethod
    def build_settings_widget(parent: QWidget | None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)
        min_det = QDoubleSpinBox()
        min_det.setRange(0.0, 1.0)
        min_det.setSingleStep(0.05)
        min_det.setValue(0.5)
        min_det.setObjectName("min_detection_confidence")
        layout.addRow("Min detection confidence:", min_det)
        min_supp = QDoubleSpinBox()
        min_supp.setRange(0.0, 1.0)
        min_supp.setSingleStep(0.05)
        min_supp.setValue(0.3)
        min_supp.setObjectName("min_suppression_threshold")
        layout.addRow("Min suppression threshold:", min_supp)
        return widget

    def init(self, settings: dict[str, Any]) -> None:
        self.close()
        model_path = str(get_model_path("blaze_face_short_range.tflite"))
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=float(
                settings.get("min_detection_confidence", 0.5)
            ),
            min_suppression_threshold=float(
                settings.get("min_suppression_threshold", 0.3)
            ),
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def process(
        self, frame_bgr: np.ndarray, timestamp_s: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._detector is None:
            return frame_bgr, unified_results_schema(
                self.plugin_id, timestamp_s, metadata={"error": "not initialized"}
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(timestamp_s * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        annotated = frame_bgr.copy()
        drawing_utils = mp.tasks.vision.drawing_utils
        detections_list: list[dict[str, Any]] = []
        if result.detections:
            for det in result.detections:
                drawing_utils.draw_detection(annotated, det)
                score = 0.0
                if det.categories:
                    score = det.categories[0].score or 0.0
                box = det.bounding_box
                detections_list.append({
                    "label": "face",
                    "score": score,
                    "bbox": {
                        "origin_x": box.origin_x,
                        "origin_y": box.origin_y,
                        "width": box.width,
                        "height": box.height,
                    },
                })
        results = unified_results_schema(
            self.plugin_id,
            timestamp_s,
            detections=detections_list,
            landmarks=[],
            metadata={"num_faces": len(detections_list)},
        )
        return annotated, results

    def close(self) -> None:
        self._detector = None


plugin = FaceDetectorPlugin()
