"""
MediaPipe Object Detector plugin: detects objects (COCO classes) with bounding boxes.
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


class ObjectDetectorPlugin(MediaPipePluginBase):
    plugin_id = "object_detector"
    display_name = "Object Detector"

    def __init__(self) -> None:
        self._detector: mp.tasks.vision.ObjectDetector | None = None

    @staticmethod
    def default_settings() -> dict[str, Any]:
        return {
            "max_results": 5,
            "score_threshold": 0.3,
        }

    @staticmethod
    def build_settings_widget(parent: QWidget | None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)
        max_results = QSpinBox()
        max_results.setRange(1, 20)
        max_results.setValue(5)
        max_results.setObjectName("max_results")
        layout.addRow("Max results:", max_results)
        score_thresh = QDoubleSpinBox()
        score_thresh.setRange(0.0, 1.0)
        score_thresh.setSingleStep(0.05)
        score_thresh.setValue(0.3)
        score_thresh.setObjectName("score_threshold")
        layout.addRow("Score threshold:", score_thresh)
        return widget

    def init(self, settings: dict[str, Any]) -> None:
        self.close()
        model_path = str(get_model_path("efficientdet_lite0.tflite"))
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            max_results=int(settings.get("max_results", 5)),
            score_threshold=float(settings.get("score_threshold", 0.3)),
        )
        self._detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

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
                label = ""
                score = 0.0
                if det.categories:
                    c = det.categories[0]
                    label = c.category_name or ""
                    score = c.score or 0.0
                box = det.bounding_box
                detections_list.append({
                    "label": label,
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
            metadata={"num_detections": len(detections_list)},
        )
        return annotated, results

    def close(self) -> None:
        self._detector = None


plugin = ObjectDetectorPlugin()
