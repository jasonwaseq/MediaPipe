"""
MediaPipe Pose plugin using the Tasks API (PoseLandmarker).
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


class PosePlugin(MediaPipePluginBase):
    plugin_id = "pose"
    display_name = "Pose"

    def __init__(self) -> None:
        self._landmarker: mp.tasks.vision.PoseLandmarker | None = None

    @staticmethod
    def default_settings() -> dict[str, Any]:
        return {
            "num_poses": 1,
            "min_pose_detection_confidence": 0.5,
            "min_pose_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }

    @staticmethod
    def build_settings_widget(parent: QWidget | None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)
        num_poses = QSpinBox()
        num_poses.setRange(1, 4)
        num_poses.setValue(1)
        num_poses.setObjectName("num_poses")
        layout.addRow("Max poses:", num_poses)
        min_det = QDoubleSpinBox()
        min_det.setRange(0.0, 1.0)
        min_det.setSingleStep(0.05)
        min_det.setValue(0.5)
        min_det.setObjectName("min_pose_detection_confidence")
        layout.addRow("Min pose detection confidence:", min_det)
        min_presence = QDoubleSpinBox()
        min_presence.setRange(0.0, 1.0)
        min_presence.setSingleStep(0.05)
        min_presence.setValue(0.5)
        min_presence.setObjectName("min_pose_presence_confidence")
        layout.addRow("Min pose presence confidence:", min_presence)
        min_track = QDoubleSpinBox()
        min_track.setRange(0.0, 1.0)
        min_track.setSingleStep(0.05)
        min_track.setValue(0.5)
        min_track.setObjectName("min_tracking_confidence")
        layout.addRow("Min tracking confidence:", min_track)
        return widget

    def init(self, settings: dict[str, Any]) -> None:
        self.close()
        model_path = str(get_model_path("pose_landmarker_lite.task"))
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=int(settings.get("num_poses", 1)),
            min_pose_detection_confidence=float(
                settings.get("min_pose_detection_confidence", 0.5)
            ),
            min_pose_presence_confidence=float(
                settings.get("min_pose_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(settings.get("min_tracking_confidence", 0.5)),
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

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
        connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        landmarks_list: list[list[dict[str, float]]] = []
        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:
                drawing_utils.draw_landmarks(
                    annotated, pose_landmarks, connections
                )
                points = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z or 0.0,
                        "visibility": (lm.visibility if lm.visibility is not None else 0.0),
                    }
                    for lm in pose_landmarks
                ]
                landmarks_list.append(points)
        results = unified_results_schema(
            self.plugin_id,
            timestamp_s,
            detections=[],
            landmarks=landmarks_list,
            metadata={"has_pose": len(landmarks_list) > 0},
        )
        return annotated, results

    def close(self) -> None:
        self._landmarker = None


plugin = PosePlugin()
