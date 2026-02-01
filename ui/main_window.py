"""
Main window: left sidebar (source, start/stop, pipeline, settings), center video, right tabs.
Export and Explain Results wiring.
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.camera_list import get_camera_list
from core.capture import VideoCaptureSource
from core.runner import FrameProcessorRunner
from plugins.base import MediaPipePluginBase
from plugins import discover_plugins
from ui.panels import LogsPanel, PerformancePanel, ResultsPanel


class PluginInitWorker(QObject):
    """Runs plugin.init(settings) in a background thread so the UI stays responsive."""

    init_done = Signal(bool, str, object)  # success, error_message, plugin

    def __init__(self, plugin: MediaPipePluginBase, settings: dict[str, Any]) -> None:
        super().__init__()
        self._plugin = plugin
        self._settings = settings

    def run(self) -> None:
        try:
            self._plugin.init(self._settings)
            self.init_done.emit(True, "", self._plugin)
        except Exception as e:
            self.init_done.emit(False, str(e), self._plugin)


def get_settings_from_widget(widget: QWidget) -> dict[str, Any]:
    """Collect current settings from a plugin's settings widget (spinboxes by objectName)."""
    from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox
    out: dict[str, Any] = {}
    for child in list(widget.findChildren(QDoubleSpinBox)) + list(widget.findChildren(QSpinBox)):
        name = child.objectName()
        if name:
            val = child.value()
            if isinstance(child, QSpinBox):
                out[name] = int(val)
            else:
                out[name] = float(val)
    return out


class MainWindow(QWidget):
    """Main application window with sidebar, video view, and right panels."""

    def __init__(self, explain_results_fn: Callable[[dict], str] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("MediaPipe GUI")
        self._explain_results_fn = explain_results_fn
        self._capture = VideoCaptureSource()
        self._runner: FrameProcessorRunner | None = None
        self._plugins: list[MediaPipePluginBase] = []
        self._current_plugin: MediaPipePluginBase | None = None
        self._current_frame: cv2.typing.MatLike | None = None
        self._latest_results: dict[str, Any] = {}
        self._recording = False
        self._video_writer: cv2.VideoWriter | None = None
        self._rolling_avg_ms = 0.0
        self._init_thread: QThread | None = None
        self._init_worker: PluginInitWorker | None = None

        layout = QHBoxLayout(self)
        # --- Left sidebar ---
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.addWidget(QLabel("Input"))
        input_hint = QLabel("Select a camera by name, or open a video file.")
        input_hint.setWordWrap(True)
        input_hint.setStyleSheet("color: #666; font-size: 11px;")
        sidebar_layout.addWidget(input_hint)
        self._camera_combo = QComboBox()
        self._camera_combo.setToolTip("Select the camera to use. Names match your system (e.g. built-in vs USB).")
        sidebar_layout.addWidget(self._camera_combo)
        refresh_cam_btn = QPushButton("Refresh cameras")
        refresh_cam_btn.setToolTip("Re-detect connected cameras.")
        refresh_cam_btn.clicked.connect(self._refresh_cameras)
        sidebar_layout.addWidget(refresh_cam_btn)
        self._refresh_cameras()
        self._open_video_btn = QPushButton("Open Video")
        self._open_video_btn.setToolTip("Use a video file instead of the camera.")
        self._open_video_btn.clicked.connect(self._on_open_video)
        sidebar_layout.addWidget(self._open_video_btn)
        sidebar_layout.addWidget(QLabel("Pipeline"))
        self._pipeline_combo = QComboBox()
        self._pipeline_combo.currentIndexChanged.connect(self._on_pipeline_changed)
        sidebar_layout.addWidget(self._pipeline_combo)
        self._settings_stack = QStackedWidget()
        self._settings_placeholder = QLabel("Select a pipeline for settings.")
        self._settings_stack.addWidget(self._settings_placeholder)
        settings_group = QGroupBox("Settings")
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(self._settings_stack)
        settings_inner = QVBoxLayout()
        settings_inner.addWidget(settings_scroll)
        settings_group.setLayout(settings_inner)
        sidebar_layout.addWidget(settings_group)
        self._start_stop_btn = QPushButton("Start")
        self._start_stop_btn.clicked.connect(self._on_start_stop)
        sidebar_layout.addWidget(self._start_stop_btn)
        sidebar_layout.addStretch()
        layout.addWidget(sidebar)

        # --- Center: video ---
        self._video_label = QLabel()
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet("background-color: #1e1e1e; color: #888;")
        self._video_label.setText("No video")
        layout.addWidget(self._video_label, stretch=1)

        # --- Right: tabs (Results, Logs, Performance, Export) — full height of sidebar ---
        self._save_frame_btn = QPushButton("Save Frame (PNG)")
        self._save_frame_btn.clicked.connect(self._on_save_frame)
        self._save_json_btn = QPushButton("Save Results JSON")
        self._save_json_btn.clicked.connect(self._on_save_json)
        self._record_btn = QPushButton("Start Recording")
        self._record_btn.clicked.connect(self._on_toggle_recording)
        self._explain_btn = QPushButton("Explain Results")
        self._explain_btn.clicked.connect(self._on_explain_results)
        tabs = QTabWidget()
        self._results_panel = ResultsPanel()
        tabs.addTab(self._results_panel, "Results")
        self._logs_panel = LogsPanel()
        tabs.addTab(self._logs_panel, "Logs")
        self._performance_panel = PerformancePanel()
        tabs.addTab(self._performance_panel, "Performance")
        export_panel = QWidget()
        export_layout = QVBoxLayout(export_panel)
        export_layout.addWidget(self._save_frame_btn)
        export_layout.addWidget(self._save_json_btn)
        export_layout.addWidget(self._record_btn)
        export_layout.addWidget(self._explain_btn)
        export_layout.addStretch()
        tabs.addTab(export_panel, "Export")
        layout.addWidget(tabs)

        self._load_plugins()
        self._logs_panel.append("Application started. Select input and pipeline, then Start.")
        self.resize(1200, 700)

    def _refresh_cameras(self) -> None:
        """Populate camera combo with (index, name) from get_camera_list()."""
        cameras = get_camera_list()
        self._camera_combo.clear()
        for index, name in cameras:
            self._camera_combo.addItem(name, index)
        if not cameras:
            self._camera_combo.addItem("No cameras found", 0)
            self._logs_panel.append("No cameras detected. Connect a camera and click Refresh cameras.")

    def _load_plugins(self) -> None:
        self._plugins = discover_plugins()
        self._pipeline_combo.clear()
        self._pipeline_combo.addItem("Camera only (no pipeline)", None)
        for p in self._plugins:
            self._pipeline_combo.addItem(p.display_name, p)
            settings_widget = p.build_settings_widget(self._settings_stack)
            self._settings_stack.addWidget(settings_widget)
        self._pipeline_combo.setCurrentIndex(0)
        self._on_pipeline_changed(0)

    def _on_pipeline_changed(self, index: int) -> None:
        if index < 0:
            return
        if index == 0:
            self._current_plugin = None
            self._settings_stack.setCurrentIndex(0)
            return
        # index 1..n = plugins 0..n-1
        self._current_plugin = self._plugins[index - 1]
        self._settings_stack.setCurrentIndex(index)

    def _on_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video (*.mp4 *.avi *.mov *.mkv);;All (*)"
        )
        if not path:
            return
        if self._capture.open_file(path):
            self._logs_panel.append(f"Opened video: {path}")
        else:
            self._logs_panel.append(f"Failed to open video: {path}")

    def _on_start_stop(self) -> None:
        if self._runner is not None and self._capture.is_opened():
            self._stop_processing()
            return
        if not self._capture.is_opened():
            cam_index = self._camera_combo.currentData()
            if cam_index is None:
                cam_index = 0
            cam_name = self._camera_combo.currentText()
            if not self._capture.open_camera(cam_index):
                self._logs_panel.append(f"Failed to open camera: {cam_name!r} (index {cam_index}).")
                return
            self._logs_panel.append(f"Opened camera: {cam_name} (index {cam_index}).")
        plugin = self._current_plugin
        if plugin is None:
            plugin = self._pipeline_combo.currentData()
        if plugin is not None:
            settings_widget = self._settings_stack.currentWidget()
            if settings_widget is self._settings_placeholder:
                settings = plugin.default_settings()
            else:
                settings = get_settings_from_widget(settings_widget)
            self._start_stop_btn.setEnabled(False)
            self._start_stop_btn.setText("Loading...")
            self._logs_panel.append("Loading pipeline (may take a few seconds)...")
            self._init_worker = PluginInitWorker(plugin, settings)
            self._init_thread = QThread()
            self._init_worker.moveToThread(self._init_thread)
            self._init_thread.started.connect(self._init_worker.run)
            self._init_worker.init_done.connect(self._on_plugin_init_done)
            self._init_thread.start()
            return
        self._start_runner(plugin)

    def _on_plugin_init_done(self, success: bool, error_msg: str, plugin: MediaPipePluginBase | None) -> None:
        self._start_stop_btn.setEnabled(True)
        self._start_stop_btn.setText("Start")
        if self._init_thread is not None:
            self._init_thread.quit()
            self._init_thread.wait(2000)
            self._init_thread = None
        self._init_worker = None
        if not success:
            self._logs_panel.append(f"Plugin init error: {error_msg}")
            return
        self._start_runner(plugin)

    def _start_runner(self, plugin: MediaPipePluginBase | None) -> None:
        self._runner = FrameProcessorRunner(self._capture)
        self._runner.set_plugin(plugin)
        self._runner.frame_processed.connect(self._on_frame_processed)
        self._runner.error_occurred.connect(self._on_runner_error)
        self._runner.stopped.connect(self._on_runner_stopped)
        self._runner.start()
        self._start_stop_btn.setText("Stop")
        self._performance_panel.reset()
        self._logs_panel.append("Processing started.")

    def _stop_processing(self) -> None:
        if self._runner is None:
            return
        self._runner.stop()
        self._runner.finish_thread()
        if self._current_plugin:
            try:
                self._current_plugin.close()
            except Exception:
                pass
        self._runner = None
        self._capture.close()
        self._start_stop_btn.setText("Start")
        self._performance_panel.reset()
        self._logs_panel.append("Processing stopped.")
        if self._recording:
            self._stop_recording()

    @Slot(object, object, float, float, float)
    def _on_frame_processed(
        self,
        annotated_frame: cv2.typing.MatLike,
        results: dict,
        fps: float,
        latency_ms: float,
        rolling_avg_ms: float,
    ) -> None:
        self._current_frame = annotated_frame
        self._latest_results = results
        self._rolling_avg_ms = rolling_avg_ms
        self._results_panel.update_results(results)
        self._performance_panel.update_metrics(fps, latency_ms, rolling_avg_ms)
        h, w = annotated_frame.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(
            annotated_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_BGR888,
        )
        self._video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        if self._video_writer is not None:
            self._video_writer.write(annotated_frame)

    @Slot(str)
    def _on_runner_error(self, message: str) -> None:
        self._logs_panel.append(f"Error: {message}")
        self._stop_processing()

    @Slot()
    def _on_runner_stopped(self) -> None:
        if self._runner is not None:
            self._runner.finish_thread()
            self._runner = None
        self._capture.close()
        self._start_stop_btn.setText("Start")
        self._performance_panel.reset()
        if self._recording:
            self._stop_recording()

    def _on_save_frame(self) -> None:
        if self._current_frame is None:
            self._logs_panel.append("No frame to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame", "", "PNG (*.png);;All (*)"
        )
        if not path:
            return
        if cv2.imwrite(path, self._current_frame):
            self._logs_panel.append(f"Saved frame: {path}")
        else:
            self._logs_panel.append(f"Failed to save: {path}")

    def _on_save_json(self) -> None:
        if not self._latest_results:
            self._logs_panel.append("No results to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results JSON", "", "JSON (*.json);;All (*)"
        )
        if not path:
            return
        import json
        try:
            with open(path, "w") as f:
                json.dump(self._latest_results, f, indent=2, default=str)
            self._logs_panel.append(f"Saved results: {path}")
        except Exception as e:
            self._logs_panel.append(f"Failed to save JSON: {e}")

    def _on_toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Record Video", "", "MP4 (*.mp4);;All (*)"
        )
        if not path:
            return
        if self._current_frame is None:
            self._logs_panel.append("Start processing first to begin recording.")
            return
        h, w = self._current_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = self._capture.get_fps()
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not writer.isOpened():
            self._logs_panel.append("Failed to create video writer.")
            return
        self._video_writer = writer
        self._recording = True
        self._record_btn.setText("Stop Recording")
        self._record_path = path
        self._logs_panel.append(f"Recording to: {path}")

    def _stop_recording(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._recording = False
        self._record_btn.setText("Start Recording")
        if getattr(self, "_record_path", None):
            self._logs_panel.append(f"Recording saved: {self._record_path}")

    def _on_explain_results(self) -> None:
        if self._explain_results_fn is not None:
            text = self._explain_results_fn(self._latest_results)
            QMessageBox.information(self, "Explain Results", text)
        else:
            QMessageBox.information(
                self, "Explain Results", "No explain function configured."
            )

    def closeEvent(self, event) -> None:
        self._stop_processing()
        self._capture.close()
        if self._video_writer is not None:
            self._video_writer.release()
        event.accept()
