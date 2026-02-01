"""
MediaPipe GUI — entry point.
Run: python main.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Reduce TensorFlow/MediaPipe console noise (INFO and WARNING)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def explain_results(results: dict[str, Any]) -> str:
    """
    Placeholder for LLM explanation. Returns a stub message.
    Wire your real LLM here later (no web calls or external keys in this stub).
    """
    if not results:
        return "No results to explain. Start a pipeline and run a frame first."
    pipeline = results.get("pipeline", "?")
    ts = results.get("timestamp_s", 0)
    det = len(results.get("detections", []))
    land = len(results.get("landmarks", []))
    return (
        f"Explanation placeholder: wire your LLM here to explain the current results.\n\n"
        f"Pipeline: {pipeline}\n"
        f"Timestamp: {ts:.3f}s\n"
        f"Detections: {det}\n"
        f"Landmark groups: {land}\n\n"
        "Replace this function with a call to your LLM API to get a natural-language explanation."
    )


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow(explain_results_fn=explain_results)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
