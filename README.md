# MediaPipe GUI

A desktop application that provides a GUI to run MediaPipe pipelines (Hands, Pose, Face Mesh, etc.) with live preview, structured results, performance metrics, and export options.

## Setup

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # or: source .venv/bin/activate  # Linux/macOS
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   python main.py
   ```

   The first time you **Start** a pipeline, the app will download the corresponding model file (`.task` or `.tflite`) from Google into the project’s `models/` folder. This is a one-time download per pipeline.

## Usage

- **Left sidebar — Input**
  - **Camera index** (0, 1, 2, 3): Choose which camera to use. **0** is usually your built-in webcam; **1** and above are external USB cameras. Pick the index that matches the camera you want.
  - **Open Video**: Use a video file from disk instead of a live camera.
  - **Pipeline**: Choose **Camera only** to see the raw feed, or pick a solution: **Hands**, **Pose**, **Face Mesh**, **Object Detector**, **Face Detector**, or **Gesture Recognizer**.
  - Click **Start** to begin; the main view will show the live (or video) feed, with overlays when a pipeline is selected.
- **Main view**: Live video with overlays (landmarks, connections).
- **Right tabs**:
  - **Results**: Pretty-printed JSON of the latest pipeline output (updates in real time).
  - **Logs**: Application and plugin messages/errors.
  - **Performance**: FPS, per-frame latency (ms), and rolling average.
- **Settings**: When you select a pipeline, a settings section appears below the pipeline dropdown (e.g. confidence thresholds). Adjust as needed.
- **Export**:
  - **Save Frame**: Save the current frame with overlays as PNG.
  - **Save Results JSON**: Save the latest structured results to a JSON file.
  - **Record Video**: Optionally record annotated video to disk (start/stop recording).
- **Explain Results**: Button that calls a placeholder `explain_results(results)` (no web calls); you can wire a real LLM later.

## Adding a New Plugin

1. Create a new Python file in the `plugins/` folder (e.g. `plugins/my_solution.py`).

2. Implement the plugin interface by defining a class that provides:
   - `plugin_id: str` — unique identifier (e.g. `"my_solution"`).
   - `display_name: str` — label shown in the pipeline dropdown.
   - `default_settings() -> dict` — default config (e.g. confidence thresholds).
   - `build_settings_widget(parent) -> QWidget` — Qt widget for editing settings.
   - `init(settings: dict)` — initialize the model with the given settings.
   - `process(frame_bgr: np.ndarray, timestamp_s: float) -> (annotated_frame_bgr, results: dict)` — process one frame; return annotated BGR frame and a results dict following the unified schema.
   - `close()` — release resources.

3. Use the unified results schema:

   ```python
   results = {
       "pipeline": plugin_id,
       "timestamp_s": float,
       "detections": [...],
       "landmarks": [...],
       "metadata": {...}
   }
   ```

4. Drop the file into `plugins/`. The app discovers plugins by importing modules in that folder that define a `Plugin` class or a `plugin` instance; the loader in `plugins/__init__.py` expects each module to expose a `plugin` object (instance of a class that implements the interface).

See `plugins/hands.py`, `plugins/pose.py`, and `plugins/facemesh.py` for examples.

## Project Structure

```
MediaPipe/
  README.md
  requirements.txt
  main.py
  core/           # Capture, runner, models, utils
  plugins/        # Pipeline plugins (hands, pose, facemesh)
  ui/             # Qt widgets and main window
  assets/         # Optional icons
```

## License

Use MediaPipe and dependencies according to their respective licenses.
