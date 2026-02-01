"""
Plugin loader: discover plugins by scanning plugins/ folder and importing
modules that define a 'plugin' instance (or a 'Plugin' class we instantiate).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.base import MediaPipePluginBase

# Directory containing this file = plugins/
_PLUGINS_DIR = Path(__file__).resolve().parent

# Built-in plugin module names (no .py); we only load these to avoid loading __init__ etc.
_BUILTIN_PLUGINS = (
    "hands",
    "pose",
    "facemesh",
    "object_detector",
    "face_detector",
    "gesture_recognizer",
)


def _load_plugin_from_module(module_name: str) -> MediaPipePluginBase | None:
    """Load a single plugin from a module in plugins/."""
    file_path = _PLUGINS_DIR / f"{module_name}.py"
    if not file_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        f"plugins.{module_name}", file_path, submodule_search_locations=[str(_PLUGINS_DIR)]
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "plugin"):
        return getattr(module, "plugin")
    if hasattr(module, "Plugin"):
        cls = getattr(module, "Plugin")
        return cls()
    return None


def discover_plugins() -> list[MediaPipePluginBase]:
    """
    Discover all plugins: first load built-in list, then scan plugins/ for
    any extra .py files (excluding __init__.py and base.py) that define 'plugin'.
    """
    loaded: list[MediaPipePluginBase] = []
    seen_ids: set[str] = set()

    for name in _BUILTIN_PLUGINS:
        p = _load_plugin_from_module(name)
        if p is not None and p.plugin_id and p.plugin_id not in seen_ids:
            loaded.append(p)
            seen_ids.add(p.plugin_id)

    for path in _PLUGINS_DIR.glob("*.py"):
        if path.name.startswith("_") or path.name == "base.py":
            continue
        module_name = path.stem
        if module_name in _BUILTIN_PLUGINS:
            continue
        p = _load_plugin_from_module(module_name)
        if p is not None and p.plugin_id and p.plugin_id not in seen_ids:
            loaded.append(p)
            seen_ids.add(p.plugin_id)

    return loaded
