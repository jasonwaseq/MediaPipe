"""
Shared data models and the unified results schema for all plugins.
"""

from typing import Any

# Type alias for the unified results dict returned by every plugin
UnifiedResults = dict[str, Any]


def unified_results_schema(
    pipeline: str,
    timestamp_s: float,
    detections: list[Any] | None = None,
    landmarks: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> UnifiedResults:
    """Build a results dict that conforms to the unified schema."""
    return {
        "pipeline": pipeline,
        "timestamp_s": timestamp_s,
        "detections": detections if detections is not None else [],
        "landmarks": landmarks if landmarks is not None else [],
        "metadata": metadata if metadata is not None else {},
    }
