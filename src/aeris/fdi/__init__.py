"""Reusable sensor-level fault detection and isolation utilities."""

from .scoring import (
    build_feature_matrix,
    cross_sensor_scores,
    derivative_sensor_scores,
    temporal_persistence,
)

__all__ = [
    "build_feature_matrix",
    "cross_sensor_scores",
    "derivative_sensor_scores",
    "temporal_persistence",
]
