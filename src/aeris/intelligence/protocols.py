"""Small interfaces for composing existing research capabilities."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from aeris.intelligence.contracts import (
    AnomalyResult,
    FaultIsolationResult,
    HealthStateResult,
    SensorHealthResult,
)


class SensorFaultDetector(Protocol):
    def detect(self, window: np.ndarray, feature_names: list[str]) -> SensorHealthResult:
        ...


class SystemAnomalyDetector(Protocol):
    def detect(self, window: np.ndarray, feature_names: list[str]) -> AnomalyResult:
        ...


class FaultIsolator(Protocol):
    def isolate(self, window: np.ndarray, feature_names: list[str]) -> FaultIsolationResult:
        ...


class HealthEstimator(Protocol):
    def estimate(self, window: np.ndarray) -> HealthStateResult:
        ...
