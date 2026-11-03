"""Deterministic, bounded detectors used by the Phase 16 composition layer."""

from __future__ import annotations

import numpy as np

from aeris.intelligence.contracts import (
    AnomalyResult,
    FaultIsolationResult,
    HealthStateResult,
    SensorHealthResult,
)


def _validate(window: np.ndarray, feature_names: list[str]) -> np.ndarray:
    values = np.asarray(window, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("window must be a non-empty finite 2-D array")
    if len(feature_names) != values.shape[1] or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must match the unique sensor dimension")
    return values


def _scores(window: np.ndarray) -> np.ndarray:
    baseline = np.median(window, axis=0)
    mad = np.median(np.abs(window - baseline), axis=0)
    scale = np.maximum(1.4826 * mad, np.std(window, axis=0) * 0.1)
    scale = np.maximum(scale, 1e-6)
    return np.abs(window[-1] - baseline) / scale


class RobustWindowSensorDetector:
    """Uses within-window robust deviation; no external labels or calibration data."""

    threshold = 3.0
    detector_name = "phase16_robust_window_mad"

    def detect(self, window: np.ndarray, feature_names: list[str]) -> SensorHealthResult:
        values = _validate(window, feature_names)
        scores = _scores(values)
        normalized = {name: float(1.0 / (1.0 + score)) for name, score in zip(feature_names, scores)}
        flagged = tuple(name for name, score in zip(feature_names, scores) if score >= self.threshold)
        return SensorHealthResult(
            status="FAULT_SUSPECTED" if flagged else "HEALTHY",
            overall_score=float(np.mean(list(normalized.values()))),
            sensor_scores=normalized,
            flagged_sensors=flagged,
            detector=self.detector_name,
            threshold=self.threshold,
            provenance={"baseline": "same telemetry window median/MAD", "calibration": "none"},
        )


class RobustWindowAnomalyDetector:
    threshold = 3.0
    detector_name = "phase16_robust_window_system"

    def detect(self, window: np.ndarray, feature_names: list[str]) -> AnomalyResult:
        values = _validate(window, feature_names)
        sensor_scores = _scores(values)
        score = float(np.mean(np.sort(sensor_scores)[-min(3, len(sensor_scores)):]))
        return AnomalyResult(
            status="ANOMALOUS" if score >= self.threshold else "NORMAL",
            score=score,
            threshold=self.threshold,
            detector=self.detector_name,
            event_recall_scope="single-window heuristic; not a validated event metric",
            provenance={"aggregation": "mean top-3 terminal sensor deviations"},
        )


class RankedDeviationIsolator:
    isolator_name = "phase16_terminal_deviation_rank"

    def isolate(self, window: np.ndarray, feature_names: list[str]) -> FaultIsolationResult:
        values = _validate(window, feature_names)
        scores = _scores(values)
        order = np.argsort(scores)[::-1]
        ranked = tuple(feature_names[index] for index in order)
        top = {feature_names[index]: float(scores[index]) for index in order[: min(5, len(order))]}
        return FaultIsolationResult(
            status="ISOLATED" if scores[order[0]] >= 3.0 else "NO_FAULT_ISOLATED",
            ranked_sensors=ranked,
            top_k_scores=top,
            isolator=self.isolator_name,
            provenance={"ranking": "terminal robust deviation", "threshold": 3.0},
        )


class ConstantHealthEstimator:
    """Explicit fallback used when a research health model is not supplied."""

    model_id = "phase16_constant_health_fallback"

    def estimate(self, window: np.ndarray) -> HealthStateResult:
        values = _validate(window, [str(index) for index in range(np.asarray(window).shape[1])])
        score = float(np.clip(1.0 - np.mean(np.abs(_scores(values))) / 10.0, 0.0, 1.0))
        state = "HEALTHY" if score >= 0.7 else "DEGRADING" if score >= 0.35 else "NEAR_FAILURE"
        return HealthStateResult(
            health_score=score,
            degradation_state=state,
            model_id=self.model_id,
            uncertainty=None,
            provenance={"status": "fallback", "confidence": "not calibrated"},
        )
