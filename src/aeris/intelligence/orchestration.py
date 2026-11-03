"""Composition service for Phase 16 aerospace intelligence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from aeris.intelligence.contracts import (
    AerospaceHealthState,
    HealthStateResult,
    IntelligenceResult,
)
from aeris.intelligence.detectors import (
    ConstantHealthEstimator,
    RankedDeviationIsolator,
    RobustWindowAnomalyDetector,
    RobustWindowSensorDetector,
)
from aeris.intelligence.diagnosis import diagnose
from aeris.intelligence.evidence import build_evidence
from aeris.intelligence.protocols import (
    FaultIsolator,
    HealthEstimator,
    SensorFaultDetector,
    SystemAnomalyDetector,
)
from aeris.domain.schemas import TelemetryInferenceRequest
from aeris.services.inference import InferenceOrchestrator


class IntelligenceOrchestrator:
    def __init__(
        self,
        sensor_detector: SensorFaultDetector | None = None,
        anomaly_detector: SystemAnomalyDetector | None = None,
        isolator: FaultIsolator | None = None,
        health_estimator: HealthEstimator | None = None,
    ) -> None:
        self.sensor_detector = sensor_detector or RobustWindowSensorDetector()
        self.anomaly_detector = anomaly_detector or RobustWindowAnomalyDetector()
        self.isolator = isolator or RankedDeviationIsolator()
        self.health_estimator = health_estimator or ConstantHealthEstimator()

    def analyze(
        self,
        asset_id: str,
        window: np.ndarray,
        feature_names: list[str] | None = None,
        operating_condition: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> IntelligenceResult:
        values = np.asarray(window, dtype=float)
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise ValueError("window must be a non-empty finite 2-D array")
        names = feature_names or [f"feature_{index}" for index in range(values.shape[1])]
        if len(names) != values.shape[1]:
            raise ValueError("feature_names must match window width")
        sensor = self.sensor_detector.detect(values, names)
        anomaly = self.anomaly_detector.detect(values, names)
        isolation = self.isolator.isolate(values, names)
        health = self.health_estimator.estimate(values)
        diagnosis = diagnose(sensor, anomaly, isolation, health)
        evidence = build_evidence(sensor, anomaly, isolation, health)
        timestamp = datetime.now(timezone.utc)
        provenance = {
            "phase": "16",
            "version": "phase16-intelligence-v1",
            "components": {
                "sensor_health": sensor.detector,
                "system_anomaly": anomaly.detector,
                "fault_isolation": isolation.isolator,
                "health_estimation": health.model_id,
                "diagnosis": diagnosis.provenance["rule_set"],
            },
            "operating_condition": operating_condition or {},
        }
        unified = AerospaceHealthState(
            asset_id=asset_id,
            health_score=health.health_score,
            degradation_state=health.degradation_state,
            anomaly_status=anomaly.status,
            sensor_health=sensor,
            subsystem_health={},
            failure_risk=None,
            uncertainty=health.uncertainty,
            operating_condition=operating_condition or {},
            evidence=evidence,
            provenance=provenance,
            timestamp=timestamp,
        )
        return IntelligenceResult(
            asset_id=asset_id,
            request_id=request_id,
            health_state=unified,
            anomaly=anomaly,
            sensor_health=sensor,
            fault_isolation=isolation,
            diagnosis=diagnosis,
            health_estimation=health,
            evidence=evidence,
            provenance=provenance,
        )


class ResearchHealthEstimator:
    """Adapter that keeps the validated research health path as the primary model."""

    def __init__(
        self,
        registry: Any,
        repository: Any = None,
        metrics: Any = None,
        model_id: str = "aeris-transformer-health",
        source: str = "phase16-intelligence",
    ) -> None:
        self.registry = registry
        self.repository = repository
        self.metrics = metrics
        self.model_id = model_id
        self.source = source

    def estimate(self, window: np.ndarray) -> HealthStateResult:
        values = np.asarray(window, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("window must be a finite 2-D array")
        request = TelemetryInferenceRequest(
            asset_id="phase16-inference",
            window=values.tolist(),
            source=self.source,
            model_id=self.model_id,
        )
        result = InferenceOrchestrator(
            self.registry,
            repository=self.repository,
            metrics=self.metrics,
        ).infer_health(request)
        if result.health_score >= 0.7:
            state = "HEALTHY"
        elif result.health_score >= 0.35:
            state = "DEGRADING"
        else:
            state = "NEAR_FAILURE"
        return HealthStateResult(
            health_score=result.health_score,
            degradation_state=state,
            model_id=result.model_id,
            uncertainty=result.uncertainty,
            provenance={
                "reference": "validated Step 13 Transformer health path",
                "model_version": result.model_version,
                "inference_id": result.inference_id,
                "evidence_ref": result.evidence_ref,
            },
        )
