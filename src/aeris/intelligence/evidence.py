"""Structured evidence generation without natural-language or LLM interpretation."""

from __future__ import annotations

from aeris.intelligence.contracts import (
    AnomalyResult,
    DiagnosisResult,
    EvidenceItem,
    FaultIsolationResult,
    HealthStateResult,
    SensorHealthResult,
)


def build_evidence(
    sensor: SensorHealthResult,
    anomaly: AnomalyResult,
    isolation: FaultIsolationResult,
    health: HealthStateResult,
) -> tuple[EvidenceItem, ...]:
    evidence = [
        EvidenceItem(
            evidence_type="sensor_health",
            source=sensor.detector,
            value=sensor.overall_score,
            unit="score",
            relevance=1.0,
            provenance=sensor.provenance,
        ),
        EvidenceItem(
            evidence_type="system_anomaly_score",
            source=anomaly.detector,
            value=anomaly.score,
            unit="score",
            relevance=1.0,
            provenance=anomaly.provenance,
        ),
        EvidenceItem(
            evidence_type="health_prediction",
            source=health.model_id,
            value=health.health_score,
            unit="normalized_health",
            relevance=1.0,
            provenance=health.provenance,
        ),
    ]
    for sensor_name, score in isolation.top_k_scores.items():
        evidence.append(EvidenceItem(
            evidence_type="sensor_deviation",
            source=isolation.isolator,
            value=score,
            unit="robust_deviation",
            relevance=0.8,
            provenance={"sensor": sensor_name, **isolation.provenance},
        ))
    return tuple(evidence)
