"""Framework-independent Phase 16 result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DiagnosisState(str, Enum):
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    SUSPECTED_DEGRADATION = "SUSPECTED_DEGRADATION"
    SUSPECTED_SENSOR_FAULT = "SUSPECTED_SENSOR_FAULT"
    SUSPECTED_SUBSYSTEM_FAULT = "SUSPECTED_SUBSYSTEM_FAULT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EvidenceItem:
    evidence_type: str
    source: str
    value: float | str | bool
    unit: str | None = None
    time_window: tuple[int, int] | None = None
    relevance: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorHealthResult:
    status: str
    overall_score: float
    sensor_scores: dict[str, float]
    flagged_sensors: tuple[str, ...]
    detector: str
    threshold: float
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnomalyResult:
    status: str
    score: float
    threshold: float
    detector: str
    event_recall_scope: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FaultIsolationResult:
    status: str
    ranked_sensors: tuple[str, ...]
    top_k_scores: dict[str, float]
    isolator: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthStateResult:
    health_score: float
    degradation_state: str
    model_id: str
    uncertainty: float | None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosisResult:
    state: DiagnosisState
    suspected_subsystem: str | None
    severity: str
    supporting_evidence: tuple[str, ...]
    conflicting_evidence: tuple[str, ...]
    confidence: float | None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AerospaceHealthState:
    asset_id: str
    health_score: float
    degradation_state: str
    anomaly_status: str
    sensor_health: SensorHealthResult
    subsystem_health: dict[str, float]
    failure_risk: float | None
    uncertainty: float | None
    operating_condition: dict[str, Any]
    evidence: tuple[EvidenceItem, ...]
    provenance: dict[str, Any]
    timestamp: datetime


@dataclass(frozen=True)
class IntelligenceResult:
    asset_id: str
    request_id: str | None
    health_state: AerospaceHealthState
    anomaly: AnomalyResult
    sensor_health: SensorHealthResult
    fault_isolation: FaultIsolationResult
    diagnosis: DiagnosisResult
    health_estimation: HealthStateResult
    evidence: tuple[EvidenceItem, ...]
    provenance: dict[str, Any]
