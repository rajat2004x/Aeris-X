"""Deterministic evidence/rule-based diagnosis."""

from __future__ import annotations

from aeris.intelligence.contracts import (
    AnomalyResult,
    DiagnosisResult,
    DiagnosisState,
    FaultIsolationResult,
    HealthStateResult,
    SensorHealthResult,
)


def diagnose(
    sensor: SensorHealthResult,
    anomaly: AnomalyResult,
    isolation: FaultIsolationResult,
    health: HealthStateResult,
) -> DiagnosisResult:
    supporting: list[str] = []
    conflicting: list[str] = []
    if sensor.flagged_sensors:
        supporting.append("sensor_deviation")
    if anomaly.status == "ANOMALOUS":
        supporting.append("system_anomaly")
    if health.degradation_state != "HEALTHY":
        supporting.append("health_decline")
    if sensor.flagged_sensors and anomaly.status == "NORMAL":
        conflicting.append("system_anomaly_not_confirmed")

    if sensor.flagged_sensors and anomaly.status == "NORMAL":
        state = DiagnosisState.SUSPECTED_SENSOR_FAULT
        severity = "medium"
    elif anomaly.status == "ANOMALOUS" and health.degradation_state != "HEALTHY":
        state = DiagnosisState.SUSPECTED_DEGRADATION
        severity = "high" if health.degradation_state == "NEAR_FAILURE" else "medium"
    elif anomaly.status == "ANOMALOUS":
        state = DiagnosisState.MONITOR
        severity = "medium"
    elif health.degradation_state != "HEALTHY":
        state = DiagnosisState.SUSPECTED_DEGRADATION
        severity = "medium"
    else:
        state = DiagnosisState.NORMAL
        severity = "low"

    return DiagnosisResult(
        state=state,
        suspected_subsystem=None,
        severity=severity,
        supporting_evidence=tuple(supporting),
        conflicting_evidence=tuple(conflicting),
        confidence=None,
        provenance={
            "rule_set": "phase16-deterministic-v1",
            "fault_isolation_status": isolation.status,
            "confidence": "not empirically calibrated",
        },
    )
