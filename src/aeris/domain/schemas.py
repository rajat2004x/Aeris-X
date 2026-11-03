"""Validated API/domain contracts."""

from datetime import datetime
from enum import Enum
import math
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Asset(BaseModel):
    asset_id: str = Field(min_length=1)
    asset_type: str = Field(min_length=1)
    name: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TelemetryObservation(BaseModel):
    asset_id: str = Field(min_length=1)
    timestamp: datetime | None = None
    sequence: int | None = Field(default=None, ge=0)
    channel_id: str = Field(min_length=1)
    value: float
    quality: dict[str, Any] = Field(default_factory=dict)


class TelemetrySample(BaseModel):
    timestamp: datetime | None = None
    sequence: int | None = Field(default=None, ge=0)
    channels: dict[str, Any]
    quality: dict[str, Any] = Field(default_factory=dict)

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("telemetry sample must contain at least one channel")
        if len(value) != len(set(value)):
            raise ValueError("telemetry channel identifiers must be unique")
        normalized: dict[str, float] = {}
        for channel, reading in value.items():
            if not channel.strip():
                raise ValueError("telemetry channel identifiers must be non-empty")
            if not isinstance(reading, (int, float)) or not math.isfinite(float(reading)):
                raise ValueError("telemetry channel values must be finite numbers")
            normalized[channel] = float(reading)
        return normalized


class TelemetryWindow(BaseModel):
    asset_id: str = Field(min_length=1)
    samples: list[TelemetrySample]
    source: str = Field(min_length=1)
    operating_conditions: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, value: list[TelemetrySample]) -> list[TelemetrySample]:
        if not value:
            raise ValueError("telemetry window must not be empty")
        return value

    @model_validator(mode="after")
    def validate_dimensions(self) -> "TelemetryWindow":
        channels = set(self.samples[0].channels)
        if any(set(sample.channels) != channels for sample in self.samples[1:]):
            raise ValueError("telemetry samples must have consistent channel identifiers")
        return self


class DataQualityResult(BaseModel):
    valid: bool
    missing_channels: list[str] = Field(default_factory=list)
    non_finite_values: list[str] = Field(default_factory=list)
    duplicate_channels: list[str] = Field(default_factory=list)
    timestamp_issues: list[str] = Field(default_factory=list)
    range_violations: list[str] = Field(default_factory=list)
    quality_score: float | None = None
    sequence_length: int | None = None
    channel_count: int | None = None
    duplicate_timestamps: int = 0
    feature_statistics: dict[str, dict[str, float | int]] = Field(default_factory=dict)


class ReferenceProfile(BaseModel):
    source: str
    feature_names: list[str]
    count: int = Field(default=0, ge=0)
    means: dict[str, float] = Field(default_factory=dict)
    standard_deviations: dict[str, float] = Field(default_factory=dict)
    minima: dict[str, float] = Field(default_factory=dict)
    maxima: dict[str, float] = Field(default_factory=dict)
    missingness: dict[str, float] = Field(default_factory=dict)
    expected_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)


class DriftComparison(BaseModel):
    feature: str
    reference_mean: float | None = None
    observed_mean: float | None = None
    mean_delta: float | None = None
    statistic: float | None = None
    comparable: bool
    status: str
    note: str | None = None


class InferenceMetricSnapshot(BaseModel):
    inference_count: int
    successful_inference_count: int
    failed_inference_count: int
    validation_failure_count: int
    model_unavailable_count: int
    total_duration_ms: float
    average_duration_ms: float | None = None
    model_usage: dict[str, int] = Field(default_factory=dict)


class StructuredEvent(BaseModel):
    event: str
    timestamp: datetime
    request_id: str | None = None
    inference_id: str | None = None
    asset_id: str | None = None
    model_id: str | None = None
    model_version: str | None = None
    task: str | None = None
    status: str | None = None
    duration_ms: float | None = None
    error_category: str | None = None


class PreprocessingMetadata(BaseModel):
    model_id: str
    model_version: str
    window_length: int
    feature_count: int
    feature_order: list[str]
    normalization: str


class TelemetryInferenceRequest(BaseModel):
    asset_id: str = Field(min_length=1, max_length=128)
    window: list[list[float]]
    source: str = Field(default="api", min_length=1, max_length=128)
    model_id: str = Field(
        default="aeris-transformer-health",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    feature_order: list[str] | None = None
    operating_conditions: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)

    @field_validator("window")
    @classmethod
    def validate_window(cls, value: list[list[float]]) -> list[list[float]]:
        if not value or any(not row for row in value):
            raise ValueError("telemetry window must not be empty")
        if len(value) > 1000 or len(value[0]) > 256:
            raise ValueError("telemetry window exceeds the bounded request limit")
        width = len(value[0])
        if any(len(row) != width for row in value):
            raise ValueError("telemetry window rows must have consistent dimensions")
        for row in value:
            for reading in row:
                if not isinstance(reading, (int, float)) or not math.isfinite(float(reading)):
                    raise ValueError("telemetry values must be finite numbers")
        return value

    @model_validator(mode="after")
    def validate_feature_order(self) -> "TelemetryInferenceRequest":
        if self.feature_order is not None:
            if len(self.feature_order) != len(self.window[0]):
                raise ValueError("feature_order must match telemetry feature count")
            if len(set(self.feature_order)) != len(self.feature_order):
                raise ValueError("feature_order must contain unique channel identifiers")
        return self


class HealthState(BaseModel):
    asset_id: str = Field(min_length=1)
    health_score: float = Field(ge=0, le=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    timestamp: datetime
    uncertainty: float | None = Field(default=None, ge=0)
    evidence_ref: str | None = None
    inference_id: str | None = None
    request_id: str | None = None
    inference_duration_ms: float | None = Field(default=None, ge=0)


class InferenceRecord(BaseModel):
    inference_id: str
    asset_id: str
    request_id: str | None = None
    model_id: str | None = None
    model_version: str | None = None
    task: str = "HEALTH_ESTIMATION"
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    input_quality_status: str = "unknown"
    output_status: str = "not_available"
    error_category: str | None = None
    error_message: str | None = None
    health_score: float | None = Field(default=None, ge=0, le=1)


class InferenceHistoryItem(BaseModel):
    inference_id: str
    asset_id: str
    task: str
    model_id: str | None
    model_version: str | None
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    duration_ms: float | None = None
    health_score: float | None = None
    error_category: str | None = None


class HealthEstimateRequest(BaseModel):
    asset_id: str = Field(min_length=1)
    window: list[list[float]]
    model_id: str = "aeris-transformer-health"


class Anomaly(BaseModel):
    anomaly_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    timestamp: datetime
    severity: Severity
    score: float
    detector: str = Field(min_length=1)
    status: str = Field(min_length=1)
    evidence_ref: str | None = None


class FDIResult(BaseModel):
    fault_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    score: float
    confidence: float = Field(ge=0, le=1)
    model_version: str = Field(min_length=1)
    timestamp: datetime
    diagnostic_metadata: dict[str, Any] = Field(default_factory=dict)


class RiskAssessment(BaseModel):
    asset_id: str = Field(min_length=1)
    risk_level: Severity
    score: float = Field(ge=0, le=1)
    contributing_evidence: list[str] = Field(default_factory=list)
    model_version: str = Field(min_length=1)
    timestamp: datetime


class ModelMetadata(BaseModel):
    model_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    task: str = Field(min_length=1)
    framework: str = Field(min_length=1)
    status: str = Field(min_length=1)
    input_shape: tuple[int, int] | None = None
    expected_features: int | None = None
    dataset_reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_sequence_length: int | None = None
    checkpoint_reference: str | None = None
    loading_strategy: str = "lazy"
    lifecycle_state: str = "AVAILABLE"
    output_contract: str = "health_score in [0, 1], finite"
