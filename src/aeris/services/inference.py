"""Application orchestration for bounded telemetry inference."""

from datetime import datetime, timezone
from time import perf_counter
from typing import Any
import logging
import uuid

import numpy as np

from aeris.domain.schemas import HealthState, InferenceRecord, TelemetryInferenceRequest
from aeris.models.adapters import ModelNotFoundError, ModelRegistry, ModelUnavailableError
from aeris.preprocessing.telemetry import ModelInputPreprocessor, TelemetryValidationError
from aeris.monitoring.metrics import InferenceMetrics
from aeris.monitoring.events import emit_event, event
from aeris.monitoring.quality import evaluate_window

logger = logging.getLogger(__name__)


class InvalidModelOutputError(RuntimeError):
    """Raised when a model violates its declared output contract."""


class InferencePersistenceError(RuntimeError):
    """Raised when an inference result cannot be persisted."""


class InferenceOrchestrator:
    def __init__(
        self,
        model_registry: ModelRegistry,
        preprocessor: ModelInputPreprocessor | None = None,
        repository: Any = None,
        metrics: InferenceMetrics | None = None,
    ) -> None:
        self.model_registry = model_registry
        self.preprocessor = preprocessor or ModelInputPreprocessor()
        self.repository = repository
        self.metrics = metrics or InferenceMetrics()

    def _save_record(self, record: InferenceRecord) -> None:
        if self.repository is not None:
            self.repository.save(record)

    def infer_health(
        self,
        request: TelemetryInferenceRequest,
        timestamp: datetime | None = None,
        request_id: str | None = None,
    ) -> HealthState:
        inference_id = str(uuid.uuid4())
        started = perf_counter()
        created_at = timestamp or datetime.now(timezone.utc)
        record = InferenceRecord(
            inference_id=inference_id, asset_id=request.asset_id, request_id=request_id,
            model_id=request.model_id, task="HEALTH_ESTIMATION", status="RECEIVED",
            created_at=created_at, input_quality_status="unknown",
        )
        self._save_record(record)
        logger.info("inference_received", extra={"inference_id": inference_id, "request_id": request_id, "asset_id": request.asset_id})
        try:
            record.status = "VALIDATING"
            quality = evaluate_window(request.window, request.feature_order)
            record.input_quality_status = "valid" if quality.valid else "invalid"
            if not quality.valid:
                raise TelemetryValidationError("Telemetry quality validation failed.")
            resolver = getattr(self.model_registry, "resolve", None)
            model = (
                resolver(request.model_id, "health_estimation")
                if resolver is not None
                else self.model_registry.get(request.model_id)
            )
            metadata = model.metadata()
            record.model_id = metadata.model_id
            record.model_version = metadata.model_version
            record.status = "PREPROCESSING"
            prepared, preprocessing = self.preprocessor.prepare(request, metadata)
            record.status = "INFERENCING"
            logger.info("inference_started", extra={"inference_id": inference_id, "model_id": metadata.model_id})
            health = model.predict(prepared)
            if not isinstance(health, (int, float)) or not np.isfinite(float(health)):
                raise InvalidModelOutputError("Model produced a non-finite health estimate.")
            if not 0.0 <= float(health) <= 1.0:
                raise InvalidModelOutputError("Model produced a health estimate outside [0, 1].")
        except Exception as exc:
            duration_ms = (perf_counter() - started) * 1000
            record.status = "FAILED"
            record.completed_at = datetime.now(timezone.utc)
            record.duration_ms = duration_ms
            record.error_category = self._error_category(exc)
            record.error_message = str(exc)[:512]
            record.output_status = "not_available"
            self._save_record(record)
            self.metrics.record(record.status, duration_ms, record.error_category, record.model_id)
            emit_event(event(
                "inference_failed", inference_id=inference_id, request_id=request_id,
                asset_id=request.asset_id, model_id=record.model_id, model_version=record.model_version,
                task=record.task, status=record.status, duration_ms=duration_ms,
                error_category=record.error_category,
            ))
            logger.warning("inference_failed", extra={"inference_id": inference_id, "error_category": record.error_category})
            raise
        duration_ms = (perf_counter() - started) * 1000
        record.status = "SUCCEEDED"
        record.completed_at = datetime.now(timezone.utc)
        record.duration_ms = duration_ms
        record.output_status = "valid"
        record.health_score = float(health)
        self._save_record(record)
        self.metrics.record(record.status, duration_ms, model_id=record.model_id)
        emit_event(event(
            "inference_completed", inference_id=inference_id, request_id=request_id,
            asset_id=request.asset_id, model_id=metadata.model_id, model_version=metadata.model_version,
            task=record.task, status=record.status, duration_ms=duration_ms,
        ))
        logger.info("inference_completed", extra={"inference_id": inference_id, "duration_ms": duration_ms})
        result = HealthState(
            asset_id=request.asset_id,
            health_score=float(health),
            model_id=metadata.model_id,
            model_version=metadata.model_version,
            timestamp=timestamp or datetime.now(timezone.utc),
            evidence_ref=f"{request.source}:window:{preprocessing.window_length}x{preprocessing.feature_count}",
            inference_id=inference_id,
            request_id=request_id,
            inference_duration_ms=duration_ms,
        )
        return result

    @staticmethod
    def _error_category(error: Exception) -> str:
        if isinstance(error, TelemetryValidationError):
            return "PREPROCESSING_ERROR"
        if isinstance(error, ModelNotFoundError):
            return "MODEL_NOT_FOUND"
        if isinstance(error, ModelUnavailableError):
            return "MODEL_UNAVAILABLE"
        if isinstance(error, InvalidModelOutputError):
            return "OUTPUT_VALIDATION_ERROR"
        return "INFERENCE_ERROR"
