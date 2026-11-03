"""Dataset-agnostic telemetry validation and model input preparation."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aeris.domain.schemas import DataQualityResult, PreprocessingMetadata, TelemetryInferenceRequest


class TelemetryValidationError(ValueError):
    """Raised when telemetry cannot satisfy a model input contract."""


class TelemetryPreprocessor(ABC):
    @abstractmethod
    def prepare(
        self, request: TelemetryInferenceRequest, metadata: Any
    ) -> tuple[np.ndarray, PreprocessingMetadata]:
        raise NotImplementedError


class ModelInputPreprocessor(TelemetryPreprocessor):
    """Validate already-prepared telemetry without fitting or refitting anything."""

    def validate(self, request: TelemetryInferenceRequest, metadata: Any) -> DataQualityResult:
        rows = np.asarray(request.window, dtype=np.float32)
        expected = tuple(metadata.input_shape or ())
        if expected and rows.shape != expected:
            return DataQualityResult(valid=False, range_violations=[
                f"expected shape {expected}, received {rows.shape}"
            ])
        if not np.isfinite(rows).all():
            return DataQualityResult(valid=False, non_finite_values=["window"])
        return DataQualityResult(valid=True)

    def prepare(
        self, request: TelemetryInferenceRequest, metadata: Any
    ) -> tuple[np.ndarray, PreprocessingMetadata]:
        quality = self.validate(request, metadata)
        if not quality.valid:
            raise TelemetryValidationError(quality.model_dump_json())
        feature_count = rows_width = len(request.window[0])
        feature_order = request.feature_order or [f"feature_{i}" for i in range(rows_width)]
        return np.asarray(request.window, dtype=np.float32), PreprocessingMetadata(
            model_id=metadata.model_id,
            model_version=metadata.model_version,
            window_length=len(request.window),
            feature_count=feature_count,
            feature_order=feature_order,
            normalization="already_preprocessed",
        )
