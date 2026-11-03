"""In-process metrics facade suitable for future exporter integrations."""

from threading import Lock

from aeris.domain.schemas import InferenceMetricSnapshot


class InferenceMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._count = 0
        self._success = 0
        self._failed = 0
        self._validation = 0
        self._unavailable = 0
        self._duration = 0.0
        self._model_usage: dict[str, int] = {}

    def record(
        self, status: str, duration_ms: float, error_category: str | None = None,
        model_id: str | None = None,
    ) -> None:
        with self._lock:
            self._count += 1
            self._duration += duration_ms
            if model_id:
                self._model_usage[model_id] = self._model_usage.get(model_id, 0) + 1
            if status == "SUCCEEDED":
                self._success += 1
            else:
                self._failed += 1
            if error_category in {"PREPROCESSING_ERROR", "VALIDATION_ERROR"}:
                self._validation += 1
            if error_category == "MODEL_UNAVAILABLE":
                self._unavailable += 1

    def snapshot(self) -> InferenceMetricSnapshot:
        with self._lock:
            return InferenceMetricSnapshot(
                inference_count=self._count,
                successful_inference_count=self._success,
                failed_inference_count=self._failed,
                validation_failure_count=self._validation,
                model_unavailable_count=self._unavailable,
                total_duration_ms=self._duration,
                average_duration_ms=self._duration / self._count if self._count else None,
                model_usage=dict(sorted(self._model_usage.items())),
            )
