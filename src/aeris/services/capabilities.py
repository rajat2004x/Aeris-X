"""Explicit capability status; no fabricated model predictions."""


class CapabilityNotConfiguredError(RuntimeError):
    """Raised when a production model adapter has not been wired."""


class TelemetryService:
    def ingest(self, payload: object) -> None:
        raise CapabilityNotConfiguredError("Telemetry ingestion is not configured.")


class HealthService:
    def __init__(self, model_registry=None, repository=None):
        self.model_registry = model_registry
        self.repository = repository

    def estimate(self, payload, timestamp=None):
        if self.model_registry is None:
            raise CapabilityNotConfiguredError("Health estimation is not configured.")
        from aeris.services.inference import InferenceOrchestrator
        return InferenceOrchestrator(self.model_registry, repository=self.repository).infer_health(
            payload, timestamp
        )


class AnomalyService:
    def detect(self, payload: object) -> None:
        raise CapabilityNotConfiguredError("Anomaly detection is not configured.")


class FDIService:
    def diagnose(self, payload: object) -> None:
        raise CapabilityNotConfiguredError("FDI inference is not configured.")


class RiskService:
    def assess(self, payload: object) -> None:
        raise CapabilityNotConfiguredError("Risk assessment is not configured.")
