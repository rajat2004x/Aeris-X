"""Research model metadata and health inference endpoints."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aeris.domain.schemas import HealthState, ModelMetadata, TelemetryInferenceRequest
from aeris.models.adapters import ModelRegistry
from aeris.services.inference import InferenceOrchestrator
from aeris.intelligence.orchestration import IntelligenceOrchestrator, ResearchHealthEstimator

router = APIRouter()


def get_registry() -> ModelRegistry:
    from aeris.api.app import get_model_registry
    return get_model_registry()


def get_inference_repository():
    from aeris.api.app import get_inference_repository as resolve_repository
    return resolve_repository()


def get_metrics():
    from aeris.api.app import get_inference_metrics
    return get_inference_metrics()


@router.get("/models", response_model=list[ModelMetadata])
def models(registry: ModelRegistry = Depends(get_registry)):
    return registry.list_metadata()


@router.post("/telemetry/infer", response_model=HealthState)
@router.post("/health-estimates", response_model=HealthState)
def health_estimate(
    payload: TelemetryInferenceRequest,
    http_request: Request,
    registry: ModelRegistry = Depends(get_registry),
    repository=Depends(get_inference_repository),
    metrics=Depends(get_metrics),
):
    return InferenceOrchestrator(registry, repository=repository, metrics=metrics).infer_health(
        payload, request_id=http_request.state.request_id
    )


@router.post("/intelligence/analyze")
def intelligence_analyze(
    payload: TelemetryInferenceRequest,
    http_request: Request,
    registry: ModelRegistry = Depends(get_registry),
    repository=Depends(get_inference_repository),
    metrics=Depends(get_metrics),
):
    """Run the Phase 16 evidence-preserving intelligence composition."""
    try:
        feature_names = payload.feature_order or [
            f"feature_{index}" for index in range(len(payload.window[0]))
        ]
        result = IntelligenceOrchestrator(
            health_estimator=ResearchHealthEstimator(
                registry,
                repository=repository,
                metrics=metrics,
            )
        ).analyze(
            asset_id=payload.asset_id,
            window=payload.window,
            feature_names=feature_names,
            operating_condition=payload.operating_conditions,
            request_id=http_request.state.request_id,
        )
        return asdict(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/metrics")
def metrics(metrics=Depends(get_metrics)):
    return metrics.snapshot()


@router.get("/inferences/{inference_id}")
def inference(inference_id: str, repository=Depends(get_inference_repository)):
    result = repository.get(inference_id)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Inference not found.")
    return result


@router.get("/assets/{asset_id}/inferences")
def asset_inferences(
    asset_id: str,
    limit: int = Query(default=100, ge=1, le=100),
    repository=Depends(get_inference_repository),
):
    return repository.list_by_asset(asset_id, limit)


@router.get("/assets/{asset_id}/health")
def asset_health(
    asset_id: str,
    limit: int = Query(default=100, ge=1, le=100),
    repository=Depends(get_inference_repository),
):
    return repository.list_health(asset_id, limit)
