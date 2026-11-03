"""FastAPI application factory."""

import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from aeris.config.settings import get_settings
from aeris.core.errors import capability_error_handler, model_not_found_handler
from aeris.core.logging import configure_logging
from aeris.services.capabilities import CapabilityNotConfiguredError
from aeris.api.routes.system import router
from aeris.api.routes.models import router as model_router
from aeris.models.adapters import InvalidModelInputError, ModelNotFoundError, ModelRegistry, ModelUnavailableError
from aeris.preprocessing.telemetry import TelemetryValidationError
from aeris.services.inference import InferencePersistenceError, InvalidModelOutputError
from aeris.repositories.operational import InMemoryInferenceRepository
from aeris.monitoring.metrics import InferenceMetrics
from fastapi.responses import JSONResponse


_registry: ModelRegistry | None = None
_inference_repository = InMemoryInferenceRepository()
_metrics = InferenceMetrics()


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(Path(__file__).resolve().parents[3])
    return _registry


def get_inference_repository() -> InMemoryInferenceRepository:
    return _inference_repository


def get_inference_metrics() -> InferenceMetrics:
    return _metrics


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    application = FastAPI(title=settings.app_name, version=settings.version)
    cors_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    trusted_hosts = [host.strip() for host in settings.trusted_hosts.split(",") if host.strip()]
    if cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=[settings.request_id_header, "Content-Type"],
        )
    if trusted_hosts:
        application.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    application.include_router(router, prefix="/api/v1")
    application.include_router(model_router, prefix="/api/v1")
    application.add_exception_handler(CapabilityNotConfiguredError, capability_error_handler)
    application.add_exception_handler(ModelUnavailableError, capability_error_handler)
    application.add_exception_handler(ModelNotFoundError, model_not_found_handler)
    application.add_exception_handler(InvalidModelInputError, lambda _, exc: JSONResponse(
        status_code=422, content={"error": {"code": "invalid_model_input", "message": str(exc)}}
    ))
    application.add_exception_handler(TelemetryValidationError, lambda _, exc: JSONResponse(
        status_code=422, content={"error": {"code": "telemetry_validation_failed", "message": str(exc)}}
    ))
    application.add_exception_handler(InvalidModelOutputError, lambda _, exc: JSONResponse(
        status_code=502, content={"error": {"code": "invalid_model_output", "message": str(exc)}}
    ))
    application.add_exception_handler(InferencePersistenceError, lambda _, exc: JSONResponse(
        status_code=503, content={"error": {"code": "persistence_failed", "message": str(exc)}}
    ))

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get(settings.request_id_header, str(uuid.uuid4()))
        request.state.request_id = request_id
        content_length = request.headers.get("content-length")
        try:
            request_size = int(content_length) if content_length else 0
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "invalid_content_length", "message": "Invalid Content-Length header."}},
            )
        if request_size > settings.max_request_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"error": {"code": "request_too_large", "message": "Request body exceeds configured limit."}},
            )
        response = await call_next(request)
        response.headers[settings.request_id_header] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    return application


app = create_app()
