"""System and dependency endpoints."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from aeris.config.settings import get_settings
from aeris.db.session import build_engine

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/version")
def version() -> dict[str, str]:
    settings = get_settings()
    return {"app": settings.app_name, "version": settings.version, "environment": settings.environment}


@router.get("/readiness")
def readiness() -> dict[str, object]:
    try:
        with build_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "application": "alive", "database": "unavailable"},
        )
    return {"status": "ready", "application": "alive", "database": "ready"}
