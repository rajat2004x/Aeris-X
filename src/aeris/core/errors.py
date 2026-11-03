"""Safe API error contracts."""

from fastapi import Request
from fastapi.responses import JSONResponse


async def capability_error_handler(_: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": {"code": "capability_not_configured", "message": str(exc)}})


async def model_not_found_handler(_: Request, exc: LookupError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": {"code": "model_not_found", "message": str(exc)}})
