from fastapi.testclient import TestClient

from aeris.api.app import create_app
from aeris.config.settings import Settings, get_settings


def test_runtime_limits_and_security_headers(monkeypatch):
    monkeypatch.setenv("AERIS_MAX_REQUEST_BODY_BYTES", "1024")
    get_settings.cache_clear()
    try:
        response = TestClient(create_app()).get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Cache-Control"] == "no-store"
        assert Settings().max_request_body_bytes == 1024
    finally:
        get_settings.cache_clear()


def test_oversized_request_is_rejected(monkeypatch):
    monkeypatch.setenv("AERIS_MAX_REQUEST_BODY_BYTES", "1024")
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/telemetry/infer",
            headers={"Content-Length": "1025"},
            content=b'{"asset_id":"a"}',
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"
    finally:
        get_settings.cache_clear()
