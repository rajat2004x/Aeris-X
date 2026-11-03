import os

from fastapi.testclient import TestClient

from aeris.api.app import create_app
from aeris.config.settings import Settings
from aeris.domain.schemas import Asset, HealthState, Severity
from aeris.models.interfaces import ModelPredictor
from aeris.repositories.base import Repository
from aeris.services.capabilities import CapabilityNotConfiguredError, HealthService


def test_system_endpoints_and_openapi():
    client = TestClient(create_app())
    assert client.get("/api/v1/health").json() == {"status": "alive"}
    assert client.get("/api/v1/version").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    response = client.get("/api/v1/readiness")
    assert response.status_code in {200, 503}
    assert response.json()["application"] == "alive"


def test_request_id_is_returned():
    response = TestClient(create_app()).get("/api/v1/health", headers={"X-Request-ID": "test-request"})
    assert response.headers["X-Request-ID"] == "test-request"


def test_domain_contracts_validate():
    asset = Asset(asset_id="engine-1", asset_type="turbofan")
    assert asset.asset_id == "engine-1"
    state = HealthState(
        asset_id="engine-1",
        health_score=0.8,
        model_id="model",
        model_version="1",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert state.health_score == 0.8
    assert Severity.high.value == "high"


def test_configuration_is_environment_driven(monkeypatch):
    monkeypatch.setenv("AERIS_APP_NAME", "Test App")
    assert Settings().app_name == "Test App"
    assert Settings().database_url.endswith("localhost:5432/aeris")


def test_unconfigured_capability_is_explicit():
    try:
        HealthService().estimate({})
    except CapabilityNotConfiguredError as error:
        assert "not configured" in str(error)
    else:
        raise AssertionError("Expected explicit capability error")


def test_interfaces_are_separate_from_http():
    assert issubclass(ModelPredictor, object)
    assert issubclass(Repository, object)
    assert "fastapi" not in Asset.__module__
