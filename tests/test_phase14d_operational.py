from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from aeris.api.app import create_app
from aeris.db.models import Base
from aeris.domain.schemas import Asset, InferenceRecord, ModelMetadata, TelemetryInferenceRequest
from aeris.models.adapters import ModelRegistry
from aeris.repositories.sqlalchemy import AssetRepository, InferenceRepository, ModelRepository
from aeris.services.inference import InferenceOrchestrator


def _window():
    return np.zeros((30, 15), dtype=float).tolist()


def _record(inference_id="i-1", asset_id="engine-1"):
    return InferenceRecord(
        inference_id=inference_id,
        asset_id=asset_id,
        request_id="request-1",
        model_id="model-1",
        model_version="step8-fd001",
        status="SUCCEEDED",
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_ms=1.5,
        input_quality_status="valid",
        output_status="valid",
        health_score=0.8,
    )


def test_inference_repository_round_trip_and_histories():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        AssetRepository(session).save(Asset(asset_id="engine-1", asset_type="test"))
        ModelRepository(session).save(ModelMetadata(
            model_id="model-1", model_name="test", model_version="step8-fd001",
            task="health_estimation", framework="test", status="available",
        ))
        repository = InferenceRepository(session)
        repository.save(_record())
        session.commit()
        assert repository.get("i-1").health_score == 0.8
        assert len(repository.list_by_asset("engine-1", 10)) == 1
        assert len(repository.list_health("engine-1", 10)) == 1


def test_orchestrator_records_request_identity_and_duration():
    class Repository:
        def __init__(self):
            self.items = {}

        def save(self, item):
            self.items[item.inference_id] = item

    repository = Repository()
    request = TelemetryInferenceRequest(asset_id="engine-1", window=_window())
    result = InferenceOrchestrator(
        ModelRegistry(Path(__file__).parents[1]), repository=repository
    ).infer_health(request, request_id="request-1")
    record = repository.items[result.inference_id]
    assert record.request_id == "request-1"
    assert record.status == "SUCCEEDED"
    assert record.duration_ms is not None
    assert result.inference_id == record.inference_id


def test_api_e2e_inference_and_history():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/telemetry/infer",
        headers={"X-Request-ID": "request-e2e"},
        json={"asset_id": "engine-e2e", "window": _window()},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["inference_id"]
    assert body["request_id"] == "request-e2e"
    inference = client.get(f"/api/v1/inferences/{body['inference_id']}")
    assert inference.status_code == 200
    assert inference.json()["status"] == "SUCCEEDED"
    history = client.get("/api/v1/assets/engine-e2e/health")
    assert history.status_code == 200
    assert history.json()[0]["inference_id"] == body["inference_id"]


def test_api_history_has_safe_not_found_and_limit_validation():
    client = TestClient(create_app())
    missing = client.get("/api/v1/inferences/missing")
    assert missing.status_code == 404
    assert "traceback" not in missing.text.lower()
    invalid_limit = client.get("/api/v1/assets/a/inferences?limit=0")
    assert invalid_limit.status_code == 422
