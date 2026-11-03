from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aeris.api.app import create_app
from aeris.domain.schemas import ModelMetadata, TelemetryInferenceRequest, TelemetryWindow
from aeris.models.adapters import ModelNotFoundError, ModelRegistry
from aeris.preprocessing.telemetry import ModelInputPreprocessor, TelemetryValidationError
from aeris.services.inference import InferenceOrchestrator, InvalidModelOutputError


def _window() -> list[list[float]]:
    return np.zeros((30, 15), dtype=float).tolist()


def test_telemetry_window_rejects_empty_and_inconsistent_samples():
    with pytest.raises(ValueError):
        TelemetryWindow(asset_id="a", samples=[], source="test")
    with pytest.raises(ValueError):
        TelemetryWindow(
            asset_id="a",
            source="test",
            samples=[
                {"channels": {"s1": 1.0, "s2": 2.0}},
                {"channels": {"s1": 1.0}},
            ],
        )


def test_inference_request_rejects_non_finite_and_inconsistent_rows():
    with pytest.raises(ValueError):
        TelemetryInferenceRequest(asset_id="a", window=[[float("nan")]])
    with pytest.raises(ValueError):
        TelemetryInferenceRequest(asset_id="a", window=[[1.0], [1.0, 2.0]])


def test_preprocessor_validates_shape_and_propagates_metadata():
    request = TelemetryInferenceRequest(asset_id="a", window=_window(), source="fd001")
    metadata = ModelMetadata(
        model_id="m", model_name="test", model_version="1",
        task="health_estimation", framework="test", status="available",
        input_shape=(30, 15), expected_features=15,
    )
    prepared, details = ModelInputPreprocessor().prepare(request, metadata)
    assert prepared.shape == (30, 15)
    assert details.normalization == "already_preprocessed"
    assert details.feature_order == [f"feature_{i}" for i in range(15)]


def test_preprocessor_rejects_wrong_model_shape():
    request = TelemetryInferenceRequest(asset_id="a", window=[[1.0]])
    metadata = ModelMetadata(
        model_id="m", model_name="test", model_version="1",
        task="health_estimation", framework="test", status="available",
        input_shape=(30, 15), expected_features=15,
    )
    with pytest.raises(TelemetryValidationError):
        ModelInputPreprocessor().prepare(request, metadata)


def test_unknown_model_is_not_reported_as_unavailable():
    with pytest.raises(ModelNotFoundError):
        ModelRegistry(Path(__file__).parents[1]).get("missing")


def test_orchestrator_persists_successful_health_result():
    class Predictor:
        def metadata(self):
            return ModelMetadata(
                model_id="m", model_name="test", model_version="1",
                task="health_estimation", framework="test", status="available",
                input_shape=(30, 15), expected_features=15,
            )

        def predict(self, payload):
            assert payload.shape == (30, 15)
            return 0.75

    class Registry:
        def get(self, model_id):
            assert model_id == "m"
            return Predictor()

    class Repository:
        def __init__(self):
            self.items = []

        def save(self, item):
            self.items.append(item)

    repository = Repository()
    request = TelemetryInferenceRequest(asset_id="a", window=_window(), model_id="m")
    result = InferenceOrchestrator(Registry(), repository=repository).infer_health(request)
    assert result.health_score == 0.75
    assert repository.items[0].model_id == "m"


def test_orchestrator_rejects_invalid_model_output():
    class Predictor:
        def metadata(self):
            return ModelMetadata(
                model_id="m", model_name="test", model_version="1",
                task="health_estimation", framework="test", status="available",
                input_shape=(30, 15), expected_features=15,
            )

        def predict(self, payload):
            return float("nan")

    class Registry:
        def get(self, model_id):
            return Predictor()

    request = TelemetryInferenceRequest(asset_id="a", window=_window(), model_id="m")
    with pytest.raises(InvalidModelOutputError):
        InferenceOrchestrator(Registry()).infer_health(request)


def test_real_transformer_telemetry_endpoint_smoke():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/telemetry/infer",
        headers={"X-Request-ID": "phase14c-test"},
        json={"asset_id": "engine-1", "source": "processed-fd001", "window": _window()},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["health_score"] <= 1.0
    assert body["model_id"] == "aeris-transformer-health"
    assert response.headers["X-Request-ID"] == "phase14c-test"


def test_api_returns_structured_model_not_found():
    response = TestClient(create_app()).post(
        "/api/v1/telemetry/infer",
        json={"asset_id": "engine-1", "model_id": "missing", "window": _window()},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
