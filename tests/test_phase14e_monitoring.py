import numpy as np
import pytest
from fastapi.testclient import TestClient

from aeris.api.app import create_app
from aeris.domain.schemas import ReferenceProfile
from aeris.monitoring.metrics import InferenceMetrics
from aeris.monitoring.quality import compare_to_reference, evaluate_window


def test_quality_statistics_are_bounded_and_deterministic():
    result = evaluate_window([[1.0, 2.0], [3.0, 4.0]], ["a", "b"])
    assert result.valid is True
    assert result.sequence_length == 2
    assert result.channel_count == 2
    assert result.feature_statistics["a"]["mean"] == 2.0
    invalid = evaluate_window([[1.0, float("nan")]])
    assert invalid.valid is False
    assert invalid.non_finite_values == ["window"]


def test_reference_comparison_does_not_claim_alerts_without_thresholds():
    profile = ReferenceProfile(
        source="test", feature_names=["a"], count=2, means={"a": 1.0},
        standard_deviations={"a": 0.0}, minima={"a": 1.0}, maxima={"a": 1.0},
    )
    result = compare_to_reference([[2.0], [2.0]], profile)
    assert result[0].comparable is True
    assert result[0].mean_delta == 1.0
    assert result[0].status == "COMPARED"
    assert "no alert threshold" in result[0].note


def test_metrics_track_success_and_failure_categories():
    metrics = InferenceMetrics()
    metrics.record("SUCCEEDED", 10.0)
    metrics.record("FAILED", 5.0, "MODEL_UNAVAILABLE")
    snapshot = metrics.snapshot()
    assert snapshot.inference_count == 2
    assert snapshot.successful_inference_count == 1
    assert snapshot.failed_inference_count == 1
    assert snapshot.model_unavailable_count == 1
    assert snapshot.average_duration_ms == 7.5


def test_runtime_metrics_endpoint_is_safe():
    response = TestClient(create_app()).get("/api/v1/metrics")
    assert response.status_code == 200
    assert "total_duration_ms" in response.json()
    assert "checkpoint" not in response.text.lower()


def test_telemetry_payload_limit_is_enforced():
    with pytest.raises(ValueError):
        from aeris.domain.schemas import TelemetryInferenceRequest
        TelemetryInferenceRequest(asset_id="a", window=np.zeros((1001, 1)).tolist())


def test_model_lifecycle_is_explicit_and_lazy():
    from pathlib import Path
    from aeris.models.adapters import ModelRegistry
    metadata = ModelRegistry(Path(__file__).parents[1]).list_metadata()
    assert all(item.lifecycle_state == "AVAILABLE" for item in metadata)
    assert all(item.loading_strategy == "lazy" for item in metadata)
