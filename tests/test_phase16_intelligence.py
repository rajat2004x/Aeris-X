import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aeris.api.app import create_app
from aeris.intelligence.contracts import DiagnosisState
from aeris.intelligence.detectors import (
    RankedDeviationIsolator,
    RobustWindowAnomalyDetector,
    RobustWindowSensorDetector,
)
from aeris.intelligence.orchestration import IntelligenceOrchestrator
from aeris.intelligence.scenarios import build_scenario_matrix


def _window() -> np.ndarray:
    values = np.zeros((30, 4), dtype=float)
    values[:, 0] = np.linspace(0.0, 1.0, 30)
    values[-1, 2] = 10.0
    return values


def test_contracts_and_orchestration_preserve_provenance():
    result = IntelligenceOrchestrator().analyze(
        "asset-1", _window(), ["pressure", "temperature", "vibration", "flow"]
    )
    assert result.asset_id == "asset-1"
    assert result.health_state.provenance["phase"] == "16"
    assert result.health_state.evidence
    assert result.diagnosis.confidence is None


def test_detectors_are_deterministic_and_rank_sensor():
    window = _window()
    sensor = RobustWindowSensorDetector().detect(window, ["a", "b", "c", "d"])
    anomaly = RobustWindowAnomalyDetector().detect(window, ["a", "b", "c", "d"])
    isolation = RankedDeviationIsolator().isolate(window, ["a", "b", "c", "d"])
    assert sensor == RobustWindowSensorDetector().detect(window, ["a", "b", "c", "d"])
    assert anomaly.score >= anomaly.threshold
    assert isolation.ranked_sensors[0] == "c"


def test_malformed_input_and_duplicate_features_are_rejected():
    with pytest.raises(ValueError):
        IntelligenceOrchestrator().analyze("asset", np.ones((3, 2)), ["a"])
    with pytest.raises(ValueError):
        RobustWindowSensorDetector().detect(np.ones((3, 2)), ["a", "a"])
    with pytest.raises(ValueError):
        IntelligenceOrchestrator().analyze("asset", np.array([[np.nan]]))


def test_scenario_matrix_is_held_out_and_reproducible():
    first = build_scenario_matrix(seed=9)
    second = build_scenario_matrix(seed=9)
    assert first == second
    assert all(item.split == "heldout_evaluation" for item in first)
    assert all(item.fault_type != "NORMAL" or item.affected_sensor is None for item in first)


def test_intelligence_api_validation_and_response():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/intelligence/analyze",
        headers={"X-Request-ID": "phase16-test"},
        json={
            "asset_id": "asset-1",
            "source": "phase16-test",
            "window": np.zeros((30, 15), dtype=float).tolist(),
            "feature_order": [f"sensor_{i}" for i in range(15)],
            "operating_conditions": {"throttle": 0.5},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["health_state"]["asset_id"] == "asset-1"
    assert body["provenance"]["phase"] == "16"
    assert body["request_id"] == "phase16-test"
    assert body["health_estimation"]["uncertainty"] is None

    invalid = client.post(
        "/api/v1/intelligence/analyze",
        json={"asset_id": "asset-1", "window": [[1.0], [2.0, 3.0]]},
    )
    assert invalid.status_code == 422


def test_phase16_artifact_runner_creates_isolated_outputs(tmp_path):
    from scripts.run_phase16_foundation import run

    result = run(tmp_path)
    assert result["scenario_count"] == 6
    for directory in result["output_directories"]:
        assert (tmp_path / directory.split("\\")[-1] / "leakage_audit.json").exists()
    audit = json.loads((tmp_path / "phase16_intelligence" / "leakage_audit.json").read_text())
    assert audit["final_labels_used_for_selection"] is False
