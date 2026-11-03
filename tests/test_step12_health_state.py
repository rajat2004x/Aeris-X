import json
from pathlib import Path

import numpy as np
import pytest

from aeris.health.evaluation import regression_metrics, trajectory_metrics
from aeris.health.labels import health_state, rul_to_health
from aeris.health.pipeline import run_step12
from aeris.health.trajectory import split_by_engine_order


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/experiments/step12_health_state"


def test_health_labels_are_normalized_and_states_are_deterministic():
    values = rul_to_health(np.array([0.0, 5.0, 10.0]), 10.0)
    np.testing.assert_allclose(values, [0.0, 0.5, 1.0])
    assert health_state(values).tolist() == ["Critical", "Degrading", "Healthy"]


def test_metrics_and_engine_split():
    actual = np.array([1.0, 0.5, 0.0])
    predicted = np.array([0.9, 0.6, 0.1])
    metrics = regression_metrics(actual, predicted)
    assert metrics["mae"] == pytest.approx(0.1)
    assert trajectory_metrics(actual, predicted)["monotonicity_violations"] == 0
    dev, final, dev_ids, final_ids = split_by_engine_order(np.array([2, 2, 3, 4, 4]))
    assert dev_ids == [2]
    assert final_ids == [3, 4]
    assert dev.tolist() == [True, True, False, False, False]
    assert final.tolist() == [False, False, True, True, True]


def test_step12_artifacts_and_leakage_audit():
    result = run_step12(output_dir=OUTPUT)
    assert result["selected_model"] in {"mean_health", "linear_regression", "xgboost", "transformer"}
    assert result["final_metrics"]["mae"] >= 0
    assert (OUTPUT / "FINAL_STEP12_REPORT.md").exists()
    audit = json.loads((OUTPUT / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["final_holdout_used_for_selection"] is False
    assert audit["final_holdout_evaluated_after_freezing"] is True


def test_malformed_health_inputs_are_rejected():
    with pytest.raises(ValueError):
        rul_to_health(np.array([[1.0]]), 10.0)
    with pytest.raises(ValueError):
        split_by_engine_order(np.array([1, 1]))
