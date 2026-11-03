import json
from pathlib import Path

import numpy as np
import pytest

from aeris.anomaly.events import event_metrics, event_segments, persistence_flags
from aeris.anomaly.evaluation import percentile_threshold
from aeris.anomaly.features import fit_robust_baseline, sensor_anomaly_scores, system_anomaly_scores
from aeris.anomaly.pipeline import run_step11


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/experiments/step11_anomaly_detection"


def test_features_threshold_and_events_are_deterministic():
    windows = np.zeros((4, 5, 3), dtype=float)
    windows[1, :, 0] = 3
    baseline = fit_robust_baseline(windows[[0, 2, 3]])
    sensor_scores = sensor_anomaly_scores(windows, baseline)
    assert system_anomaly_scores(sensor_scores, top_k=2).shape == (4,)
    assert percentile_threshold(np.array([1.0, 2.0, 3.0]), 95) > 2
    flags = persistence_flags(np.array([False, True, True, True, False]), 3)
    assert flags.tolist() == [False, False, False, True, False]
    assert event_segments(flags) == [(3, 3)]
    assert event_metrics(np.array([False, True, True, False]), flags)["actual_events"] == 1


def test_step11_artifacts_and_leakage_audit():
    result = run_step11(output_dir=OUTPUT)
    assert result["manifest"]["final_labels_used_for_fit_or_threshold"] is False
    assert result["results"]["dataset"] == "FD001"
    assert (OUTPUT / "final_results.csv").exists()
    assert (OUTPUT / "FINAL_STEP11_REPORT.md").exists()
    audit = json.loads((OUTPUT / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["final_holdout_used_for_model_selection"] is False
    assert audit["future_cycles_used"] is False


def test_invalid_feature_inputs_are_rejected():
    with pytest.raises(ValueError):
        sensor_anomaly_scores(np.ones((2, 3)), {"median": np.ones(3), "scale": np.ones(3)})
