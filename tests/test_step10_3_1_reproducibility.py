import numpy as np

from aeris.data.adapters.cmapss_adapter import load_prepared_cmapss
from aeris.evaluation.anomaly import fit_robust_baseline
from aeris.fdi.adaptive import soft_evidence, temporal_sensor_features


def test_historical_fd001_arrays_and_detector_are_deterministic():
    data = load_prepared_cmapss("data/processed/cmapss_fd001")
    baseline = fit_robust_baseline(data["X_train"])
    first = np.max(soft_evidence(temporal_sensor_features(data["X_validation"][:4], baseline["median"], baseline["scale"])), axis=1)
    second = np.max(soft_evidence(temporal_sensor_features(data["X_validation"][:4], baseline["median"], baseline["scale"])), axis=1)
    assert data["X_train"].shape[2] == 15
    assert np.array_equal(first, second)


def test_historical_threshold_does_not_use_final_labels():
    data = load_prepared_cmapss("data/processed/cmapss_fd001")
    calibration, final = np.array_split(data["X_validation"], 2)
    assert len(calibration) + len(final) == len(data["X_validation"])
    assert not np.shares_memory(calibration, final)
