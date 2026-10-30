import numpy as np
import pytest

from aeris.faults.injection import inject_fault
from aeris.data.adapters.cmapss_adapter import load_prepared_cmapss
from aeris.fdi.scoring import build_feature_matrix, cross_sensor_scores, derivative_sensor_scores, temporal_persistence


def test_open_set_faults_modify_data():
    source = np.ones((10, 3), dtype=np.float32)
    for fault_type in ("SPIKE", "SATURATION", "GAIN_ERROR"):
        changed, _ = inject_fault(source, 1, fault_type, 0.5, 0.5, 1.0, seed=42)
        assert not np.array_equal(source, changed)


def test_cmapss_adapter_reads_existing_arrays():
    data = load_prepared_cmapss("data/processed/cmapss_fd001")
    assert data["X_train"].shape[1:] == (30, 15)
    assert np.isfinite(data["X_validation"]).all()


def test_sensor_aware_features_are_deterministic_and_finite():
    windows = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
    derivatives = derivative_sensor_scores(windows)
    cross_sensor = cross_sensor_scores(windows, np.ones(3))
    persistence_input = np.zeros((2, 5, 3), dtype=float)
    persistence_input[0, 1:4, 2] = 4
    persistence = temporal_persistence(persistence_input, threshold=3, consecutive=2)
    features = build_feature_matrix(windows, derivatives, derivatives, cross_sensor)
    assert derivatives.shape == (2, 3)
    assert persistence[0, 2] == 3
    assert features.shape == (2, 18)
    assert np.isfinite(features).all()


def test_extended_fault_families_change_signal():
    source = np.tile(np.linspace(0, 1, 20, dtype=np.float32)[:, None], (1, 3))
    for fault_type in ("OFFSET_ERROR", "DEADBAND", "QUANTIZATION", "MISSING_BLOCK", "BURST_NOISE", "COMPOSITE"):
        changed, _ = inject_fault(source, 1, fault_type, 0.25, 0.5, 1.0, seed=42)
        assert not np.array_equal(source, changed, equal_nan=True)
