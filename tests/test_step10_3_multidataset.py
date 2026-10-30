import numpy as np

from aeris.data.cmapss import SENSOR_COLUMNS, fit_sensor_scaler, load_cmapss, make_windows, split_engines
from aeris.faults.injection import inject_fault


def test_all_cmapss_adapters_load():
    for name in ("FD001", "FD002", "FD003", "FD004"):
        dataset = load_cmapss("data/raw", name)
        assert dataset.train.shape[1] == 26
        assert len(dataset.sensor_columns) == 21


def test_engine_split_has_no_overlap():
    dataset = load_cmapss("data/raw", "FD001")
    groups = split_engines(dataset.get_engine_ids(), 42)
    assert not (set(groups[0]) & set(groups[1]) or set(groups[0]) & set(groups[2]) or set(groups[1]) & set(groups[2]))


def test_windows_and_train_only_scaler():
    dataset = load_cmapss("data/raw", "FD002")
    train, _, final = split_engines(dataset.get_engine_ids(), 42)
    train_windows, train_owners = make_windows(dataset.train, train, SENSOR_COLUMNS, 30)
    _, final_owners = make_windows(dataset.train, final, SENSOR_COLUMNS, 30)
    scaler = fit_sensor_scaler(train_windows)
    assert train_windows.shape[1:] == (30, 21)
    assert not set(train_owners) & set(final_owners)
    assert np.all(np.isfinite(scaler["mean"]))


def test_fault_injection_is_deterministic_and_ordered():
    sequence = np.ones((30, 21), dtype=np.float32)
    first, metadata = inject_fault(sequence, 3, "NOISE", 0.5, 0.6, 1.0, seed=42)
    second, _ = inject_fault(sequence, 3, "NOISE", 0.5, 0.6, 1.0, seed=42)
    assert SENSOR_COLUMNS[3] == "sensor_4"
    assert np.array_equal(first, second)
    assert metadata["sensor_index"] == 3
