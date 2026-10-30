import numpy as np

from aeris.fdi.optimization import persistence_flags, sensor_feature_tensor, sensor_set_metrics


def test_sensor_feature_tensor_shape_and_finiteness():
    windows = np.arange(2 * 6 * 3, dtype=float).reshape(2, 6, 3)
    features = sensor_feature_tensor(windows, np.zeros(3), np.ones(3))
    assert features.shape == (2, 3, 12)
    assert np.isfinite(features).all()


def test_persistence_requires_n_of_m():
    scores = np.array([0.0, 2.0, 2.0, 0.0, 2.0])
    assert persistence_flags(scores, 1.0, window=3, required=2).tolist() == [False, False, True, True, True]


def test_multi_sensor_set_metrics():
    result = sensor_set_metrics([{1, 2}, {1}], [{1, 2}, {2}])
    assert result["set_precision"] == 0.5
    assert result["set_recall"] == 0.5
