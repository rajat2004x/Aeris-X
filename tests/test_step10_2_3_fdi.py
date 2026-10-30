import numpy as np

from aeris.fdi.isolation import expected_calibration_error, sensor_feature_tensor, standardized_distance


def test_sensor_feature_tensor_preserves_sensor_axis():
    values = np.arange(2 * 8 * 3, dtype=float).reshape(2, 8, 3)
    features = sensor_feature_tensor(values, np.zeros(3), np.ones(3), np.ones((2, 3)))
    assert features.shape == (2, 3, 20)
    assert np.isfinite(features).all()


def test_standardized_distance_selects_nearest_class():
    values = np.array([[0.0, 0.0], [5.0, 5.0]])
    distance, index = standardized_distance(values, {"a": np.zeros(2), "b": np.ones(2) * 5})
    assert index.tolist() == [0, 1]
    assert np.allclose(distance, 0)


def test_expected_calibration_error_is_bounded():
    result = expected_calibration_error(np.array([0.1, 0.9]), np.array([0, 1]))
    assert 0 <= result <= 1
