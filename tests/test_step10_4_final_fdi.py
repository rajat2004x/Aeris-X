import numpy as np

from aeris.fdi.final import expected_calibration_error, fit_peer_models, peer_residual_scores, sensor_scores, soft_scores, temporal_features


def test_final_sensor_and_temporal_features():
    windows = np.ones((4, 5, 3), dtype=float)
    assert temporal_features(windows).shape == (4, 3, 5)
    assert sensor_scores(windows, np.zeros(3), np.ones(3)).shape == (4, 3)
    assert soft_scores(sensor_scores(windows, np.zeros(3), np.ones(3))).shape == (4,)


def test_peer_consistency_and_calibration_are_bounded():
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(8, 5, 3))
    models = fit_peer_models(windows)
    residual = peer_residual_scores(windows, models)
    assert residual.shape == (8, 3)
    assert 0.0 <= expected_calibration_error(np.array([0.2, 0.8]), np.array([0, 1])) <= 1.0
