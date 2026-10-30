import numpy as np

from aeris.evaluation.anomaly import calibrate_threshold, fit_robust_baseline, persistence_flags, robust_sensor_scores


def test_robust_scores_and_threshold_use_normal_data():
    normal = np.zeros((4, 5, 2), dtype=float)
    normal[:, :, 0] = np.arange(5)
    baseline = fit_robust_baseline(normal)
    scores = robust_sensor_scores(normal, baseline)
    assert scores.shape == (4, 2)
    assert calibrate_threshold(scores.ravel()) >= 0


def test_persistence_requires_consecutive_flags():
    result = persistence_flags(np.array([True, False, True, True, True]), 3)
    assert result.tolist() == [False, False, False, False, True]
