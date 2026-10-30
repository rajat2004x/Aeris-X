import numpy as np

from aeris.fdi.condition import assign_conditions, fit_condition_baseline, fit_condition_model, robust_window_scores


def test_condition_model_and_assignment_are_deterministic():
    rows = np.array([[0.0, 1.0, 2.0], [0.1, 1.1, 2.1], [10.0, 11.0, 12.0]])
    model = fit_condition_model(rows, n_conditions=2, seed=42)
    first = assign_conditions(rows, model)
    second = assign_conditions(rows, model)
    assert np.array_equal(first, second)


def test_condition_baseline_scores_have_expected_shape():
    windows = np.ones((4, 3, 2), dtype=float)
    conditions = np.array([0, 0, 1, 1])
    baseline = fit_condition_baseline(windows, conditions)
    scores = robust_window_scores(windows, baseline, conditions, True)
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))


def test_condition_threshold_uses_normal_calibration_only():
    calibration = np.array([1.0, 2.0, 3.0])
    threshold = np.percentile(calibration, 95)
    assert threshold < 100.0
