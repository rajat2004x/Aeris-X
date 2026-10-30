import numpy as np
import pytest

from aeris.evaluation.benchmark import validate_ml_ready_dataset
from aeris.evaluation.metrics import mean_absolute_error, phm_score, root_mean_squared_error
from aeris.models.classical import MeanRULRegressor, flatten_temporal_windows


def test_dataset_validation_accepts_aligned_finite_arrays():
    X = np.ones((3, 2, 4))
    validate_ml_ready_dataset(X, np.ones(3), X.copy(), np.ones(3))


def test_dataset_validation_rejects_misaligned_targets():
    X = np.ones((3, 2, 4))
    with pytest.raises(ValueError, match="aligned"):
        validate_ml_ready_dataset(X, np.ones(2), X, np.ones(3))


def test_flattening_preserves_values_without_mutating_original():
    X = np.arange(24, dtype=float).reshape(2, 3, 4)
    flattened = flatten_temporal_windows(X)
    flattened[0, 0] = -1
    assert flattened.shape == (2, 12)
    assert X[0, 0, 0] == 0


def test_mean_baseline_fit_predict():
    model = MeanRULRegressor().fit(np.zeros((3, 2)), np.array([1.0, 3.0, 5.0]))
    np.testing.assert_allclose(model.predict(np.zeros((2, 2))), [3.0, 3.0])


def test_regression_metrics():
    actual = np.array([1.0, 2.0, 4.0])
    predicted = np.array([2.0, 2.0, 1.0])
    assert mean_absolute_error(actual, predicted) == pytest.approx(4 / 3)
    assert root_mean_squared_error(actual, predicted) == pytest.approx(np.sqrt(10 / 3))


def test_phm_score_is_zero_for_perfect_predictions_and_asymmetric():
    actual = np.array([10.0])
    assert phm_score(actual, actual) == pytest.approx(0.0)
    assert phm_score(actual, np.array([9.0])) < phm_score(actual, np.array([11.0]))


def test_model_fit_predict_interfaces():
    model = MeanRULRegressor().fit(np.ones((2, 3)), np.array([2.0, 4.0]))
    predictions = model.predict(np.ones((4, 3)))
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
