import numpy as np
import pytest

from aeris.evaluation.uncertainty import interval_coverage, relative_uncertainty, summarize_mc_predictions


def test_mc_summary_shapes_and_values():
    values = np.array([[1.0, 3.0], [2.0, 2.0]])
    summary = summarize_mc_predictions(values)
    assert summary["mean_prediction"].tolist() == [2.0, 2.0]
    assert summary["predictive_std"][1] == pytest.approx(0.0)
    assert summary["interval_width"].shape == (2,)


def test_interval_coverage():
    assert interval_coverage(np.array([1.0, 3.0, 8.0]), np.array([0.0, 2.0, 9.0]), np.array([2.0, 4.0, 10.0])) == pytest.approx(2 / 3)


def test_relative_uncertainty_handles_zero_prediction():
    result = relative_uncertainty(np.array([0.0, -2.0]), np.array([1.0, 1.0]))
    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(1e8)
    assert result[1] == pytest.approx(0.5)


def test_mc_summary_rejects_invalid_shape():
    with pytest.raises(ValueError):
        summarize_mc_predictions(np.ones(3))
