import numpy as np

from aeris.fdi.adaptive import ewma, soft_evidence, temporal_sensor_features


def test_temporal_features_are_causal_and_finite():
    windows = np.arange(2 * 8 * 3, dtype=float).reshape(2, 8, 3)
    features = temporal_sensor_features(windows, np.zeros(3), np.ones(3))
    assert features.shape == (2, 3, 12)
    assert np.isfinite(features).all()


def test_soft_evidence_is_bounded():
    features = np.ones((4, 3, 12), dtype=float) * 100
    scores = soft_evidence(features)
    assert scores.shape == (4, 3)
    assert np.all((scores >= 0) & (scores <= 1))


def test_ewma_is_deterministic():
    result = ewma(np.array([0.0, 1.0, 1.0]), alpha=0.5)
    assert np.allclose(result, [0.0, 0.5, 0.75])
