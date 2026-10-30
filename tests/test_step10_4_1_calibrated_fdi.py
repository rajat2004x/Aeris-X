import numpy as np

from aeris.fdi.fusion import calibration_metrics, evidence_features


def test_evidence_features_have_stable_shape():
    local = np.ones((4, 5))
    peer = np.ones((4, 5))
    temporal = np.ones((4, 5))
    values = evidence_features(local, peer, temporal, local.mean(axis=1), np.zeros(4))
    assert values.shape == (4, 12)


def test_calibration_metrics_are_finite():
    result = calibration_metrics(np.array([0.1, 0.9]), np.array([0, 1]))
    assert set(result) == {"brier", "ece"}
    assert all(np.isfinite(list(result.values())))
