import numpy as np

from aeris.fdi.domain_invariant import (
    fit_residual_models,
    residual_features,
    residual_windows,
)


def test_condition_residual_model_preserves_shapes():
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(8, 6, 3))
    settings = rng.normal(size=(8, 3))
    model = fit_residual_models(windows, settings)
    residual = residual_windows(windows, settings, model)
    evidence, features = residual_features(residual, model["scale"])
    assert residual.shape == windows.shape
    assert evidence.shape == (8, 3)
    assert features.shape == (8, 3, 7)
    assert np.isfinite(evidence).all()


def test_fault_residual_changes_local_evidence():
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(10, 6, 3))
    settings = np.zeros((10, 3))
    model = fit_residual_models(windows, settings)
    clean = residual_windows(windows, settings, model)
    corrupted = windows.copy()
    corrupted[0, 3:, 1] += 8.0
    fault = residual_windows(corrupted, settings, model)
    clean_score = residual_features(clean, model["scale"])[0][0, 1]
    fault_score = residual_features(fault, model["scale"])[0][0, 1]
    assert fault_score > clean_score
