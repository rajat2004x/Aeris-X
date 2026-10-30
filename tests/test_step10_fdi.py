import numpy as np
import pytest
import torch

from aeris.evaluation.fdi import detection_metrics, isolation_rank, reconstruction_scores, threshold_from_normal
from aeris.faults.injection import inject_fault
from aeris.models.anomaly import SequenceAutoencoder


def test_fault_injectors_are_deterministic_and_preserve_input():
    original = np.zeros((10, 3), dtype=np.float32)
    for fault_type in ("BIAS", "DRIFT", "STUCK_AT", "NOISE", "INTERMITTENT_DROPOUT"):
        changed, metadata = inject_fault(original, 1, fault_type, 0.5, 0.5, 1.0, seed=7)
        repeat, _ = inject_fault(original, 1, fault_type, 0.5, 0.5, 1.0, seed=7)
        assert np.array_equal(changed, repeat)
        assert np.array_equal(original, np.zeros_like(original))
        assert metadata["fault_type"] == fault_type


def test_scores_threshold_and_detection_metrics():
    original = np.zeros((2, 3, 4))
    reconstructed = original.copy()
    reconstructed[1, :, 2] = 2
    scores, sensor_scores = reconstruction_scores(original, reconstructed)
    assert sensor_scores[1, 2] > sensor_scores[1, 0]
    threshold = threshold_from_normal(scores, 95)
    metrics = detection_metrics(np.array([0, 1]), scores, threshold)
    assert set(("precision", "recall", "f1", "pr_auc")) <= metrics.keys()


def test_isolation_and_autoencoder_forward():
    rank, top = isolation_rank(np.array([1.0, 5.0, 2.0]), 1)
    assert rank == 1 and top[0] == 1
    output = SequenceAutoencoder(input_size=4, hidden_size=3)(torch.randn(2, 5, 4))
    assert output.shape == (2, 5, 4)
