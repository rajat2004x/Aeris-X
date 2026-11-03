import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aeris.health.improvement import (
    LifecycleHuberLoss,
    lifecycle_metrics,
    temporal_consistency_penalty,
)
from aeris.health.improvement import predict_transformer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/experiments/step12_health_state_improvement"
sys.path.insert(0, str(ROOT / "scripts"))
from run_step12_health_state_improvement import run_improvement


def test_lifecycle_metrics_and_huber_are_deterministic():
    actual = np.array([0.9, 0.6, 0.2])
    predicted = np.array([0.8, 0.5, 0.3])
    assert lifecycle_metrics(actual, predicted) == {
        "early_life_mae": pytest.approx(0.1),
        "mid_life_mae": pytest.approx(0.1),
        "late_life_mae": pytest.approx(0.1),
    }
    loss = LifecycleHuberLoss()(torch.tensor([0.8, 0.5, 0.3]), torch.tensor([0.9, 0.6, 0.2]))
    assert torch.isfinite(loss)


def test_temporal_penalty_respects_engine_boundaries():
    prediction = torch.tensor([0.4, 0.6, 0.2, 0.1])
    target = torch.tensor([0.5, 0.4, 0.3, 0.2])
    owners = torch.tensor([1, 1, 2, 2])
    assert temporal_consistency_penalty(prediction, target, owners).item() == pytest.approx(0.1)


def test_baseline_artifacts_are_unchanged_and_pipeline_is_reproducible():
    baseline = ROOT / "reports/experiments/step12_health_state/final_results.csv"
    original = baseline.read_bytes()
    first = run_improvement(output_dir=OUTPUT, epochs=1)
    assert baseline.read_bytes() == original
    second = run_improvement(output_dir=OUTPUT, epochs=1)
    assert first["selected"] == second["selected"]
    assert first["accepted"] == second["accepted"]
    audit = json.loads((OUTPUT / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["final_holdout_used_for_selection"] is False
    assert (OUTPUT / "FINAL_STEP12_IMPROVEMENT_REPORT.md").exists()
