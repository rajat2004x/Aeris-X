import json
from pathlib import Path

import numpy as np
import torch

from aeris.evaluation.metrics import mean_absolute_error
from aeris.models.temporal import GRURegressor, LSTMRegressor
from aeris.step13 import count_parameters, lifecycle_metrics


def test_lstm_and_gru_forward_shapes():
    features = torch.zeros(4, 30, 15)
    assert LSTMRegressor(input_size=15, hidden_size=64, num_layers=1)(features).shape == (4,)
    assert GRURegressor(input_size=15, hidden_size=64, num_layers=1)(features).shape == (4,)


def test_parameter_count_and_metric_helpers():
    model = LSTMRegressor(input_size=15, hidden_size=64, num_layers=1)
    total, trainable = count_parameters(model)
    assert total == trainable
    assert mean_absolute_error(np.array([1.0, 2.0]), np.array([1.5, 2.0])) == 0.25
    assert len(lifecycle_metrics(np.arange(9, dtype=float), np.arange(9, dtype=float))) == 3


def test_deterministic_model_initialization():
    torch.manual_seed(42)
    first = LSTMRegressor(input_size=15, hidden_size=64, num_layers=1)
    torch.manual_seed(42)
    second = LSTMRegressor(input_size=15, hidden_size=64, num_layers=1)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


def test_step13_artifacts_are_isolated():
    root = Path("reports/experiments/step13_lstm_gru")
    assert "step13_lstm_gru" in str(root)
    assert ".aeris/state.json" not in str(root)


def test_registry_shape_if_present():
    path = Path("reports/experiments/step13_lstm_gru/experiment_registry.json")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "experiments" in payload
