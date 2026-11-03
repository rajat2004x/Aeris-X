from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from aeris.evaluation.metrics import phm_score
from aeris.evaluation.temporal_benchmark import _load_classical_reference
from aeris.models.temporal import GRURegressor, LSTMRegressor, TCNRegressor, TransformerRegressor
from aeris.training.datasets import TemporalDataset
from aeris.training.trainer import Trainer, select_device

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_data():
    return np.random.default_rng(42).normal(size=(8, 6, 15)).astype(np.float32), np.ones(8, dtype=np.float32)


def test_dataset_loading_shape_and_finite_values(sample_data):
    dataset = TemporalDataset(*sample_data)
    assert len(dataset) == 8
    assert dataset[0][0].shape == (6, 15)
    assert torch.isfinite(dataset[0][0]).all()


@pytest.mark.parametrize("model", [
    LSTMRegressor(input_size=15, hidden_size=8, num_layers=1),
    GRURegressor(input_size=15, hidden_size=8, num_layers=1),
    TCNRegressor(input_size=15, channels=(8, 8, 8)),
    TransformerRegressor(input_size=15, d_model=8, nhead=2, num_layers=1, max_length=6),
])
def test_temporal_model_output_is_finite_and_counted(model):
    output = model(torch.randn(3, 6, 15))
    assert output.shape == (3,)
    assert torch.isfinite(output).all()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_trainer_executes_tiny_training_step(sample_data, tmp_path):
    dataset = TemporalDataset(*sample_data)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = LSTMRegressor(input_size=15, hidden_size=4, num_layers=1)
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), nn.MSELoss(), select_device(), tmp_path / "model.pt")
    history = trainer.fit(loader, loader, epochs=1)
    assert len(history["train_loss"]) == 1
    assert np.isfinite(history["train_loss"]).all()


def test_device_selection_and_existing_phm_metric():
    assert select_device().type in {"cpu", "cuda"}
    assert phm_score(np.array([1.0]), np.array([1.0])) == pytest.approx(0.0)


def test_step7_reference_has_reproducible_evaluation_metadata():
    reference = _load_classical_reference(
        ROOT / "reports/experiments/step8_temporal_dl",
        validation_size=2627,
    )
    assert reference["mae"] == pytest.approx(18.698258043515352)
    assert reference["training_time_seconds"] > 0
    assert reference["inference_time_seconds"] > 0
    assert reference["model_size_mb"] > 0
