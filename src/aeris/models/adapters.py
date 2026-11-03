"""Lazy adapters for validated research checkpoints."""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from aeris.domain.schemas import ModelMetadata
from aeris.models.interfaces import ModelPredictor
from aeris.models.temporal import GRURegressor, TransformerRegressor


class ModelUnavailableError(RuntimeError):
    """Raised when a configured research artifact is unavailable."""


class ModelNotFoundError(LookupError):
    """Raised when a requested model ID is not registered."""


class InvalidModelInputError(ValueError):
    """Raised when inference input violates the model data contract."""


class TemporalHealthPredictor(ModelPredictor):
    def __init__(
        self,
        model_id: str,
        version: str,
        checkpoint: Path,
        builder: Callable[[], torch.nn.Module],
        model_name: str,
    ) -> None:
        self._model_id = model_id
        self._version = version
        self._checkpoint = checkpoint
        self._builder = builder
        self._model_name = model_name
        self._model: torch.nn.Module | None = None

    def metadata(self) -> ModelMetadata:
        available = self._checkpoint.exists()
        return ModelMetadata(
            model_id=self._model_id,
            model_name=self._model_name,
            model_version=self._version,
            task="health_estimation",
            framework="pytorch",
            dataset_reference="NASA C-MAPSS FD001",
            status="available" if available else "unavailable",
            input_shape=(30, 15),
            expected_features=15,
            expected_sequence_length=30,
            checkpoint_reference=self._checkpoint.name,
            loading_strategy="lazy",
            lifecycle_state="AVAILABLE" if available else "UNAVAILABLE",
            output_contract="finite health score in [0, 1]",
            metadata={
                "artifact_available": available,
                "health_definition": "clip(predicted_rul / 332.0, 0, 1)",
                "device": self._device().type,
            },
        )

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self) -> torch.nn.Module:
        if not self._checkpoint.exists():
            raise ModelUnavailableError(f"Model artifact unavailable: {self._model_id}")
        if self._model is None:
            model = self._builder().to(self._device())
            model.load_state_dict(torch.load(self._checkpoint, map_location=self._device(), weights_only=True))
            model.eval()
            self._model = model
        return self._model

    def predict(self, payload: Any) -> float:
        values = np.asarray(payload, dtype=np.float32)
        if values.shape != (30, 15):
            raise InvalidModelInputError("health input must have shape [30, 15].")
        if not np.isfinite(values).all():
            raise InvalidModelInputError("health input must contain only finite numbers.")
        model = self._load()
        with torch.no_grad():
            rul = float(model(torch.as_tensor(values, device=self._device()).unsqueeze(0)).item())
        if not np.isfinite(rul):
            raise RuntimeError("Model produced a non-finite RUL prediction.")
        return float(np.clip(rul / 332.0, 0.0, 1.0))


class TransformerHealthPredictor(TemporalHealthPredictor):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__(
            "aeris-transformer-health", "step8-fd001", checkpoint,
            lambda: TransformerRegressor(input_size=15, max_length=30), "Transformer Health Estimator",
        )


class GRUHealthPredictor(TemporalHealthPredictor):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__(
            "aeris-gru-health", "step13-fd001", checkpoint,
            lambda: GRURegressor(input_size=15, hidden_size=64, num_layers=1), "GRU Health Estimator",
        )


class ModelRegistry:
    """Lazy model registry; listing metadata never loads checkpoints."""

    def __init__(self, repository_root: Path) -> None:
        self._factories: dict[str, Callable[[], TemporalHealthPredictor]] = {
            "aeris-transformer-health": lambda: TransformerHealthPredictor(
                repository_root / "reports/experiments/step8_temporal_dl/models/transformer.pt"
            ),
            "aeris-gru-health": lambda: GRUHealthPredictor(
                repository_root / "reports/experiments/step13_lstm_gru/models/gru.pt"
            ),
        }
        self._models: dict[str, TemporalHealthPredictor] = {}

    def get(self, model_id: str) -> TemporalHealthPredictor:
        if model_id not in self._factories:
            raise ModelNotFoundError(f"Unknown model: {model_id}")
        if model_id not in self._models:
            self._models[model_id] = self._factories[model_id]()
        return self._models[model_id]

    def list_metadata(self) -> list[ModelMetadata]:
        return [self.get(model_id).metadata() for model_id in self._factories]

    def resolve(self, model_id: str, task: str = "health_estimation") -> TemporalHealthPredictor:
        model = self.get(model_id)
        if model.metadata().task != task:
            raise InvalidModelInputError(f"Model {model_id} does not support task {task}.")
        return model
