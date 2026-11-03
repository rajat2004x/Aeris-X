"""Framework-agnostic model-serving interfaces."""

from abc import ABC, abstractmethod
from typing import Any

from aeris.domain.schemas import ModelMetadata


class ModelPredictor(ABC):
    """Contract implemented by future Transformer, recurrent, FDI, and anomaly adapters."""

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        raise NotImplementedError

    @abstractmethod
    def predict(self, payload: Any) -> Any:
        raise NotImplementedError
