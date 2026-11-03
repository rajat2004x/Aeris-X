"""Repository interfaces kept independent from SQLAlchemy."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Entity = TypeVar("Entity")


class Repository(ABC, Generic[Entity]):
    @abstractmethod
    def get(self, entity_id: str) -> Entity | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, entity: Entity) -> Entity:
        raise NotImplementedError
