"""Small operational history store used when no database session is configured."""

from aeris.domain.schemas import InferenceHistoryItem, InferenceRecord


class InMemoryInferenceRepository:
    """Process-local fallback for development and database-unavailable smoke tests."""

    def __init__(self) -> None:
        self.records: dict[str, InferenceRecord] = {}

    def save(self, entity: InferenceRecord) -> InferenceRecord:
        self.records[entity.inference_id] = entity
        return entity

    def get(self, inference_id: str) -> InferenceHistoryItem | None:
        record = self.records.get(inference_id)
        return self._item(record) if record else None

    def list_by_asset(self, asset_id: str, limit: int = 100) -> list[InferenceHistoryItem]:
        records = [record for record in self.records.values() if record.asset_id == asset_id]
        records.sort(key=lambda record: record.created_at, reverse=True)
        return [self._item(record) for record in records[:limit]]

    def list_health(self, asset_id: str, limit: int = 100) -> list[InferenceHistoryItem]:
        return [item for item in self.list_by_asset(asset_id, limit) if item.task == "HEALTH_ESTIMATION"]

    @staticmethod
    def _item(record: InferenceRecord) -> InferenceHistoryItem:
        return InferenceHistoryItem(
            inference_id=record.inference_id,
            asset_id=record.asset_id,
            task=record.task,
            model_id=record.model_id,
            model_version=record.model_version,
            status=record.status,
            created_at=record.created_at,
            completed_at=record.completed_at,
            duration_ms=record.duration_ms,
            health_score=record.health_score,
            error_category=record.error_category,
        )
