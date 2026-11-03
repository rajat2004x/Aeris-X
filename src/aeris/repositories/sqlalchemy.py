"""Concrete SQLAlchemy repositories behind the domain repository boundary."""

import json
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from aeris.db.models import (
    AnomalyEventRecord,
    AssetRecord,
    FaultEventRecord,
    HealthEstimateRecord,
    InferenceRecord,
    ModelVersionRecord,
    RiskAssessmentRecord,
)
from aeris.domain.schemas import (
    Anomaly, Asset, FDIResult, HealthState, InferenceHistoryItem, InferenceRecord as InferenceRecordSchema,
    ModelMetadata, RiskAssessment,
)
from aeris.repositories.base import Repository


class AssetRepository(Repository[Asset]):
    def __init__(self, session: Session):
        self.session = session

    def get(self, entity_id: str) -> Asset | None:
        record = self.session.get(AssetRecord, entity_id)
        return Asset(
            asset_id=record.asset_id, asset_type=record.asset_type, name=record.name,
            source=record.source, metadata=json.loads(record.metadata_json)
        ) if record else None

    def save(self, entity: Asset) -> Asset:
        record = self.session.get(AssetRecord, entity.asset_id)
        if record is None:
            record = AssetRecord(asset_id=entity.asset_id)
            self.session.add(record)
        record.asset_type = entity.asset_type
        record.name = entity.name
        record.source = entity.source
        record.metadata_json = json.dumps(entity.metadata)
        self.session.flush()
        return entity


class ModelRepository(Repository[ModelMetadata]):
    def __init__(self, session: Session):
        self.session = session

    def get(self, entity_id: str) -> ModelMetadata | None:
        record = self.session.get(ModelVersionRecord, entity_id)
        stored = dict(record.metadata_json or {}) if record else {}
        return ModelMetadata(
            model_id=record.model_id, model_name=record.model_name, model_version=record.model_version,
            task=record.task, framework=record.framework, status=record.status,
            input_shape=tuple(stored.pop("input_shape")) if stored.get("input_shape") else None,
            expected_features=stored.pop("expected_features", None),
            dataset_reference=stored.pop("dataset_reference", None),
            metadata=stored.pop("metadata", {}),
            expected_sequence_length=stored.pop("expected_sequence_length", None),
            checkpoint_reference=stored.pop("checkpoint_reference", None),
            loading_strategy=stored.pop("loading_strategy", "lazy"),
            lifecycle_state=stored.pop("lifecycle_state", "AVAILABLE"),
            output_contract=stored.pop("output_contract", "health_score in [0, 1], finite"),
        ) if record else None

    def save(self, entity: ModelMetadata) -> ModelMetadata:
        record = self.session.get(ModelVersionRecord, entity.model_id)
        if record is None:
            record = ModelVersionRecord(model_id=entity.model_id)
            self.session.add(record)
        record.model_name = entity.model_name
        record.model_version = entity.model_version
        record.task = entity.task
        record.framework = entity.framework
        record.status = entity.status
        record.metadata_json = entity.model_dump(exclude={"model_id", "model_name", "model_version", "task", "framework", "status"})
        self.session.flush()
        return entity


class HealthRepository(Repository[HealthState]):
    def __init__(self, session: Session):
        self.session = session

    def get(self, entity_id: str) -> HealthState | None:
        record = self.session.get(HealthEstimateRecord, entity_id)
        return HealthState(
            asset_id=record.asset_id, health_score=record.health_score, model_id=record.model_id,
            model_version=record.model_version, timestamp=record.timestamp,
            uncertainty=record.uncertainty, evidence_ref=record.evidence_json.get("evidence_ref"),
        ) if record else None

    def save(self, entity: HealthState) -> HealthState:
        record = HealthEstimateRecord(
            id=str(uuid.uuid4()), asset_id=entity.asset_id, health_score=entity.health_score,
            model_id=entity.model_id, model_version=entity.model_version, timestamp=entity.timestamp,
            uncertainty=entity.uncertainty, evidence_json={"evidence_ref": entity.evidence_ref},
        )
        self.session.add(record)
        self.session.flush()
        return entity


class InferenceRepository:
    """Persistence boundary for operational inference executions."""

    def __init__(self, session: Session):
        self.session = session

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

    def save(self, entity: InferenceRecordSchema) -> InferenceRecordSchema:
        record = self.session.get(InferenceRecord, entity.inference_id)
        values = entity.model_dump()
        if record is None:
            record = InferenceRecord(**values)
            self.session.add(record)
        else:
            for key, value in values.items():
                setattr(record, key, value)
        self.session.flush()
        return entity

    def get(self, inference_id: str) -> InferenceHistoryItem | None:
        record = self.session.get(InferenceRecord, inference_id)
        return self._item(record) if record else None

    def list_by_asset(self, asset_id: str, limit: int = 100) -> list[InferenceHistoryItem]:
        records = self.session.scalars(
            select(InferenceRecord)
            .where(InferenceRecord.asset_id == asset_id)
            .order_by(InferenceRecord.created_at.desc())
            .limit(limit)
        ).all()
        return [self._item(record) for record in records]

    def list_health(self, asset_id: str, limit: int = 100) -> list[InferenceHistoryItem]:
        return [item for item in self.list_by_asset(asset_id, limit) if item.task == "HEALTH_ESTIMATION"]


class AnomalyRepository(Repository[Anomaly]):
    def __init__(self, session: Session):
        self.session = session

    def get(self, entity_id: str) -> Anomaly | None:
        record = self.session.get(AnomalyEventRecord, entity_id)
        return Anomaly(
            anomaly_id=record.id, asset_id=record.asset_id, timestamp=record.timestamp,
            severity=record.severity, score=record.score, detector=record.detector,
            status=record.status, evidence_ref=record.evidence_json.get("evidence_ref"),
        ) if record else None

    def save(self, entity: Anomaly) -> Anomaly:
        self.session.add(AnomalyEventRecord(
            id=entity.anomaly_id, asset_id=entity.asset_id, timestamp=entity.timestamp,
            score=entity.score, severity=entity.severity.value, detector=entity.detector,
            status=entity.status, evidence_json={"evidence_ref": entity.evidence_ref},
        ))
        self.session.flush()
        return entity


class FaultRepository(Repository[FDIResult]):
    def __init__(self, session: Session):
        self.session = session

    def get(self, entity_id: str) -> FDIResult | None:
        record = self.session.get(FaultEventRecord, entity_id)
        return FDIResult(
            fault_id=record.id, asset_id=record.asset_id, channel_id=record.channel_id,
            score=record.score, confidence=record.diagnostic_json.get("confidence", 0.0),
            model_version=record.diagnostic_json.get("model_version", "unknown"),
            timestamp=record.timestamp, diagnostic_metadata=record.diagnostic_json,
        ) if record else None

    def save(self, entity: FDIResult) -> FDIResult:
        self.session.add(FaultEventRecord(
            id=entity.fault_id, asset_id=entity.asset_id, timestamp=entity.timestamp,
            channel_id=entity.channel_id, score=entity.score, diagnostic_json={
                **entity.diagnostic_metadata, "confidence": entity.confidence, "model_version": entity.model_version
            },
        ))
        self.session.flush()
        return entity


class RiskRepository(Repository[RiskAssessment]):
    def __init__(self, session: Session):
        self.session = session

    def get(self, entity_id: str) -> RiskAssessment | None:
        record = self.session.get(RiskAssessmentRecord, entity_id)
        return RiskAssessment(
            asset_id=record.asset_id, risk_level=record.risk_level, score=record.risk_score,
            contributing_evidence=record.evidence_json.get("contributing_evidence", []),
            model_version=record.evidence_json.get("model_version", "unknown"), timestamp=record.timestamp,
        ) if record else None

    def save(self, entity: RiskAssessment) -> RiskAssessment:
        self.session.add(RiskAssessmentRecord(
            id=str(uuid.uuid4()), asset_id=entity.asset_id, timestamp=entity.timestamp,
            risk_score=entity.score, risk_level=entity.risk_level.value,
            evidence_json={"contributing_evidence": entity.contributing_evidence, "model_version": entity.model_version},
        ))
        self.session.flush()
        return entity
