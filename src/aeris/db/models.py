"""Initial persistence entities, separate from API schemas."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AssetRecord(Base):
    __tablename__ = "assets"
    asset_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_type: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str | None] = mapped_column(String(256))
    source: Mapped[str | None] = mapped_column(String(256))
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"
    model_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    task: Mapped[str] = mapped_column(String(128), nullable=False)
    framework: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class HealthEstimateRecord(Base):
    __tablename__ = "health_estimates"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(128), ForeignKey("assets.asset_id"), nullable=False, index=True)
    health_score: Mapped[float] = mapped_column(Float, nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), ForeignKey("model_versions.model_id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    uncertainty: Mapped[float | None] = mapped_column(Float)
    evidence_json: Mapped[dict] = mapped_column(JSON, default=dict)


class InferenceRecord(Base):
    __tablename__ = "inference_records"
    inference_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(128), ForeignKey("assets.asset_id"), nullable=False, index=True)
    request_id: Mapped[str | None] = mapped_column(String(128), index=True)
    model_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("model_versions.model_id"))
    model_version: Mapped[str | None] = mapped_column(String(128))
    task: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    input_quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    output_status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_category: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String(512))
    health_score: Mapped[float | None] = mapped_column(Float)


class AnomalyEventRecord(Base):
    __tablename__ = "anomaly_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(128), ForeignKey("assets.asset_id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    detector: Mapped[str] = mapped_column(String(128), nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("model_versions.model_id"))
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_json: Mapped[dict] = mapped_column(JSON, default=dict)


class FaultEventRecord(Base):
    __tablename__ = "fault_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(128), ForeignKey("assets.asset_id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("model_versions.model_id"))
    diagnostic_json: Mapped[dict] = mapped_column(JSON, default=dict)


class RiskAssessmentRecord(Base):
    __tablename__ = "risk_assessments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(128), ForeignKey("assets.asset_id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False)
    evidence_json: Mapped[dict] = mapped_column(JSON, default=dict)
    model_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("model_versions.model_id"))


Index("ix_anomaly_asset_time", AnomalyEventRecord.asset_id, AnomalyEventRecord.timestamp)
Index("ix_health_asset_time", HealthEstimateRecord.asset_id, HealthEstimateRecord.timestamp)
Index("ix_inference_asset_created", InferenceRecord.asset_id, InferenceRecord.created_at)
