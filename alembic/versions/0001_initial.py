"""Initial Phase 14B persistence schema."""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "assets",
        sa.Column("asset_id", sa.String(128), primary_key=True),
        sa.Column("asset_type", sa.String(128), nullable=False),
        sa.Column("name", sa.String(256)),
        sa.Column("source", sa.String(256)),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "model_versions",
        sa.Column("model_id", sa.String(128), primary_key=True),
        sa.Column("model_name", sa.String(256), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("task", sa.String(128), nullable=False),
        sa.Column("framework", sa.String(128), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "health_estimates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("asset_id", sa.String(128), sa.ForeignKey("assets.asset_id"), nullable=False),
        sa.Column("health_score", sa.Float(), nullable=False),
        sa.Column("model_id", sa.String(128), sa.ForeignKey("model_versions.model_id"), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("uncertainty", sa.Float()),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
    )
    op.create_index("ix_health_asset_time", "health_estimates", ["asset_id", "timestamp"])
    op.create_table(
        "anomaly_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("asset_id", sa.String(128), sa.ForeignKey("assets.asset_id"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("severity", sa.String(32), nullable=False),
        sa.Column("detector", sa.String(128), nullable=False),
        sa.Column("model_id", sa.String(128), sa.ForeignKey("model_versions.model_id")),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
    )
    op.create_index("ix_anomaly_asset_time", "anomaly_events", ["asset_id", "timestamp"])
    op.create_table(
        "fault_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("asset_id", sa.String(128), sa.ForeignKey("assets.asset_id"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("channel_id", sa.String(128), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("model_id", sa.String(128), sa.ForeignKey("model_versions.model_id")),
        sa.Column("diagnostic_json", sa.JSON(), nullable=False),
    )
    op.create_table(
        "risk_assessments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("asset_id", sa.String(128), sa.ForeignKey("assets.asset_id"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(32), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("model_id", sa.String(128), sa.ForeignKey("model_versions.model_id")),
    )


def downgrade():
    for table in ("risk_assessments", "fault_events", "anomaly_events", "health_estimates", "model_versions", "assets"):
        op.drop_table(table)
