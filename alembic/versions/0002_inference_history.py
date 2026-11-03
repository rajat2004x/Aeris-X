"""Add operational inference history."""

from alembic import op
import sqlalchemy as sa

revision = "0002_inference_history"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "inference_records",
        sa.Column("inference_id", sa.String(36), primary_key=True),
        sa.Column("asset_id", sa.String(128), sa.ForeignKey("assets.asset_id"), nullable=False),
        sa.Column("request_id", sa.String(128)),
        sa.Column("model_id", sa.String(128), sa.ForeignKey("model_versions.model_id")),
        sa.Column("model_version", sa.String(128)),
        sa.Column("task", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("duration_ms", sa.Float()),
        sa.Column("input_quality_status", sa.String(32), nullable=False),
        sa.Column("output_status", sa.String(32), nullable=False),
        sa.Column("error_category", sa.String(64)),
        sa.Column("error_message", sa.String(512)),
        sa.Column("health_score", sa.Float()),
    )
    op.create_index("ix_inference_asset_created", "inference_records", ["asset_id", "created_at"])
    op.create_index("ix_inference_records_asset_id", "inference_records", ["asset_id"])
    op.create_index("ix_inference_records_request_id", "inference_records", ["request_id"])
    op.create_index("ix_inference_records_status", "inference_records", ["status"])


def downgrade():
    op.drop_index("ix_inference_records_status", table_name="inference_records")
    op.drop_index("ix_inference_records_request_id", table_name="inference_records")
    op.drop_index("ix_inference_records_asset_id", table_name="inference_records")
    op.drop_index("ix_inference_asset_created", table_name="inference_records")
    op.drop_table("inference_records")
