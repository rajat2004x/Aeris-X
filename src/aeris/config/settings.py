"""Environment-driven application settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="AERIS-X", alias="AERIS_APP_NAME")
    version: str = Field(default="0.14.0", alias="AERIS_VERSION")
    environment: str = Field(default="development", alias="AERIS_ENV")
    database_url: str = Field(
        default="postgresql+psycopg://localhost:5432/aeris",
        alias="DATABASE_URL",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    request_id_header: str = "X-Request-ID"
    max_telemetry_rows: int = Field(default=1000, alias="AERIS_MAX_TELEMETRY_ROWS", ge=1, le=10000)
    max_telemetry_features: int = Field(default=256, alias="AERIS_MAX_TELEMETRY_FEATURES", ge=1, le=2048)
    max_history_limit: int = Field(default=100, alias="AERIS_MAX_HISTORY_LIMIT", ge=1, le=1000)
    max_request_body_bytes: int = Field(default=2_000_000, alias="AERIS_MAX_REQUEST_BODY_BYTES", ge=1024, le=20_000_000)
    cors_origins: str = Field(default="", alias="AERIS_CORS_ORIGINS")
    trusted_hosts: str = Field(default="", alias="AERIS_TRUSTED_HOSTS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
