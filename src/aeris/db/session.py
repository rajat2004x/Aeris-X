"""Lazy SQLAlchemy engine and session factory."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from aeris.config.settings import get_settings


def build_engine(database_url: str | None = None):
    return create_engine(database_url or get_settings().database_url, pool_pre_ping=True)


def get_session(database_url: str | None = None) -> Generator[Session, None, None]:
    factory = sessionmaker(bind=build_engine(database_url), autoflush=False, autocommit=False)
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
