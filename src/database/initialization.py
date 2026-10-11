"""Database engine/session management and initialization.

``init_db`` creates tables and syncs the material knowledge table from the JSON
source of truth. It is idempotent and **never deletes user data**.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ..config import SETTINGS
from ..logging_config import get_logger
from .models import Base, MaterialKnowledgeRecord

logger = get_logger("database")


@lru_cache(maxsize=1)
def _engine():
    SETTINGS.ensure_directories()
    engine = create_engine(
        f"sqlite:///{SETTINGS.db_path}",
        future=True,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _record):  # pragma: no cover - driver level
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=_engine(), expire_on_commit=False, future=True)


@lru_cache(maxsize=1)
def _session_factory_cached() -> sessionmaker:
    return get_session_factory()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session scope."""
    factory = _session_factory_cached()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _sync_material_knowledge(session: Session) -> int:
    try:
        raw = json.loads(SETTINGS.knowledge_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not sync material knowledge: %s", exc)
        return 0

    count = 0
    for entry in raw.get("categories", []):
        slug = entry["category_slug"]
        record = session.query(MaterialKnowledgeRecord).filter_by(category_slug=slug).one_or_none()
        values = dict(
            display_name=entry.get("display_name", slug),
            description=entry.get("description", ""),
            reuse_guidance=entry.get("reuse_guidance", ""),
            disposal_guidance=entry.get("disposal_guidance", ""),
            safety_notes=entry.get("safety_notes", ""),
            source_url=entry.get("source_url", ""),
            geographic_scope=entry.get("geographic_scope", ""),
            reviewed_at=entry.get("reviewed_at", ""),
        )
        if record is None:
            session.add(MaterialKnowledgeRecord(category_slug=slug, **values))
        else:
            for key, value in values.items():
                setattr(record, key, value)
        count += 1
    return count


def init_db(sync_knowledge: bool = True) -> None:
    """Create tables, run backward-compatible migrations, sync knowledge.

    Idempotent and **never destructive**: new tables are added by
    ``create_all``; existing tables receive only additive column migrations.
    """
    SETTINGS.ensure_directories()
    Base.metadata.create_all(_engine())
    _run_lightweight_migrations()
    if sync_knowledge:
        with session_scope() as session:
            _sync_material_knowledge(session)
            logger.info("Knowledge synced", extra={"event": "kb_sync"})
    logger.info("Database ready", extra={"event": "db_init"})


def _table_names(conn) -> set[str]:
    rows = conn.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _column_names(conn, table: str) -> set[str]:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _run_lightweight_migrations() -> None:
    """Additive-only schema migrations for databases created by earlier builds.

    SQLite ``ALTER TABLE ... ADD COLUMN`` is non-destructive; existing rows and
    stored records are preserved. No column is ever dropped or rewritten.
    """
    engine = _engine()
    with engine.begin() as conn:
        tables = _table_names(conn)

        if "feedback" in tables:
            columns = _column_names(conn, "feedback")
            if "updated_at" not in columns:
                conn.exec_driver_sql("ALTER TABLE feedback ADD COLUMN updated_at DATETIME")
                conn.exec_driver_sql(
                    "UPDATE feedback SET updated_at = created_at WHERE updated_at IS NULL"
                )
                logger.info("Migration applied: feedback.updated_at", extra={"event": "migrate"})

            duplicates = conn.exec_driver_sql(
                "SELECT analysis_id, source FROM feedback "
                "GROUP BY analysis_id, source HAVING COUNT(*) > 1"
            ).fetchall()
            if not duplicates:
                conn.exec_driver_sql(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_analysis_source "
                    "ON feedback (analysis_id, source)"
                )


def reset_db() -> None:
    """Drop and recreate all tables. Destructive — for maintenance/tests only."""
    Base.metadata.drop_all(_engine())
    Base.metadata.create_all(_engine())
    with session_scope() as session:
        _sync_material_knowledge(session)
    logger.warning("Database reset", extra={"event": "db_reset"})


def reset_caches() -> None:
    """Clear cached engine/session factory (used by tests switching DB paths)."""
    _engine.cache_clear()
    _session_factory_cached.cache_clear()
