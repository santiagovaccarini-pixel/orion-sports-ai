from __future__ import annotations

import logging
from threading import RLock

from psycopg import errors
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


# Every row carries an owner so multi-user support later becomes an auth change
# and a WHERE clause, not a migration. Until then everything shares this one id.
DEFAULT_OWNER_ID = "orion"

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS memory_entries (
        id          TEXT PRIMARY KEY,
        owner_id    TEXT NOT NULL DEFAULT 'orion',
        content     TEXT NOT NULL,
        category    TEXT NOT NULL DEFAULT 'general',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS memory_entries_owner_created
        ON memory_entries (owner_id, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_documents (
        id          TEXT PRIMARY KEY,
        owner_id    TEXT NOT NULL DEFAULT 'orion',
        name        TEXT NOT NULL,
        content     TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS knowledge_documents_owner_created
        ON knowledge_documents (owner_id, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,
        owner_id    TEXT NOT NULL DEFAULT 'orion',
        title       TEXT NOT NULL,
        sport       TEXT NOT NULL DEFAULT 'general',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS conversations_owner_updated
        ON conversations (owner_id, updated_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_messages (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
        owner_id        TEXT NOT NULL DEFAULT 'orion',
        position        INTEGER NOT NULL,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS conversation_messages_conversation_position
        ON conversation_messages (conversation_id, position)
    """,
)


class DatabaseUnavailableError(RuntimeError):
    """The database is configured but could not be reached or prepared."""


_pool: ConnectionPool | None = None
# Reentrant on purpose: ensure_schema() holds this lock and then calls
# get_pool(), which takes it again. A plain Lock deadlocks there.
_pool_lock = RLock()
_schema_ready = False


def _build_pool(database_url: str) -> ConnectionPool:
    # Neon's pooled endpoint already fronts PgBouncer, so this pool stays small:
    # it exists to reuse a handful of warm connections, not to fan out.
    return ConnectionPool(
        conninfo=database_url,
        min_size=0,
        max_size=4,
        timeout=20.0,
        max_idle=120.0,
        kwargs={"connect_timeout": 15},
        open=True,
        check=ConnectionPool.check_connection,
    )


def get_pool(database_url: str) -> ConnectionPool:
    """Return the process-wide pool, creating it on first use."""

    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _build_pool(database_url)
        return _pool


def ensure_schema(database_url: str) -> None:
    """Create the tables if they do not exist yet. Safe to call repeatedly."""

    global _schema_ready
    if _schema_ready:
        return
    with _pool_lock:
        if _schema_ready:
            return
        pool = get_pool(database_url)
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    for statement in SCHEMA_STATEMENTS:
                        cur.execute(statement)
                conn.commit()
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc
        _schema_ready = True


def reset_pool() -> None:
    """Drop the cached pool and schema flag. Used by tests."""

    global _pool, _schema_ready
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.close()
            except Exception:  # pragma: no cover - close is best effort
                logger.debug("No se pudo cerrar el pool de base de datos", exc_info=True)
        _pool = None
        _schema_ready = False
