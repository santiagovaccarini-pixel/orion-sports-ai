from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from psycopg import errors
from psycopg.rows import tuple_row

from backend.app.services.database import (
    DEFAULT_OWNER_ID,
    DatabaseUnavailableError,
    ensure_schema,
    get_pool,
)
from backend.app.services.memory_store import MemoryEntry, MemoryStore


class MemoryRepository(Protocol):
    """Storage for facts the user explicitly asked Orion to remember.

    Deliberately mirrors MemoryStore's method names so the file-backed store
    satisfies this protocol as-is: swapping storage is a wiring change, and
    nothing above this layer needs to know which one is in use.
    """

    def list_entries(self) -> list[MemoryEntry]: ...

    def get_entry(self, entry_id: str) -> MemoryEntry | None: ...

    def add_entry(self, entry_id: str, content: str, category: str) -> MemoryEntry: ...

    def delete_entry(self, entry_id: str) -> bool: ...

    def delete_all(self) -> None: ...


def _as_entry(row: tuple) -> MemoryEntry:
    entry_id, content, category, created_at, updated_at = row
    return MemoryEntry(
        id=entry_id,
        content=content,
        category=category,
        created_at=created_at.isoformat(),
        updated_at=updated_at.isoformat(),
    )


class PostgresMemoryStore:
    """Memory entries stored in Postgres so they survive restarts and redeploys."""

    def __init__(self, database_url: str, owner_id: str = DEFAULT_OWNER_ID) -> None:
        self._url = database_url
        self._owner_id = owner_id

    def _pool(self):
        ensure_schema(self._url)
        return get_pool(self._url)

    def list_entries(self) -> list[MemoryEntry]:
        try:
            with self._pool().connection() as conn:
                with conn.cursor(row_factory=tuple_row) as cur:
                    cur.execute(
                        """
                        SELECT id, content, category, created_at, updated_at
                        FROM memory_entries
                        WHERE owner_id = %s
                        ORDER BY created_at
                        """,
                        (self._owner_id,),
                    )
                    return [_as_entry(row) for row in cur.fetchall()]
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        try:
            with self._pool().connection() as conn:
                with conn.cursor(row_factory=tuple_row) as cur:
                    cur.execute(
                        """
                        SELECT id, content, category, created_at, updated_at
                        FROM memory_entries
                        WHERE owner_id = %s AND id = %s
                        """,
                        (self._owner_id, entry_id),
                    )
                    row = cur.fetchone()
                    return _as_entry(row) if row else None
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def add_entry(self, entry_id: str, content: str, category: str) -> MemoryEntry:
        now = datetime.now(timezone.utc)
        try:
            with self._pool().connection() as conn:
                with conn.cursor(row_factory=tuple_row) as cur:
                    cur.execute(
                        """
                        INSERT INTO memory_entries
                            (id, owner_id, content, category, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                            SET content = EXCLUDED.content,
                                category = EXCLUDED.category,
                                updated_at = EXCLUDED.updated_at
                        RETURNING id, content, category, created_at, updated_at
                        """,
                        (
                            entry_id,
                            self._owner_id,
                            content.strip(),
                            category.strip(),
                            now,
                            now,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            return _as_entry(row)
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def delete_entry(self, entry_id: str) -> bool:
        try:
            with self._pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM memory_entries WHERE owner_id = %s AND id = %s",
                        (self._owner_id, entry_id),
                    )
                    deleted = cur.rowcount
                conn.commit()
            return deleted > 0
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def delete_all(self) -> None:
        try:
            with self._pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM memory_entries WHERE owner_id = %s",
                        (self._owner_id,),
                    )
                conn.commit()
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc


def create_memory_repository(
    *,
    database_url: str | None,
    memory_path: str,
) -> MemoryRepository:
    """Postgres when a database is configured, otherwise the local JSON file.

    The file fallback keeps local development and the test suite working with
    no database at all; it is not durable on an ephemeral filesystem, which is
    exactly why the deployed service sets DATABASE_URL.
    """

    if database_url:
        return PostgresMemoryStore(database_url)
    return MemoryStore(Path(memory_path))
