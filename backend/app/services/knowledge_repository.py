from __future__ import annotations

from pathlib import Path

from psycopg import errors
from psycopg.rows import tuple_row

from backend.app.services.database import (
    DEFAULT_OWNER_ID,
    DatabaseUnavailableError,
    ensure_schema,
    get_pool,
)
from backend.app.services.knowledge_base import KnowledgeBase, KnowledgeDocument


class PostgresKnowledgeBase(KnowledgeBase):
    """Documents stored in Postgres instead of a JSON file on an ephemeral disk.

    Subclasses KnowledgeBase deliberately: only the two storage methods change,
    so chunking and search behaviour stay literally the same code rather than a
    parallel implementation that could drift.
    """

    def __init__(self, database_url: str, owner_id: str = DEFAULT_OWNER_ID) -> None:
        # No file path: this backend never touches the filesystem.
        self.path = Path()
        self._url = database_url
        self._owner_id = owner_id

    def _pool(self):
        ensure_schema(self._url)
        return get_pool(self._url)

    def list_documents(self) -> list[KnowledgeDocument]:
        try:
            with self._pool().connection() as conn:
                with conn.cursor(row_factory=tuple_row) as cur:
                    cur.execute(
                        """
                        SELECT id, name, content
                        FROM knowledge_documents
                        WHERE owner_id = %s
                        ORDER BY created_at
                        """,
                        (self._owner_id,),
                    )
                    return [KnowledgeDocument(*row) for row in cur.fetchall()]
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def add_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        try:
            with self._pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO knowledge_documents (id, owner_id, name, content)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                            SET name = EXCLUDED.name,
                                content = EXCLUDED.content
                        """,
                        (document.id, self._owner_id, document.name, document.content),
                    )
                conn.commit()
            return document
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def _read_documents(self) -> list[KnowledgeDocument]:
        return self.list_documents()


def create_knowledge_base(
    *,
    database_url: str | None,
    knowledge_path: str,
) -> KnowledgeBase:
    """Postgres when a database is configured, otherwise the local JSON file."""

    if database_url:
        return PostgresKnowledgeBase(database_url)
    return KnowledgeBase(Path(knowledge_path))
