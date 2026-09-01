from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from psycopg import errors
from psycopg.rows import tuple_row

from backend.app.services.conversation_store import (
    MAX_CONVERSATIONS,
    MAX_MESSAGES_PER_CONVERSATION,
    Conversation,
    ConversationStore,
    ConversationSummary,
    StoredMessage,
)
from backend.app.services.database import (
    DEFAULT_OWNER_ID,
    DatabaseUnavailableError,
    ensure_schema,
    get_pool,
)


class ConversationRepository(Protocol):
    """Mirrors ConversationStore's methods so the file store satisfies it as-is."""

    def list_conversations(self) -> list[ConversationSummary]: ...

    def get_conversation(self, conversation_id: str) -> Conversation | None: ...

    def create_conversation(
        self, conversation_id: str, title: str, sport: str
    ) -> ConversationSummary: ...

    def append_messages(
        self, conversation_id: str, messages: list[StoredMessage]
    ) -> bool: ...

    def delete_conversation(self, conversation_id: str) -> bool: ...


class PostgresConversationStore:
    """Conversations in Postgres so a reload, restart or redeploy keeps them."""

    def __init__(self, database_url: str, owner_id: str = DEFAULT_OWNER_ID) -> None:
        self._url = database_url
        self._owner_id = owner_id

    def _pool(self):
        ensure_schema(self._url)
        return get_pool(self._url)

    def list_conversations(self) -> list[ConversationSummary]:
        try:
            with self._pool().connection() as conn:
                with conn.cursor(row_factory=tuple_row) as cur:
                    cur.execute(
                        """
                        SELECT c.id, c.title, c.sport, c.created_at, c.updated_at,
                               count(m.id)
                        FROM conversations c
                        LEFT JOIN conversation_messages m ON m.conversation_id = c.id
                        WHERE c.owner_id = %s
                        GROUP BY c.id
                        ORDER BY c.updated_at DESC
                        """,
                        (self._owner_id,),
                    )
                    return [
                        ConversationSummary(
                            id=row[0],
                            title=row[1],
                            sport=row[2],
                            created_at=row[3].isoformat(),
                            updated_at=row[4].isoformat(),
                            message_count=row[5],
                        )
                        for row in cur.fetchall()
                    ]
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        try:
            with self._pool().connection() as conn:
                with conn.cursor(row_factory=tuple_row) as cur:
                    cur.execute(
                        """
                        SELECT id, title, sport, created_at, updated_at
                        FROM conversations
                        WHERE owner_id = %s AND id = %s
                        """,
                        (self._owner_id, conversation_id),
                    )
                    head = cur.fetchone()
                    if head is None:
                        return None
                    cur.execute(
                        """
                        SELECT role, content
                        FROM conversation_messages
                        WHERE conversation_id = %s
                        ORDER BY position
                        """,
                        (conversation_id,),
                    )
                    messages = tuple(
                        StoredMessage(role=row[0], content=row[1])
                        for row in cur.fetchall()
                    )
            return Conversation(
                id=head[0],
                title=head[1],
                sport=head[2],
                created_at=head[3].isoformat(),
                updated_at=head[4].isoformat(),
                messages=messages,
            )
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def create_conversation(
        self, conversation_id: str, title: str, sport: str
    ) -> ConversationSummary:
        now = datetime.now(timezone.utc)
        try:
            with self._pool().connection() as conn:
                with conn.cursor(row_factory=tuple_row) as cur:
                    cur.execute(
                        """
                        INSERT INTO conversations
                            (id, owner_id, title, sport, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                            SET title = EXCLUDED.title,
                                updated_at = EXCLUDED.updated_at
                        """,
                        (
                            conversation_id,
                            self._owner_id,
                            title.strip(),
                            sport.strip() or "general",
                            now,
                            now,
                        ),
                    )
                    # Rolling history: past the cap, the oldest conversation
                    # (and its messages, via ON DELETE CASCADE) makes room.
                    cur.execute(
                        """
                        DELETE FROM conversations
                        WHERE owner_id = %s AND id IN (
                            SELECT id FROM conversations
                            WHERE owner_id = %s
                            ORDER BY updated_at DESC
                            OFFSET %s
                        )
                        """,
                        (self._owner_id, self._owner_id, MAX_CONVERSATIONS),
                    )
                conn.commit()
            return ConversationSummary(
                id=conversation_id,
                title=title.strip(),
                sport=sport.strip() or "general",
                created_at=now.isoformat(),
                updated_at=now.isoformat(),
                message_count=0,
            )
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def append_messages(
        self, conversation_id: str, messages: list[StoredMessage]
    ) -> bool:
        now = datetime.now(timezone.utc)
        try:
            with self._pool().connection() as conn:
                with conn.cursor(row_factory=tuple_row) as cur:
                    cur.execute(
                        """
                        SELECT count(*) FROM conversations
                        WHERE owner_id = %s AND id = %s
                        """,
                        (self._owner_id, conversation_id),
                    )
                    if cur.fetchone()[0] == 0:
                        return False
                    cur.execute(
                        """
                        SELECT coalesce(max(position), -1)
                        FROM conversation_messages
                        WHERE conversation_id = %s
                        """,
                        (conversation_id,),
                    )
                    next_position = cur.fetchone()[0] + 1
                    for offset, message in enumerate(messages):
                        cur.execute(
                            """
                            INSERT INTO conversation_messages
                                (id, conversation_id, owner_id, position, role,
                                 content, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                f"{conversation_id}-{next_position + offset}",
                                conversation_id,
                                self._owner_id,
                                next_position + offset,
                                message.role,
                                message.content,
                                now,
                            ),
                        )
                    # Rolling window inside the conversation: keep the newest
                    # MAX_MESSAGES_PER_CONVERSATION rather than refusing the
                    # write, which would silently stop persistence mid-chat.
                    cur.execute(
                        """
                        DELETE FROM conversation_messages
                        WHERE conversation_id = %s AND position <= (
                            SELECT max(position) FROM conversation_messages
                            WHERE conversation_id = %s
                        ) - %s
                        """,
                        (conversation_id, conversation_id, MAX_MESSAGES_PER_CONVERSATION),
                    )
                    cur.execute(
                        "UPDATE conversations SET updated_at = %s WHERE id = %s",
                        (now, conversation_id),
                    )
                conn.commit()
            return True
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def delete_conversation(self, conversation_id: str) -> bool:
        try:
            with self._pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM conversations WHERE owner_id = %s AND id = %s",
                        (self._owner_id, conversation_id),
                    )
                    deleted = cur.rowcount
                conn.commit()
            return deleted > 0
        except errors.Error as exc:
            raise DatabaseUnavailableError(str(exc)) from exc


def create_conversation_repository(
    *,
    database_url: str | None,
    conversations_path: str,
) -> ConversationRepository:
    """Postgres when a database is configured, otherwise the local JSON file."""

    if database_url:
        return PostgresConversationStore(database_url)
    return ConversationStore(Path(conversations_path))
