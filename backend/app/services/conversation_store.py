"""Conversations that survive closing the page.

Until this existed, a reload erased the whole exchange: the interface warned
"las conversaciones se pierden al recargar la página" and meant it. The chat
itself was the only stateful thing in Orion still living exclusively in one
browser tab.

The store keeps whole exchanges, not keystrokes: the interface appends a user
message and its finished answer together once the answer completes. Charts,
latency chips and other per-answer diagnostics are deliberately not stored -
they describe one live run, and the diagnostics panel is where runs live.

Same shape as the memory and knowledge stores: a JSON file for local use and
tests, Postgres in deployment, one Protocol so nothing above cares which.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

# A rolling history, not an archive: when the cap is reached, creating a new
# conversation removes the oldest one. Raising the cap is one number; silently
# growing forever on a free-tier database is not an option.
MAX_CONVERSATIONS = 200
MAX_MESSAGES_PER_CONVERSATION = 400


@dataclass(frozen=True, slots=True)
class StoredMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    id: str
    title: str
    sport: str
    created_at: str
    updated_at: str
    message_count: int


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    title: str
    sport: str
    created_at: str
    updated_at: str
    messages: tuple[StoredMessage, ...]


class ConversationStore:
    """File-backed conversations for local runs and the test suite."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def list_conversations(self) -> list[ConversationSummary]:
        with self._lock:
            records = self._read()
        summaries = [self._summary(record) for record in records]
        summaries.sort(key=lambda item: item.updated_at, reverse=True)
        return summaries

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            records = self._read()
        for record in records:
            if record["id"] == conversation_id:
                return Conversation(
                    id=record["id"],
                    title=record["title"],
                    sport=record["sport"],
                    created_at=record["created_at"],
                    updated_at=record["updated_at"],
                    messages=tuple(
                        StoredMessage(item["role"], item["content"])
                        for item in record["messages"]
                    ),
                )
        return None

    def create_conversation(
        self, conversation_id: str, title: str, sport: str
    ) -> ConversationSummary:
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": conversation_id,
            "title": title.strip(),
            "sport": sport.strip() or "general",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        with self._lock:
            records = [item for item in self._read() if item["id"] != conversation_id]
            records.append(record)
            records.sort(key=lambda item: item["updated_at"])
            while len(records) > MAX_CONVERSATIONS:
                records.pop(0)
            self._write(records)
        return self._summary(record)

    def append_messages(
        self, conversation_id: str, messages: list[StoredMessage]
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            records = self._read()
            for record in records:
                if record["id"] != conversation_id:
                    continue
                record["messages"].extend(
                    {"role": item.role, "content": item.content} for item in messages
                )
                # Rolling window inside one conversation too: the earliest
                # exchanges fall off rather than the write being refused, since
                # refusing would silently stop persistence mid-conversation.
                overflow = len(record["messages"]) - MAX_MESSAGES_PER_CONVERSATION
                if overflow > 0:
                    record["messages"] = record["messages"][overflow:]
                record["updated_at"] = now
                self._write(records)
                return True
        return False

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            records = self._read()
            remaining = [item for item in records if item["id"] != conversation_id]
            if len(remaining) == len(records):
                return False
            self._write(remaining)
            return True

    def _summary(self, record: dict) -> ConversationSummary:
        return ConversationSummary(
            id=record["id"],
            title=record["title"],
            sport=record["sport"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            message_count=len(record["messages"]),
        )

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(payload, list):
            return []
        records: list[dict] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if not all(
                isinstance(item.get(key), str)
                for key in ("id", "title", "sport", "created_at", "updated_at")
            ):
                continue
            messages = item.get("messages")
            if not isinstance(messages, list):
                continue
            clean_messages = [
                {"role": str(entry.get("role")), "content": str(entry.get("content"))}
                for entry in messages
                if isinstance(entry, dict)
            ]
            records.append({**item, "messages": clean_messages})
        return records

    def _write(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
