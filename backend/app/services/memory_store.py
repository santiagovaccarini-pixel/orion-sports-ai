from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from backend.app.services.knowledge_base import _terms


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A single fact the user chose to save explicitly.

    Memory entries only exist because the user created them through the
    dedicated memory action in the interface. Orion never writes here from a
    chat message alone, and nothing here is inferred silently by the model.
    """

    id: str
    content: str
    category: str
    created_at: str
    updated_at: str


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def list_entries(self) -> list[MemoryEntry]:
        with self._lock:
            return self._read()

    def add_entry(self, entry_id: str, content: str, category: str) -> MemoryEntry:
        now = datetime.now(timezone.utc).isoformat()
        entry = MemoryEntry(entry_id, content.strip(), category, now, now)
        with self._lock:
            entries = [item for item in self._read() if item.id != entry_id]
            entries.append(entry)
            self._write(entries)
        return entry

    def update_entry(
        self, entry_id: str, content: str, category: str | None = None
    ) -> MemoryEntry | None:
        with self._lock:
            entries = self._read()
            updated: MemoryEntry | None = None
            next_entries: list[MemoryEntry] = []
            for item in entries:
                if item.id == entry_id:
                    updated = MemoryEntry(
                        item.id,
                        content.strip(),
                        category or item.category,
                        item.created_at,
                        datetime.now(timezone.utc).isoformat(),
                    )
                    next_entries.append(updated)
                else:
                    next_entries.append(item)
            if updated is not None:
                self._write(next_entries)
            return updated

    def delete_entry(self, entry_id: str) -> bool:
        with self._lock:
            entries = self._read()
            remaining = [item for item in entries if item.id != entry_id]
            if len(remaining) == len(entries):
                return False
            self._write(remaining)
            return True

    def delete_all(self) -> None:
        with self._lock:
            self._write([])

    def search(self, query: str, *, limit: int = 6) -> list[MemoryEntry]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        scored = [
            (len(query_terms & _terms(entry.content)), entry)
            for entry in self.list_entries()
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: -item[0])
        return [entry for _, entry in scored[:limit]]

    def _read(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [MemoryEntry(**item) for item in payload if isinstance(item, dict)]

    def _write(self, entries: list[MemoryEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(item) for item in entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def format_memory_context(entries: list[MemoryEntry]) -> str:
    if not entries:
        return ""
    lines = [f"- [{entry.category}] {entry.content}" for entry in entries]
    return (
        "MEMORIA PERSONAL DEL USUARIO (guardada explícitamente por consentimiento; "
        "no fue inferida por el modelo). Usala solo si es relevante para esta consulta, "
        "aclará que proviene de tu memoria guardada y no la mezcles con conocimiento "
        "deportivo general ni con fuentes web:\n" + "\n".join(lines)
    )
