from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A fact explicitly saved by the user.

    This store never infers or auto-saves information from a chat message. Retrieval
    semantics are intentionally outside this module: Orion must not decide relevance
    through keyword overlap or phrase similarity.
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

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        with self._lock:
            return next((entry for entry in self._read() if entry.id == entry_id), None)

    def add_entry(self, entry_id: str, content: str, category: str) -> MemoryEntry:
        now = datetime.now(timezone.utc).isoformat()
        entry = MemoryEntry(entry_id, content.strip(), category.strip(), now, now)
        with self._lock:
            entries = [item for item in self._read() if item.id != entry_id]
            entries.append(entry)
            self._write(entries)
        return entry

    def update_entry(
        self,
        entry_id: str,
        content: str,
        category: str | None = None,
    ) -> MemoryEntry | None:
        with self._lock:
            entries = self._read()
            updated: MemoryEntry | None = None
            next_entries: list[MemoryEntry] = []
            for item in entries:
                if item.id != entry_id:
                    next_entries.append(item)
                    continue
                updated = MemoryEntry(
                    id=item.id,
                    content=content.strip(),
                    category=(category.strip() if category is not None else item.category),
                    created_at=item.created_at,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                next_entries.append(updated)
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

    def _read(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        entries: list[MemoryEntry] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(MemoryEntry(**item))
            except TypeError:
                continue
        return entries

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
        "no inferida por el modelo). Usala solo si el plan semántico determinó que es "
        "relevante para la consulta. No la mezcles con conocimiento colectivo ni con "
        "fuentes web:\n" + "\n".join(lines)
    )
