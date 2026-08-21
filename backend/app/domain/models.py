from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SelectedMode(str, Enum):
    QUICK = "quick"
    DEEP = "deep"


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    content: str


@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    cpu_percent: float
    memory_available_gb: float
    memory_total_gb: float
    battery_percent: float | None
    plugged_in: bool | None
