from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.domain.models import SelectedMode


class RequestedMode(str, Enum):
    AUTO = "auto"
    QUICK = "quick"
    DEEP = "deep"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("El mensaje no puede estar vacío.")
        return clean


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    mode: RequestedMode = RequestedMode.AUTO
    allow_busy: bool = False

    @model_validator(mode="after")
    def validate_conversation(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("El último mensaje debe pertenecer al usuario.")
        total_characters = sum(len(message.content) for message in self.messages)
        if total_characters > 50_000:
            raise ValueError("La conversación supera el límite temporal del prototipo.")
        return self


class SystemSnapshotResponse(BaseModel):
    cpu_percent: float
    memory_available_gb: float
    memory_total_gb: float
    battery_percent: float | None
    plugged_in: bool | None


class ChatResponse(BaseModel):
    content: str
    selected_mode: SelectedMode
    recommended_mode: SelectedMode
    recommendation_reason: str
    model: str
    total_duration_ms: float | None
    tokens_per_second: float | None
    thread_limit: int


class StatusResponse(BaseModel):
    service: Literal["online"] = "online"
    version: str
    ollama_online: bool
    installed_models: list[str]
    quick_model: str
    deep_model: str
    quick_threads: int
    deep_threads: int
    loaded_models: list[str]
    snapshot: SystemSnapshotResponse
    memory_enabled: Literal[False] = False
