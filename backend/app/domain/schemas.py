from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.domain.models import SelectedMode


class RequestedMode(str, Enum):
    AUTO = "auto"
    QUICK = "quick"
    DEEP = "deep"


class SportContext(str, Enum):
    GENERAL = "general"
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    VOLLEYBALL = "volleyball"
    RUGBY = "rugby"
    TENNIS = "tennis"
    ATHLETICS = "athletics"
    SWIMMING = "swimming"
    CYCLING = "cycling"


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
    sport: SportContext = SportContext.FOOTBALL
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
    sport: SportContext
    selected_mode: SelectedMode
    recommended_mode: SelectedMode
    recommendation_reason: str
    model: str
    trace_id: str | None = None
    total_duration_ms: float | None
    load_duration_ms: float | None
    prompt_eval_duration_ms: float | None
    eval_duration_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_second: float | None
    thread_limit: int


class KnowledgeDocumentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=500_000)


class KnowledgeDocumentResponse(BaseModel):
    id: str
    name: str
    characters: int


class MemoryEntryRequest(BaseModel):
    """A fact the user explicitly asks Orion to remember.

    Orion never writes here on its own: memory is only what the user chose to
    save, so it can be reviewed and deleted by the person it describes.
    """

    content: str = Field(min_length=1, max_length=1_000)
    category: str = Field(default="general", min_length=1, max_length=60)


class MemoryEntryResponse(BaseModel):
    id: str
    content: str
    category: str
    created_at: str
    updated_at: str


class StatusResponse(BaseModel):
    service: Literal["online"] = "online"
    version: str
    # Kept in step with config.MODEL_PROVIDERS by a test; a provider missing
    # here makes the whole status route fail validation, not just this field.
    model_provider: Literal["ollama", "cloudflare", "cerebras", "groq"] = "ollama"
    model_provider_online: bool = False
    ollama_online: bool
    installed_models: list[str]
    quick_model: str
    deep_model: str
    quick_threads: int
    deep_threads: int
    loaded_models: list[str]
    snapshot: SystemSnapshotResponse
    memory_enabled: bool = False
    web_enabled: bool = False
    web_minimum_sources: int = 4
