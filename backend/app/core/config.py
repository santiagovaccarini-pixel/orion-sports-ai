from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} debe ser un número entero.") from exc


def _read_positive_int(name: str, default: int) -> int:
    value = _read_int(name, default)
    if value < 1:
        raise RuntimeError(f"{name} debe ser mayor que cero.")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "Orion Local Core"
    version: str = "0.1.1"
    host: str = "127.0.0.1"
    port: int = 8765
    ollama_base_url: str = "http://127.0.0.1:11434"
    quick_model: str = "qwen3:8b"
    deep_model: str = "qwen3:14b"
    quick_context: int = 8192
    deep_context: int = 16384
    quick_threads: int = 6
    deep_threads: int = 8
    keep_alive: str = "2m"
    request_timeout_seconds: int = 300
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = tuple(
        item.strip()
        for item in os.getenv("ORION_CORS_ORIGINS", "").split(",")
        if item.strip()
    )
    return Settings(
        host=os.getenv("ORION_HOST", "127.0.0.1"),
        port=_read_int("ORION_PORT", 8765),
        ollama_base_url=os.getenv(
            "ORION_OLLAMA_URL", "http://127.0.0.1:11434"
        ).rstrip("/"),
        quick_model=os.getenv("ORION_QUICK_MODEL", "qwen3:8b"),
        deep_model=os.getenv("ORION_DEEP_MODEL", "qwen3:14b"),
        quick_context=_read_int("ORION_QUICK_CONTEXT", 8192),
        deep_context=_read_int("ORION_DEEP_CONTEXT", 16384),
        quick_threads=_read_positive_int("ORION_QUICK_THREADS", 6),
        deep_threads=_read_positive_int("ORION_DEEP_THREADS", 8),
        keep_alive=os.getenv("ORION_KEEP_ALIVE", "2m"),
        request_timeout_seconds=_read_int("ORION_REQUEST_TIMEOUT", 300),
        cors_origins=origins or DEFAULT_CORS_ORIGINS,
    )
