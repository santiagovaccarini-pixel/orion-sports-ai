from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from backend.app.services.web_research import DEFAULT_ALLOWED_DOMAINS


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
)
VALID_REASONING_EFFORTS = frozenset({"low", "medium", "high"})


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


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_provider() -> str:
    provider = os.getenv("ORION_MODEL_PROVIDER", "ollama").strip().lower()
    if provider not in {"ollama", "cloudflare"}:
        raise RuntimeError(
            "ORION_MODEL_PROVIDER debe ser 'ollama' o 'cloudflare'."
        )
    return provider


def _read_web_provider() -> str:
    provider = os.getenv("ORION_WEB_PROVIDER", "auto").strip().lower()
    if provider not in {"auto", "tavily", "duckduckgo"}:
        raise RuntimeError(
            "ORION_WEB_PROVIDER debe ser 'auto', 'tavily' o 'duckduckgo'."
        )
    return provider


def _read_reasoning_effort(name: str, default: str) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in VALID_REASONING_EFFORTS:
        allowed = ", ".join(sorted(VALID_REASONING_EFFORTS))
        raise RuntimeError(f"{name} debe ser uno de: {allowed}.")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "Orion Core"
    version: str = "0.4.0-audit-hardening"
    host: str = "127.0.0.1"
    port: int = 8765
    model_provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    quick_model: str = "qwen3:4b-instruct"
    deep_model: str = "qwen3:8b"
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_quick_model: str = "@cf/openai/gpt-oss-120b"
    cloudflare_deep_model: str = "@cf/openai/gpt-oss-120b"
    cloudflare_quick_reasoning_effort: str = "low"
    cloudflare_deep_reasoning_effort: str = "medium"
    cloudflare_quick_max_tokens: int = 1536
    cloudflare_deep_max_tokens: int = 3072
    quick_context: int = 4096
    deep_context: int = 8192
    quick_threads: int = 8
    deep_threads: int = 8
    # Ollama keeps its previous budgets. When settings are loaded for Cloudflare,
    # get_settings maps the independent cloud budget into these transport fields.
    quick_max_tokens: int = 768
    deep_max_tokens: int = 1536
    quick_history_characters: int = 12_000
    deep_history_characters: int = 30_000
    keep_alive: str = "10m"
    request_timeout_seconds: int = 300
    api_key: str | None = None
    knowledge_path: str = ".orion-runtime/knowledge/documents.json"
    web_enabled: bool = True
    web_provider: str = "auto"
    tavily_api_key: str | None = None
    web_minimum_sources: int = 4
    web_allowed_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS
    semantic_orchestration: bool = False
    semantic_max_tool_rounds: int = 2
    semantic_local_context_characters: int = 12_000
    diagnostics_enabled: bool = False
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provider = _read_provider()
    cloud_quick_max_tokens = _read_positive_int(
        "ORION_CLOUDFLARE_QUICK_MAX_TOKENS", 1536
    )
    cloud_deep_max_tokens = _read_positive_int(
        "ORION_CLOUDFLARE_DEEP_MAX_TOKENS", 3072
    )
    local_quick_max_tokens = _read_positive_int("ORION_QUICK_MAX_TOKENS", 768)
    local_deep_max_tokens = _read_positive_int("ORION_DEEP_MAX_TOKENS", 1536)
    origins = tuple(
        item.strip()
        for item in os.getenv("ORION_CORS_ORIGINS", "").split(",")
        if item.strip()
    )
    return Settings(
        host=os.getenv("ORION_HOST", "127.0.0.1"),
        port=_read_int("ORION_PORT", 8765),
        model_provider=provider,
        ollama_base_url=os.getenv(
            "ORION_OLLAMA_URL", "http://127.0.0.1:11434"
        ).rstrip("/"),
        quick_model=os.getenv("ORION_QUICK_MODEL", "qwen3:4b-instruct"),
        deep_model=os.getenv("ORION_DEEP_MODEL", "qwen3:8b"),
        cloudflare_account_id=os.getenv("ORION_CLOUDFLARE_ACCOUNT_ID") or None,
        cloudflare_api_token=os.getenv("ORION_CLOUDFLARE_API_TOKEN") or None,
        cloudflare_quick_model=os.getenv(
            "ORION_CLOUDFLARE_QUICK_MODEL", "@cf/openai/gpt-oss-120b"
        ),
        cloudflare_deep_model=os.getenv(
            "ORION_CLOUDFLARE_DEEP_MODEL", "@cf/openai/gpt-oss-120b"
        ),
        cloudflare_quick_reasoning_effort=_read_reasoning_effort(
            "ORION_CLOUDFLARE_QUICK_REASONING_EFFORT", "low"
        ),
        cloudflare_deep_reasoning_effort=_read_reasoning_effort(
            "ORION_CLOUDFLARE_DEEP_REASONING_EFFORT", "medium"
        ),
        cloudflare_quick_max_tokens=cloud_quick_max_tokens,
        cloudflare_deep_max_tokens=cloud_deep_max_tokens,
        quick_context=_read_positive_int("ORION_QUICK_CONTEXT", 4096),
        deep_context=_read_positive_int("ORION_DEEP_CONTEXT", 8192),
        quick_threads=_read_positive_int("ORION_QUICK_THREADS", 8),
        deep_threads=_read_positive_int("ORION_DEEP_THREADS", 8),
        quick_max_tokens=(
            cloud_quick_max_tokens if provider == "cloudflare" else local_quick_max_tokens
        ),
        deep_max_tokens=(
            cloud_deep_max_tokens if provider == "cloudflare" else local_deep_max_tokens
        ),
        quick_history_characters=_read_positive_int(
            "ORION_QUICK_HISTORY_CHARACTERS", 12_000
        ),
        deep_history_characters=_read_positive_int(
            "ORION_DEEP_HISTORY_CHARACTERS", 30_000
        ),
        keep_alive=os.getenv("ORION_KEEP_ALIVE", "10m"),
        request_timeout_seconds=_read_int("ORION_REQUEST_TIMEOUT", 300),
        api_key=os.getenv("ORION_API_KEY") or None,
        knowledge_path=os.getenv(
            "ORION_KNOWLEDGE_PATH", ".orion-runtime/knowledge/documents.json"
        ),
        web_enabled=_read_bool("ORION_WEB_ENABLED", True),
        web_provider=_read_web_provider(),
        tavily_api_key=os.getenv("ORION_TAVILY_API_KEY") or None,
        web_minimum_sources=_read_positive_int("ORION_WEB_MINIMUM_SOURCES", 4),
        web_allowed_domains=tuple(
            item.strip().lower().removeprefix("www.")
            for item in os.getenv("ORION_WEB_ALLOWED_DOMAINS", "").split(",")
            if item.strip()
        )
        or DEFAULT_ALLOWED_DOMAINS,
        semantic_orchestration=_read_bool("ORION_SEMANTIC_ORCHESTRATION", False),
        semantic_max_tool_rounds=_read_positive_int(
            "ORION_SEMANTIC_MAX_TOOL_ROUNDS", 2
        ),
        semantic_local_context_characters=_read_positive_int(
            "ORION_SEMANTIC_LOCAL_CONTEXT_CHARACTERS", 12_000
        ),
        diagnostics_enabled=_read_bool("ORION_DIAGNOSTICS_ENABLED", False),
        cors_origins=origins or DEFAULT_CORS_ORIGINS,
    )
