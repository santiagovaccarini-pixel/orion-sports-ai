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


# Providers reached through a plain OpenAI Chat Completions endpoint. They differ
# only by URL, model id and key, so adding one is an entry here, not a code path.
# Each keeps its own key variable so more than one can stay configured and the
# switch (or the rollback) is a single change to ORION_MODEL_PROVIDER.
ENDPOINT_DEFAULTS: dict[str, tuple[str, str, str]] = {
    # provider: (base URL, model id, name of the variable holding the key)
    "cerebras": ("https://api.cerebras.ai/v1", "gpt-oss-120b", "ORION_CEREBRAS_API_KEY"),
    "groq": (
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "ORION_GROQ_API_KEY",
    ),
}

OPENAI_ENDPOINT_PROVIDERS = frozenset(ENDPOINT_DEFAULTS)

# Providers that run the model on someone else's hardware. They share one token
# budget; only Ollama, on the user's own machine, is sized differently.
CLOUD_MODEL_PROVIDERS = OPENAI_ENDPOINT_PROVIDERS | {"cloudflare"}


def _read_provider() -> str:
    # Orion Cloud is the canonical runtime. Ollama remains available only when an
    # explicit legacy/local environment asks for it.
    provider = os.getenv("ORION_MODEL_PROVIDER", "cloudflare").strip().lower()
    if provider not in CLOUD_MODEL_PROVIDERS | {"ollama"}:
        allowed = ", ".join(sorted(CLOUD_MODEL_PROVIDERS | {"ollama"}))
        raise RuntimeError(f"ORION_MODEL_PROVIDER debe ser uno de: {allowed}.")
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
    # Dataclass defaults stay local-test friendly; get_settings() is the canonical
    # runtime path and defaults to Cloudflare above.
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
    # Resolved from the selected provider's defaults; every one is overridable so a
    # new OpenAI-compatible endpoint needs no code change to try.
    endpoint_api_key: str | None = None
    endpoint_base_url: str = "https://api.cerebras.ai/v1"
    endpoint_quick_model: str = "gpt-oss-120b"
    endpoint_deep_model: str = "gpt-oss-120b"
    endpoint_quick_reasoning_effort: str = "low"
    endpoint_deep_reasoning_effort: str = "medium"
    quick_context: int = 4096
    deep_context: int = 8192
    quick_threads: int = 8
    deep_threads: int = 8
    # Ollama compatibility budgets. Cloudflare uses the explicit cloud fields above.
    quick_max_tokens: int = 768
    deep_max_tokens: int = 1536
    quick_history_characters: int = 12_000
    deep_history_characters: int = 30_000
    keep_alive: str = "10m"
    request_timeout_seconds: int = 300
    api_key: str | None = None
    knowledge_path: str = ".orion-runtime/knowledge/documents.json"
    memory_path: str = ".orion-runtime/memory/entries.json"
    # When set, memory and knowledge live in Postgres instead of local JSON
    # files, so they survive restarts and redeploys on an ephemeral disk.
    database_url: str | None = None
    # Memory is user-curated and every entry is shown to the planner, so it is
    # deliberately small: it must stay a set of deliberate facts, not a log.
    memory_max_entries: int = 100
    memory_max_entry_characters: int = 1_000
    web_enabled: bool = True
    web_provider: str = "auto"
    tavily_api_key: str | None = None
    web_minimum_sources: int = 4
    web_allowed_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS
    semantic_orchestration: bool = False
    semantic_max_tool_rounds: int = 3
    semantic_local_context_characters: int = 12_000
    diagnostics_enabled: bool = False
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    # A single chat request fans out into several model calls plus up to
    # semantic_max_tool_rounds web searches and MAX_READ_PAGES page fetches, so
    # unlimited concurrent callers can exhaust the Cloudflare/Tavily quotas and
    # the host itself. These bound that fan-out.
    rate_limit_chat_per_minute: int = 12
    rate_limit_uploads_per_minute: int = 6
    max_concurrent_chats: int = 4
    knowledge_max_documents: int = 200
    knowledge_max_total_characters: int = 20_000_000


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
    # Each endpoint provider keeps its own key variable, so several can stay
    # configured at once and switching between them is one value on the provider.
    endpoint_base_url, endpoint_model, endpoint_key_env = ENDPOINT_DEFAULTS.get(
        provider, ENDPOINT_DEFAULTS["cerebras"]
    )
    endpoint_key = os.getenv(endpoint_key_env) or None
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
        endpoint_api_key=endpoint_key,
        endpoint_base_url=os.getenv("ORION_ENDPOINT_BASE_URL", endpoint_base_url).rstrip(
            "/"
        ),
        endpoint_quick_model=os.getenv("ORION_ENDPOINT_QUICK_MODEL", endpoint_model),
        endpoint_deep_model=os.getenv("ORION_ENDPOINT_DEEP_MODEL", endpoint_model),
        endpoint_quick_reasoning_effort=_read_reasoning_effort(
            "ORION_ENDPOINT_QUICK_REASONING_EFFORT", "low"
        ),
        endpoint_deep_reasoning_effort=_read_reasoning_effort(
            "ORION_ENDPOINT_DEEP_REASONING_EFFORT", "medium"
        ),
        quick_context=_read_positive_int("ORION_QUICK_CONTEXT", 4096),
        deep_context=_read_positive_int("ORION_DEEP_CONTEXT", 8192),
        quick_threads=_read_positive_int("ORION_QUICK_THREADS", 8),
        deep_threads=_read_positive_int("ORION_DEEP_THREADS", 8),
        quick_max_tokens=(
            cloud_quick_max_tokens
            if provider in CLOUD_MODEL_PROVIDERS
            else local_quick_max_tokens
        ),
        deep_max_tokens=(
            cloud_deep_max_tokens
            if provider in CLOUD_MODEL_PROVIDERS
            else local_deep_max_tokens
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
        memory_path=os.getenv(
            "ORION_MEMORY_PATH", ".orion-runtime/memory/entries.json"
        ),
        database_url=os.getenv("DATABASE_URL") or None,
        memory_max_entries=_read_positive_int("ORION_MEMORY_MAX_ENTRIES", 100),
        memory_max_entry_characters=_read_positive_int(
            "ORION_MEMORY_MAX_ENTRY_CHARACTERS", 1_000
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
            "ORION_SEMANTIC_MAX_TOOL_ROUNDS", 3
        ),
        semantic_local_context_characters=_read_positive_int(
            "ORION_SEMANTIC_LOCAL_CONTEXT_CHARACTERS", 12_000
        ),
        diagnostics_enabled=_read_bool("ORION_DIAGNOSTICS_ENABLED", False),
        cors_origins=origins or DEFAULT_CORS_ORIGINS,
        rate_limit_chat_per_minute=_read_positive_int(
            "ORION_RATE_LIMIT_CHAT_PER_MINUTE", 12
        ),
        rate_limit_uploads_per_minute=_read_positive_int(
            "ORION_RATE_LIMIT_UPLOADS_PER_MINUTE", 6
        ),
        max_concurrent_chats=_read_positive_int("ORION_MAX_CONCURRENT_CHATS", 4),
        knowledge_max_documents=_read_positive_int(
            "ORION_KNOWLEDGE_MAX_DOCUMENTS", 200
        ),
        knowledge_max_total_characters=_read_positive_int(
            "ORION_KNOWLEDGE_MAX_TOTAL_CHARACTERS", 20_000_000
        ),
    )
