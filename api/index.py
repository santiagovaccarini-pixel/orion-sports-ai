from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.app.core.config import get_settings
from backend.app.core.prompt import build_system_prompt
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatRequest, ChatResponse, RequestedMode, SportContext
from backend.app.providers.cloudflare import (
    CloudflareClient,
    CloudflareConfigurationError,
    CloudflareUnavailableError,
)
from backend.app.services.mode_router import recommend_mode


settings = get_settings()

app = FastAPI(
    title="Orion Cloud Core",
    version=settings.version,
    description="Núcleo cloud experimental de Orion.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Orion-Api-Key"],
)


@dataclass(frozen=True, slots=True)
class PreparedCloudChat:
    selected_mode: SelectedMode
    recommended_mode: SelectedMode
    recommendation_reason: str
    model: str
    sport: SportContext


def require_api_key(
    api_key: str | None = Header(default=None, alias="X-Orion-Api-Key"),
) -> None:
    configured_key = settings.api_key
    if configured_key is not None and api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_api_key", "message": "La clave de Orion no es válida."},
        )


def _ndjson(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _prepare_chat(request: ChatRequest) -> PreparedCloudChat:
    recommendation = recommend_mode(request.messages)
    selected_mode = (
        recommendation.mode
        if request.mode is RequestedMode.AUTO
        else SelectedMode(request.mode.value)
    )
    model = (
        settings.cloudflare_quick_model
        if selected_mode is SelectedMode.QUICK
        else settings.cloudflare_deep_model
    )
    return PreparedCloudChat(
        selected_mode=selected_mode,
        recommended_mode=recommendation.mode,
        recommendation_reason=recommendation.reason,
        model=model,
        sport=request.sport,
    )


def _system_prompt(request: ChatRequest, prepared: PreparedCloudChat) -> str:
    return build_system_prompt(
        prepared.sport,
        prepared.selected_mode,
        request.messages[-1].content,
    )


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "orion-cloud-core",
        "provider": "cloudflare",
    }


@app.get("/api/v1/status", dependencies=[Depends(require_api_key)])
async def system_status() -> dict[str, object]:
    configured = bool(settings.cloudflare_base_url and settings.cloudflare_api_token)
    return {
        "service": "online",
        "version": settings.version,
        "provider": "cloudflare",
        "provider_configured": configured,
        "ollama_online": False,
        "installed_models": [],
        "loaded_models": [],
        "quick_model": settings.cloudflare_quick_model,
        "deep_model": settings.cloudflare_deep_model,
        "quick_threads": 0,
        "deep_threads": 0,
        "snapshot": {
            "cpu_percent": 0.0,
            "memory_available_gb": 0.0,
            "memory_total_gb": 0.0,
            "battery_percent": None,
            "plugged_in": None,
        },
        "memory_enabled": False,
        "web_enabled": False,
        "web_minimum_sources": 0,
    }


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
async def chat(request: ChatRequest) -> ChatResponse:
    prepared = _prepare_chat(request)
    try:
        result = await CloudflareClient(settings).chat(
            model=prepared.model,
            mode=prepared.selected_mode,
            messages=request.messages,
            system_prompt=_system_prompt(request, prepared),
        )
    except CloudflareConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "cloud_not_configured", "message": str(exc)},
        ) from exc
    except CloudflareUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "cloud_unavailable", "message": str(exc)},
        ) from exc

    return ChatResponse(
        content=result.content,
        sport=prepared.sport,
        selected_mode=prepared.selected_mode,
        recommended_mode=prepared.recommended_mode,
        recommendation_reason=prepared.recommendation_reason,
        model=prepared.model,
        total_duration_ms=result.total_duration_ms,
        load_duration_ms=result.load_duration_ms,
        prompt_eval_duration_ms=result.prompt_eval_duration_ms,
        eval_duration_ms=result.eval_duration_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        tokens_per_second=result.tokens_per_second,
        thread_limit=0,
    )


@app.post("/api/v1/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    prepared = _prepare_chat(request)

    async def generate() -> AsyncIterator[bytes]:
        yield _ndjson(
            {
                "type": "meta",
                "selected_mode": prepared.selected_mode.value,
                "recommended_mode": prepared.recommended_mode.value,
                "recommendation_reason": prepared.recommendation_reason,
                "model": prepared.model,
                "sport": prepared.sport.value,
            }
        )
        try:
            async for event in CloudflareClient(settings).chat_stream(
                model=prepared.model,
                mode=prepared.selected_mode,
                messages=request.messages,
                system_prompt=_system_prompt(request, prepared),
            ):
                if event.content:
                    yield _ndjson({"type": "content", "content": event.content})
                if event.done:
                    yield _ndjson(
                        {
                            "type": "done",
                            "total_duration_ms": event.total_duration_ms,
                            "load_duration_ms": event.load_duration_ms,
                            "prompt_eval_duration_ms": event.prompt_eval_duration_ms,
                            "eval_duration_ms": event.eval_duration_ms,
                            "prompt_tokens": event.prompt_tokens,
                            "completion_tokens": event.completion_tokens,
                            "tokens_per_second": event.tokens_per_second,
                            "thread_limit": 0,
                        }
                    )
        except CloudflareConfigurationError as exc:
            yield _ndjson(
                {
                    "type": "error",
                    "code": "cloud_not_configured",
                    "message": str(exc),
                }
            )
        except CloudflareUnavailableError as exc:
            yield _ndjson(
                {
                    "type": "error",
                    "code": "cloud_unavailable",
                    "message": str(exc),
                }
            )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
