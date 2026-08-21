from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.app.core.config import get_settings
from backend.app.core.prompt import build_system_prompt
from backend.app.domain.schemas import (
    ChatRequest,
    ChatResponse,
    RequestedMode,
    SportContext,
    StatusResponse,
    SystemSnapshotResponse,
)
from backend.app.domain.models import SelectedMode
from backend.app.providers.ollama import (
    ModelNotInstalledError,
    OllamaClient,
    OllamaUnavailableError,
)
from backend.app.services.mode_router import recommend_mode
from backend.app.services.resource_guard import (
    lower_ollama_priority,
    maintain_ollama_priority,
    read_snapshot,
)
from backend.app.services.resource_policy import evaluate_resources


router = APIRouter()
chat_lock = asyncio.Lock()


@dataclass(frozen=True, slots=True)
class PreparedChat:
    selected_mode: SelectedMode
    recommended_mode: SelectedMode
    recommendation_reason: str
    model: str
    sport: SportContext


def _ndjson(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _model_is_installed(model: str, installed_models: tuple[str, ...]) -> bool:
    return any(
        installed == model
        or installed.startswith(f"{model}:")
        or model.startswith(f"{installed}:")
        for installed in installed_models
    )


async def _prepare_chat(request: ChatRequest, *, preflight_model: bool) -> PreparedChat:
    settings = get_settings()
    recommendation = recommend_mode(request.messages)
    selected_mode = (
        recommendation.mode
        if request.mode is RequestedMode.AUTO
        else SelectedMode(request.mode.value)
    )
    snapshot = read_snapshot()
    resource_decision = evaluate_resources(selected_mode, snapshot)

    if resource_decision.requires_confirmation and not request.allow_busy:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "resource_confirmation_required",
                "message": (
                    "Orion recomienda esperar o utilizar el modo Rápido para no afectar "
                    "las demás aplicaciones."
                ),
                "reasons": list(resource_decision.reasons),
                "selected_mode": selected_mode.value,
                "recommended_mode": recommendation.mode.value,
                "snapshot": asdict(snapshot),
            },
        )

    model = (
        settings.quick_model
        if selected_mode is SelectedMode.QUICK
        else settings.deep_model
    )
    if preflight_model:
        ollama = await OllamaClient(settings).status()
        if not ollama.online:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "ollama_unavailable",
                    "message": "No se pudo conectar con Ollama en esta computadora.",
                },
            )
        if not _model_is_installed(model, ollama.installed_models):
            raise HTTPException(
                status_code=status.HTTP_424_FAILED_DEPENDENCY,
                detail={
                    "code": "model_not_installed",
                    "message": (
                        f"Falta instalar {model}. Orion no descargará modelos sin tu autorización."
                    ),
                    "model": model,
                },
            )

    return PreparedChat(
        selected_mode=selected_mode,
        recommended_mode=recommendation.mode,
        recommendation_reason=recommendation.reason,
        model=model,
        sport=request.sport,
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status", response_model=StatusResponse)
async def system_status() -> StatusResponse:
    settings = get_settings()
    ollama = await OllamaClient(settings).status()
    return StatusResponse(
        version=settings.version,
        ollama_online=ollama.online,
        installed_models=list(ollama.installed_models),
        loaded_models=list(ollama.loaded_models),
        quick_model=settings.quick_model,
        deep_model=settings.deep_model,
        quick_threads=settings.quick_threads,
        deep_threads=settings.deep_threads,
        snapshot=SystemSnapshotResponse(**asdict(read_snapshot())),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    prepared = await _prepare_chat(request, preflight_model=False)

    try:
        async with chat_lock:
            lower_ollama_priority()
            priority_stop = asyncio.Event()
            priority_task = asyncio.create_task(
                maintain_ollama_priority(priority_stop)
            )
            try:
                result = await OllamaClient(settings).chat(
                    model=prepared.model,
                    mode=prepared.selected_mode,
                    messages=request.messages,
                    system_prompt=build_system_prompt(
                        prepared.sport, prepared.selected_mode
                    ),
                )
            finally:
                priority_stop.set()
                await priority_task
    except ModelNotInstalledError as exc:
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail={
                "code": "model_not_installed",
                "message": f"Falta instalar {exc.model}. Orion no descargará modelos sin tu autorización.",
                "model": exc.model,
            },
        ) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "ollama_unavailable", "message": str(exc)},
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
        thread_limit=result.thread_limit,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    settings = get_settings()
    prepared = await _prepare_chat(request, preflight_model=True)

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
            async with chat_lock:
                lower_ollama_priority()
                priority_stop = asyncio.Event()
                priority_task = asyncio.create_task(
                    maintain_ollama_priority(priority_stop)
                )
                try:
                    async for event in OllamaClient(settings).chat_stream(
                        model=prepared.model,
                        mode=prepared.selected_mode,
                        messages=request.messages,
                        system_prompt=build_system_prompt(
                            prepared.sport, prepared.selected_mode
                        ),
                    ):
                        if event.content:
                            yield _ndjson(
                                {"type": "content", "content": event.content}
                            )
                        if event.done:
                            yield _ndjson(
                                {
                                    "type": "done",
                                    "total_duration_ms": event.total_duration_ms,
                                    "load_duration_ms": event.load_duration_ms,
                                    "prompt_eval_duration_ms": (
                                        event.prompt_eval_duration_ms
                                    ),
                                    "eval_duration_ms": event.eval_duration_ms,
                                    "prompt_tokens": event.prompt_tokens,
                                    "completion_tokens": event.completion_tokens,
                                    "tokens_per_second": event.tokens_per_second,
                                    "thread_limit": event.thread_limit,
                                }
                            )
                finally:
                    priority_stop.set()
                    await priority_task
        except asyncio.CancelledError:
            raise
        except ModelNotInstalledError as exc:
            yield _ndjson(
                {
                    "type": "error",
                    "code": "model_not_installed",
                    "message": f"Falta instalar {exc.model}.",
                }
            )
        except OllamaUnavailableError as exc:
            yield _ndjson(
                {
                    "type": "error",
                    "code": "ollama_unavailable",
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
