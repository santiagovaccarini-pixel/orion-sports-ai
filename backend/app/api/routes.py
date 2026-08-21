from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from backend.app.core.config import get_settings
from backend.app.core.prompt import ORION_SYSTEM_PROMPT
from backend.app.domain.schemas import (
    ChatRequest,
    ChatResponse,
    RequestedMode,
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
from backend.app.services.resource_guard import lower_ollama_priority, read_snapshot
from backend.app.services.resource_policy import evaluate_resources


router = APIRouter()
chat_lock = asyncio.Lock()


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
        snapshot=SystemSnapshotResponse(**asdict(read_snapshot())),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
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

    try:
        async with chat_lock:
            lower_ollama_priority()
            result = await OllamaClient(settings).chat(
                model=model,
                mode=selected_mode,
                messages=request.messages,
                system_prompt=ORION_SYSTEM_PROMPT,
            )
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
        selected_mode=selected_mode,
        recommended_mode=recommendation.mode,
        recommendation_reason=recommendation.reason,
        model=model,
        total_duration_ms=result.total_duration_ms,
        tokens_per_second=result.tokens_per_second,
    )
