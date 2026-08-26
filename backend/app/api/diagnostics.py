from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from backend.app.core.config import get_settings
from backend.app.services.diagnostic_trace import diagnostic_traces


router = APIRouter()


def _require_api_key(
    api_key: str | None = Header(default=None, alias="X-Orion-Api-Key"),
) -> None:
    configured_key = get_settings().api_key
    if configured_key is not None and api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_api_key",
                "message": "La clave de Orion no es válida.",
            },
        )


def _ensure_enabled() -> None:
    if not get_settings().diagnostics_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "diagnostics_disabled",
                "message": "El modo diagnóstico de Orion está desactivado.",
            },
        )


@router.get(
    "/diagnostics/traces/latest",
    dependencies=[Depends(_require_api_key)],
)
async def latest_trace() -> dict[str, object]:
    _ensure_enabled()
    trace = diagnostic_traces.latest()
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "diagnostic_trace_not_found",
                "message": "Todavía no hay trazas de diagnóstico en este proceso.",
            },
        )
    return trace.snapshot()


@router.get(
    "/diagnostics/traces/{trace_id}",
    dependencies=[Depends(_require_api_key)],
)
async def diagnostic_trace(trace_id: str) -> dict[str, object]:
    _ensure_enabled()
    trace = diagnostic_traces.get(trace_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "diagnostic_trace_not_found",
                "message": "La traza solicitada ya no está disponible en memoria.",
            },
        )
    return trace.snapshot()
