from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, status

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
            detail={"code": "invalid_api_key", "message": "La clave de Orion no es válida."},
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


@router.get("/diagnostics/traces/latest", dependencies=[])
async def latest_trace(
    _: None = _require_api_key,
) -> dict[str, object]:
    # FastAPI dependency injection cannot execute a plain default callable here; the
    # explicit call below keeps this router independent from the main chat module.
    raise RuntimeError("dependency placeholder")
