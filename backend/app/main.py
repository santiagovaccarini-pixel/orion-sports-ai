from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.diagnostics import router as diagnostics_router
from backend.app.api.routes import router
from backend.app.core.config import get_settings
from backend.app.services.resource_guard import lower_process_priority


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Process-priority protection exists to protect the user's Windows machine while
    # Ollama is doing local inference. It must not throttle a cloud-hosted Core.
    if settings.model_provider == "ollama":
        lower_process_priority()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Núcleo central de Orion con proveedor de IA intercambiable.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Orion-Api-Key"],
)
app.include_router(router, prefix="/api/v1")
app.include_router(diagnostics_router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": settings.version,
        "model_provider": settings.model_provider,
        "persistent_memory": "user_curated",
    }
