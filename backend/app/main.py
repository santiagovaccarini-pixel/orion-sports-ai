from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import get_settings
from backend.app.services.resource_guard import lower_process_priority


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    lower_process_priority()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Núcleo local y privado de Orion.",
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


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": settings.version,
        "privacy": "local-no-persistent-memory",
    }
