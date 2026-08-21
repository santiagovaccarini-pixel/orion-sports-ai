from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage


class OllamaUnavailableError(RuntimeError):
    pass


class ModelNotInstalledError(RuntimeError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"El modelo {model} todavía no está instalado.")


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    online: bool
    installed_models: tuple[str, ...]
    loaded_models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OllamaResult:
    content: str
    total_duration_ms: float | None
    tokens_per_second: float | None
    thread_limit: int


def runtime_options(settings: Settings, mode: SelectedMode) -> dict[str, int | float]:
    context = (
        settings.quick_context
        if mode is SelectedMode.QUICK
        else settings.deep_context
    )
    thread_limit = (
        settings.quick_threads
        if mode is SelectedMode.QUICK
        else settings.deep_threads
    )
    return {
        "num_ctx": context,
        "num_thread": thread_limit,
        "temperature": 0.2 if mode is SelectedMode.QUICK else 0.35,
    }


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timeout = httpx.Timeout(settings.request_timeout_seconds)

    async def status(self) -> OllamaStatus:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                tags_response = await client.get(
                    f"{self.settings.ollama_base_url}/api/tags"
                )
                tags_response.raise_for_status()
                ps_response = await client.get(f"{self.settings.ollama_base_url}/api/ps")
                ps_response.raise_for_status()
                tags_data = tags_response.json()
                ps_data = ps_response.json()
        except (httpx.HTTPError, ValueError):
            return OllamaStatus(False, (), ())

        installed = tuple(
            item.get("name", "")
            for item in tags_data.get("models", [])
            if item.get("name")
        )
        loaded = tuple(
            item.get("name", "")
            for item in ps_data.get("models", [])
            if item.get("name")
        )
        return OllamaStatus(True, installed, loaded)

    async def chat(
        self,
        *,
        model: str,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> OllamaResult:
        options = runtime_options(self.settings, mode)
        thread_limit = int(options["num_thread"])
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *[
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
            ],
            "stream": False,
            "think": mode is SelectedMode.DEEP,
            "keep_alive": self.settings.keep_alive,
            "options": options,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await self._unload_other_models(client, keep_model=model)
                response = await client.post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                "No se pudo conectar con Ollama en esta computadora."
            ) from exc

        if response.status_code == 404:
            raise ModelNotInstalledError(model)
        try:
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaUnavailableError(
                "Ollama respondió con un error inesperado."
            ) from exc

        content = str(data.get("message", {}).get("content", "")).strip()
        if not content:
            raise OllamaUnavailableError("El modelo devolvió una respuesta vacía.")

        total_duration = data.get("total_duration")
        eval_duration = data.get("eval_duration")
        eval_count = data.get("eval_count")
        tokens_per_second = None
        if isinstance(eval_duration, int) and eval_duration > 0 and isinstance(eval_count, int):
            tokens_per_second = round(eval_count / (eval_duration / 1_000_000_000), 2)

        return OllamaResult(
            content=content,
            total_duration_ms=(
                round(total_duration / 1_000_000, 2)
                if isinstance(total_duration, int)
                else None
            ),
            tokens_per_second=tokens_per_second,
            thread_limit=thread_limit,
        )

    async def _unload_other_models(
        self,
        client: httpx.AsyncClient,
        *,
        keep_model: str,
    ) -> None:
        try:
            response = await client.get(f"{self.settings.ollama_base_url}/api/ps")
            response.raise_for_status()
            loaded_models = [
                item.get("name")
                for item in response.json().get("models", [])
                if item.get("name") and item.get("name") != keep_model
            ]
            for loaded_model in loaded_models:
                await client.post(
                    f"{self.settings.ollama_base_url}/api/generate",
                    json={"model": loaded_model, "keep_alive": 0},
                )
        except (httpx.HTTPError, ValueError):
            # The main chat request will still provide the authoritative error.
            return
