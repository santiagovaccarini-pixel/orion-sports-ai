from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

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
    load_duration_ms: float | None
    prompt_eval_duration_ms: float | None
    eval_duration_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_second: float | None
    thread_limit: int


@dataclass(frozen=True, slots=True)
class OllamaStreamEvent:
    content: str
    done: bool
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    thread_limit: int | None = None


def _nanoseconds_to_milliseconds(value: object) -> float | None:
    if isinstance(value, int):
        return round(value / 1_000_000, 2)
    return None


def parse_stream_payload(
    payload: dict[str, Any],
    *,
    thread_limit: int,
) -> OllamaStreamEvent:
    eval_duration = payload.get("eval_duration")
    completion_tokens = payload.get("eval_count")
    tokens_per_second = None
    if (
        isinstance(eval_duration, int)
        and eval_duration > 0
        and isinstance(completion_tokens, int)
    ):
        tokens_per_second = round(
            completion_tokens / (eval_duration / 1_000_000_000),
            2,
        )

    return OllamaStreamEvent(
        content=str(payload.get("message", {}).get("content", "")),
        done=bool(payload.get("done", False)),
        total_duration_ms=_nanoseconds_to_milliseconds(
            payload.get("total_duration")
        ),
        load_duration_ms=_nanoseconds_to_milliseconds(payload.get("load_duration")),
        prompt_eval_duration_ms=_nanoseconds_to_milliseconds(
            payload.get("prompt_eval_duration")
        ),
        eval_duration_ms=_nanoseconds_to_milliseconds(eval_duration),
        prompt_tokens=(
            payload.get("prompt_eval_count")
            if isinstance(payload.get("prompt_eval_count"), int)
            else None
        ),
        completion_tokens=(
            completion_tokens if isinstance(completion_tokens, int) else None
        ),
        tokens_per_second=tokens_per_second,
        thread_limit=thread_limit,
    )


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
    max_tokens = (
        settings.quick_max_tokens
        if mode is SelectedMode.QUICK
        else settings.deep_max_tokens
    )
    return {
        "num_ctx": context,
        "num_thread": thread_limit,
        "num_predict": max_tokens,
        "temperature": 0.2 if mode is SelectedMode.QUICK else 0.35,
    }


def select_history(
    settings: Settings,
    mode: SelectedMode,
    messages: Sequence[ChatMessage],
) -> list[ChatMessage]:
    character_budget = (
        settings.quick_history_characters
        if mode is SelectedMode.QUICK
        else settings.deep_history_characters
    )
    message_limit = 8 if mode is SelectedMode.QUICK else 16
    selected: list[ChatMessage] = []
    used_characters = 0

    for message in reversed(messages):
        message_size = len(message.content)
        if selected and (
            len(selected) >= message_limit
            or used_characters + message_size > character_budget
        ):
            break
        selected.append(message)
        used_characters += message_size

    selected.reverse()
    while selected and selected[0].role == "assistant":
        selected.pop(0)
    return selected or [messages[-1]]


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timeout = httpx.Timeout(settings.request_timeout_seconds)

    async def status(self) -> OllamaStatus:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                tags_response, ps_response = await asyncio.gather(
                    client.get(f"{self.settings.ollama_base_url}/api/tags"),
                    client.get(f"{self.settings.ollama_base_url}/api/ps"),
                )
                tags_response.raise_for_status()
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

    async def structured_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 384,
    ) -> dict[str, Any]:
        """Run a short, deterministic structured inference for routing/planning.

        This intentionally does not unload another model first. The planner is a
        lightweight pre-pass and the main chat request remains responsible for the
        final resource policy. If planning fails, callers can safely fall back to
        deterministic routing.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": self.settings.keep_alive,
            "options": {
                "num_ctx": min(self.settings.quick_context, 4096),
                "num_thread": self.settings.quick_threads,
                "num_predict": max_tokens,
                "temperature": 0.0,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json=payload,
                )
                if response.status_code == 404:
                    raise ModelNotInstalledError(model)
                response.raise_for_status()
                data = response.json()
        except ModelNotInstalledError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaUnavailableError(
                "No se pudo completar la planificación estructurada con Ollama."
            ) from exc

        content = str(data.get("message", {}).get("content", "")).strip()
        if not content:
            raise OllamaUnavailableError("El planificador devolvió una respuesta vacía.")
        try:
            parsed = json.loads(content)
        except ValueError as exc:
            raise OllamaUnavailableError(
                "El planificador devolvió JSON inválido."
            ) from exc
        if not isinstance(parsed, dict):
            raise OllamaUnavailableError("El planificador no devolvió un objeto JSON.")
        return parsed

    async def chat(
        self,
        *,
        model: str,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> OllamaResult:
        content_parts: list[str] = []
        final_event: OllamaStreamEvent | None = None
        async for event in self.chat_stream(
            model=model,
            mode=mode,
            messages=messages,
            system_prompt=system_prompt,
        ):
            if event.content:
                content_parts.append(event.content)
            if event.done:
                final_event = event

        content = "".join(content_parts).strip()
        if not content:
            raise OllamaUnavailableError("El modelo devolvió una respuesta vacía.")
        if final_event is None:
            raise OllamaUnavailableError("Ollama cerró la respuesta antes de terminar.")

        return OllamaResult(
            content=content,
            total_duration_ms=final_event.total_duration_ms,
            load_duration_ms=final_event.load_duration_ms,
            prompt_eval_duration_ms=final_event.prompt_eval_duration_ms,
            eval_duration_ms=final_event.eval_duration_ms,
            prompt_tokens=final_event.prompt_tokens,
            completion_tokens=final_event.completion_tokens,
            tokens_per_second=final_event.tokens_per_second,
            thread_limit=final_event.thread_limit or self.settings.quick_threads,
        )

    async def chat_stream(
        self,
        *,
        model: str,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> AsyncIterator[OllamaStreamEvent]:
        options = runtime_options(self.settings, mode)
        thread_limit = int(options["num_thread"])
        selected_messages = select_history(self.settings, mode, messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *[
                    {"role": message.role, "content": message.content}
                    for message in selected_messages
                ],
            ],
            "stream": True,
            # Quick stays fast. Deep uses Qwen's internal thinking channel; only the
            # final answer content is streamed to the UI by parse_stream_payload().
            "think": mode is SelectedMode.DEEP,
            "keep_alive": self.settings.keep_alive,
            "options": options,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await self._unload_other_models(client, keep_model=model)
                async with client.stream(
                    "POST",
                    f"{self.settings.ollama_base_url}/api/chat",
                    json=payload,
                ) as response:
                    if response.status_code == 404:
                        await response.aread()
                        raise ModelNotInstalledError(model)
                    response.raise_for_status()

                    saw_done = False
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except ValueError as exc:
                            raise OllamaUnavailableError(
                                "Ollama envió una respuesta progresiva inválida."
                            ) from exc
                        if error_message := data.get("error"):
                            raise OllamaUnavailableError(str(error_message))
                        event = parse_stream_payload(data, thread_limit=thread_limit)
                        saw_done = saw_done or event.done
                        yield event

                    if not saw_done:
                        raise OllamaUnavailableError(
                            "Ollama cerró la respuesta antes de terminar."
                        )
        except ModelNotInstalledError:
            raise
        except OllamaUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                "No se pudo completar la respuesta con Ollama."
            ) from exc

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
