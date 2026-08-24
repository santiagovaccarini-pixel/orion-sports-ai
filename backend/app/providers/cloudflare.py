from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

import httpx

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage


class CloudflareConfigurationError(RuntimeError):
    pass


class CloudflareUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CloudflareResult:
    content: str
    total_duration_ms: float | None
    load_duration_ms: float | None
    prompt_eval_duration_ms: float | None
    eval_duration_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_second: float | None
    thread_limit: int = 0


@dataclass(frozen=True, slots=True)
class CloudflareStreamEvent:
    content: str
    done: bool
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    thread_limit: int = 0


def select_cloud_history(
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


class CloudflareClient:
    """Cloudflare Workers AI client using its OpenAI-compatible endpoint.

    This provider deliberately has no dependency on a Cloudflare SDK. Keeping the
    transport as plain HTTP makes Orion portable and lets us swap providers later.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timeout = httpx.Timeout(settings.request_timeout_seconds)

    def _credentials(self) -> tuple[str, str]:
        base_url = self.settings.cloudflare_base_url
        token = self.settings.cloudflare_api_token
        if not base_url or not token:
            raise CloudflareConfigurationError(
                "Faltan ORION_CLOUDFLARE_ACCOUNT_ID y/o "
                "ORION_CLOUDFLARE_API_TOKEN."
            )
        return base_url, token

    async def chat(
        self,
        *,
        model: str,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> CloudflareResult:
        parts: list[str] = []
        final_event: CloudflareStreamEvent | None = None
        async for event in self.chat_stream(
            model=model,
            mode=mode,
            messages=messages,
            system_prompt=system_prompt,
        ):
            if event.content:
                parts.append(event.content)
            if event.done:
                final_event = event

        content = "".join(parts).strip()
        if not content:
            raise CloudflareUnavailableError(
                "El proveedor cloud devolvió una respuesta vacía."
            )
        if final_event is None:
            raise CloudflareUnavailableError(
                "El proveedor cloud cerró la respuesta antes de terminar."
            )

        return CloudflareResult(
            content=content,
            total_duration_ms=final_event.total_duration_ms,
            load_duration_ms=None,
            prompt_eval_duration_ms=None,
            eval_duration_ms=final_event.eval_duration_ms,
            prompt_tokens=final_event.prompt_tokens,
            completion_tokens=final_event.completion_tokens,
            tokens_per_second=final_event.tokens_per_second,
        )

    async def chat_stream(
        self,
        *,
        model: str,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> AsyncIterator[CloudflareStreamEvent]:
        base_url, token = self._credentials()
        selected_messages = select_cloud_history(self.settings, mode, messages)
        max_tokens = (
            self.settings.quick_max_tokens
            if mode is SelectedMode.QUICK
            else self.settings.deep_max_tokens
        )
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
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
            "temperature": 0.2 if mode is SelectedMode.QUICK else 0.35,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code in {401, 403}:
                        await response.aread()
                        raise CloudflareConfigurationError(
                            "Cloudflare rechazó las credenciales o el modelo solicitado."
                        )
                    if response.status_code == 429:
                        await response.aread()
                        raise CloudflareUnavailableError(
                            "Se alcanzó temporalmente la cuota gratuita del proveedor cloud."
                        )
                    response.raise_for_status()

                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue

                        data_text = line[5:].strip()
                        if data_text == "[DONE]":
                            break
                        try:
                            data = json.loads(data_text)
                        except ValueError as exc:
                            raise CloudflareUnavailableError(
                                "El proveedor cloud envió streaming inválido."
                            ) from exc

                        usage = data.get("usage") or {}
                        if isinstance(usage.get("prompt_tokens"), int):
                            prompt_tokens = usage["prompt_tokens"]
                        if isinstance(usage.get("completion_tokens"), int):
                            completion_tokens = usage["completion_tokens"]

                        choices = data.get("choices") or []
                        content = ""
                        if choices:
                            delta = choices[0].get("delta") or {}
                            content = str(delta.get("content") or "")
                        if content:
                            yield CloudflareStreamEvent(content=content, done=False)

        except CloudflareConfigurationError:
            raise
        except CloudflareUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise CloudflareUnavailableError(
                "No se pudo completar la respuesta con el proveedor cloud."
            ) from exc

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        tokens_per_second = None
        if completion_tokens and elapsed_ms > 0:
            tokens_per_second = round(completion_tokens / (elapsed_ms / 1000), 2)

        yield CloudflareStreamEvent(
            content="",
            done=True,
            total_duration_ms=elapsed_ms,
            eval_duration_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tokens_per_second,
        )
