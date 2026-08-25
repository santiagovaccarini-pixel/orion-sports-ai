from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage


class CloudAIUnavailableError(RuntimeError):
    pass


class CloudAIConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CloudAIResult:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class CloudAIStreamEvent:
    content: str
    done: bool
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


TRANSIENT_HTTP_STATUSES = frozenset({502, 503, 504})
TRANSIENT_RETRY_DELAYS_SECONDS = (0.35, 0.9)


def _selected_model(settings: Settings, mode: SelectedMode) -> str:
    return (
        settings.cloudflare_quick_model
        if mode is SelectedMode.QUICK
        else settings.cloudflare_deep_model
    )


def _max_tokens(settings: Settings, mode: SelectedMode) -> int:
    return (
        settings.quick_max_tokens
        if mode is SelectedMode.QUICK
        else settings.deep_max_tokens
    )


def _is_transient_status(status_code: int) -> bool:
    return status_code in TRANSIENT_HTTP_STATUSES


def parse_sse_data(line: str) -> dict[str, object] | None:
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        return {"done": True}
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise CloudAIUnavailableError(
            "El proveedor cloud devolvió un fragmento inválido."
        ) from exc
    if not isinstance(data, dict):
        raise CloudAIUnavailableError(
            "El proveedor cloud devolvió un formato inesperado."
        )
    return data


class CloudflareAIClient:
    """Cliente mínimo para Workers AI usando su API compatible con OpenAI.

    No contiene lógica de Orion: solamente traduce mensajes hacia/desde el
    proveedor. Esto permite cambiar de motor sin reescribir el orquestador.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.cloudflare_account_id or not settings.cloudflare_api_token:
            raise CloudAIConfigurationError(
                "Faltan ORION_CLOUDFLARE_ACCOUNT_ID y ORION_CLOUDFLARE_API_TOKEN."
            )
        self.base_url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{settings.cloudflare_account_id}/ai/v1"
        )
        self.headers = {
            "Authorization": f"Bearer {settings.cloudflare_api_token}",
            "Content-Type": "application/json",
        }
        self.timeout = httpx.Timeout(settings.request_timeout_seconds)

    async def chat(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> CloudAIResult:
        parts: list[str] = []
        final_event: CloudAIStreamEvent | None = None
        async for event in self.chat_stream(
            mode=mode,
            messages=messages,
            system_prompt=system_prompt,
        ):
            if event.content:
                parts.append(event.content)
            if event.done:
                final_event = event

        content = "".join(parts).strip()
        if not content or final_event is None:
            raise CloudAIUnavailableError(
                "El modelo cloud cerró la respuesta antes de terminar."
            )
        return CloudAIResult(
            content=content,
            model=final_event.model,
            prompt_tokens=final_event.prompt_tokens,
            completion_tokens=final_event.completion_tokens,
        )

    async def chat_stream(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> AsyncIterator[CloudAIStreamEvent]:
        model = _selected_model(self.settings, mode)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *[
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
            ],
            "max_tokens": _max_tokens(self.settings, mode),
            "temperature": 0.2 if mode is SelectedMode.QUICK else 0.35,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        last_status_error: httpx.HTTPStatusError | None = None
        attempt_count = len(TRANSIENT_RETRY_DELAYS_SECONDS) + 1

        for attempt in range(attempt_count):
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            saw_done = False

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            data = parse_sse_data(line)
                            if data is None:
                                continue
                            if data.get("done") is True:
                                saw_done = True
                                yield CloudAIStreamEvent(
                                    content="",
                                    done=True,
                                    model=model,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                )
                                continue

                            usage = data.get("usage")
                            if isinstance(usage, dict):
                                raw_prompt = usage.get("prompt_tokens")
                                raw_completion = usage.get("completion_tokens")
                                if isinstance(raw_prompt, int):
                                    prompt_tokens = raw_prompt
                                if isinstance(raw_completion, int):
                                    completion_tokens = raw_completion

                            choices = data.get("choices")
                            if not isinstance(choices, list) or not choices:
                                continue
                            first = choices[0]
                            if not isinstance(first, dict):
                                continue
                            delta = first.get("delta")
                            if not isinstance(delta, dict):
                                continue
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                yield CloudAIStreamEvent(
                                    content=content,
                                    done=False,
                                    model=model,
                                )

                if not saw_done:
                    yield CloudAIStreamEvent(
                        content="",
                        done=True,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                return

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if (
                    _is_transient_status(status_code)
                    and attempt < len(TRANSIENT_RETRY_DELAYS_SECONDS)
                ):
                    last_status_error = exc
                    await asyncio.sleep(TRANSIENT_RETRY_DELAYS_SECONDS[attempt])
                    continue
                self._raise_status_error(exc)
            except httpx.HTTPError as exc:
                raise CloudAIUnavailableError(
                    "No se pudo conectar con el motor cloud."
                ) from exc

        if last_status_error is not None:  # pragma: no cover - defensive safeguard
            self._raise_status_error(last_status_error)

    @staticmethod
    def _raise_status_error(exc: httpx.HTTPStatusError) -> None:
        detail = ""
        try:
            detail = exc.response.text[:500]
        except Exception:
            detail = ""
        if exc.response.status_code in {401, 403}:
            raise CloudAIConfigurationError(
                "Cloudflare rechazó las credenciales configuradas para Orion."
            ) from exc
        if exc.response.status_code == 429:
            raise CloudAIUnavailableError(
                "Orion alcanzó el cupo gratuito o el límite temporal del motor cloud."
            ) from exc
        raise CloudAIUnavailableError(
            f"El motor cloud respondió con error {exc.response.status_code}. {detail}".strip()
        ) from exc
