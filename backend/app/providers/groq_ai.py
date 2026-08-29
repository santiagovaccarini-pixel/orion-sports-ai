"""Groq client for Orion's reasoning models.

Groq serves the same gpt-oss weights Orion already runs on Cloudflare, over the
OpenAI Chat Completions dialect, so the payload builder, the parsers and the
retry policy are shared with every other cloud provider through
`openai_compatible`. What is Groq-specific lives here: the endpoint, the
credentials, the model ids, and the fact that reasoning effort travels inside the
Chat Completions body instead of a separate Responses API.

Hidden reasoning is never surfaced or stored. Only the token count is read, from
`usage.completion_tokens_details`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage
from backend.app.providers.openai_compatible import (
    TRANSIENT_RETRY_DELAYS_SECONDS,
    CloudAIConfigurationError,
    CloudAIResult,
    CloudAIStreamEvent,
    CloudAIUnavailableError,
    _completion_content,
    _completion_finish_reason,
    _completion_reasoning_tokens,
    _completion_usage,
    _delta_content,
    _is_output_limit_reason,
    _is_transient_status,
    _visible_rescue_max_tokens,
    build_chat_payload,
    parse_sse_data,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def _selected_model(settings: Settings, mode: SelectedMode) -> str:
    return (
        settings.groq_quick_model
        if mode is SelectedMode.QUICK
        else settings.groq_deep_model
    )


def _max_tokens(settings: Settings, mode: SelectedMode) -> int:
    return (
        settings.quick_max_tokens
        if mode is SelectedMode.QUICK
        else settings.deep_max_tokens
    )


def _history_characters(settings: Settings, mode: SelectedMode) -> int:
    return (
        settings.quick_history_characters
        if mode is SelectedMode.QUICK
        else settings.deep_history_characters
    )


def _reasoning_effort(
    settings: Settings,
    mode: SelectedMode,
    override: str | None = None,
) -> str:
    if override:
        return override
    return (
        settings.groq_quick_reasoning_effort
        if mode is SelectedMode.QUICK
        else settings.groq_deep_reasoning_effort
    )


class GroqAIClient:
    """Talks to Groq's OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.groq_api_key:
            raise CloudAIConfigurationError("Falta ORION_GROQ_API_KEY.")
        self.base_url = (settings.groq_base_url or DEFAULT_BASE_URL).rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = httpx.Timeout(settings.request_timeout_seconds)

    def _payload(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        stream: bool,
        structured: bool,
        reasoning_effort: str | None,
    ) -> dict[str, object]:
        return build_chat_payload(
            model=_selected_model(self.settings, mode),
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=_max_tokens(self.settings, mode),
            history_characters=_history_characters(self.settings, mode),
            stream=stream,
            structured=structured,
            reasoning_effort=_reasoning_effort(self.settings, mode, reasoning_effort),
        )

    async def chat(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        structured: bool = False,
        reasoning_effort: str | None = None,
    ) -> CloudAIResult:
        model = _selected_model(self.settings, mode)
        effort = _reasoning_effort(self.settings, mode, reasoning_effort)
        payload = self._payload(
            mode=mode,
            messages=messages,
            system_prompt=system_prompt,
            stream=False,
            structured=structured,
            reasoning_effort=reasoning_effort,
        )
        transient_failures = 0
        empty_failures = 0
        rescue_used = False

        while True:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    raw = response.json()
                if not isinstance(raw, dict):
                    raise CloudAIUnavailableError(
                        "El motor cloud devolvió un formato inesperado."
                    )

                content = _completion_content(raw)
                prompt_tokens, completion_tokens = _completion_usage(raw)
                reasoning_tokens = _completion_reasoning_tokens(raw)
                finish_reason = _completion_finish_reason(raw)

                if _is_output_limit_reason(finish_reason) and not rescue_used:
                    # The visible answer was cut off by the token ceiling, not by the
                    # model finishing: retry once with more room before giving up.
                    rescue_used = True
                    payload = dict(payload)
                    payload["max_tokens"] = _visible_rescue_max_tokens(
                        int(payload["max_tokens"])
                    )
                    continue

                if finish_reason and finish_reason not in {"stop", "completed"}:
                    raise CloudAIUnavailableError(
                        "El modelo cloud no completó la generación "
                        f"(motivo: {finish_reason})."
                    )

                if content:
                    return CloudAIResult(
                        content=content,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        reasoning_tokens=reasoning_tokens,
                        finish_reason=finish_reason or "stop",
                        reasoning_effort=effort,
                        endpoint="chat_completions",
                    )

                if empty_failures < len(TRANSIENT_RETRY_DELAYS_SECONDS):
                    delay = TRANSIENT_RETRY_DELAYS_SECONDS[empty_failures]
                    empty_failures += 1
                    await asyncio.sleep(delay)
                    continue
                raise CloudAIUnavailableError(
                    "El modelo cloud devolvió una respuesta vacía en una etapa interna."
                )

            except ValueError as exc:
                raise CloudAIUnavailableError(
                    "El motor cloud devolvió una respuesta que no es JSON válido."
                ) from exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if (
                    _is_transient_status(status_code)
                    and transient_failures < len(TRANSIENT_RETRY_DELAYS_SECONDS)
                ):
                    delay = TRANSIENT_RETRY_DELAYS_SECONDS[transient_failures]
                    transient_failures += 1
                    await asyncio.sleep(delay)
                    continue
                self._raise_status_error(exc)
            except httpx.HTTPError as exc:
                raise CloudAIUnavailableError(
                    "No se pudo conectar con el motor cloud."
                ) from exc

    async def chat_stream(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[CloudAIStreamEvent]:
        model = _selected_model(self.settings, mode)
        effort = _reasoning_effort(self.settings, mode, reasoning_effort)
        payload = self._payload(
            mode=mode,
            messages=messages,
            system_prompt=system_prompt,
            stream=True,
            structured=False,
            reasoning_effort=reasoning_effort,
        )
        transient_failures = 0
        rescue_used = False

        while True:
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            reasoning_tokens: int | None = None
            visible_content_emitted = False
            finish_reason: str | None = None
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
                            if data is None or data.get("done") is True:
                                continue
                            usage = data.get("usage")
                            if isinstance(usage, dict):
                                chunk_prompt, chunk_completion = _completion_usage(data)
                                if chunk_prompt is not None:
                                    prompt_tokens = chunk_prompt
                                if chunk_completion is not None:
                                    completion_tokens = chunk_completion
                                chunk_reasoning = _completion_reasoning_tokens(data)
                                if chunk_reasoning is not None:
                                    reasoning_tokens = chunk_reasoning
                            chunk_finish = _completion_finish_reason(data)
                            if chunk_finish:
                                finish_reason = chunk_finish
                            # Only `delta.content` is read, so the chunks that carry
                            # hidden reasoning never reach the transcript.
                            content = _delta_content(data)
                            if content:
                                visible_content_emitted = True
                                yield CloudAIStreamEvent(
                                    content=content,
                                    done=False,
                                    model=model,
                                    reasoning_effort=effort,
                                    endpoint="chat_completions",
                                )

                if (
                    _is_output_limit_reason(finish_reason)
                    and not visible_content_emitted
                    and not rescue_used
                ):
                    rescue_used = True
                    payload = dict(payload)
                    payload["max_tokens"] = _visible_rescue_max_tokens(
                        int(payload["max_tokens"])
                    )
                    continue
                if finish_reason and finish_reason not in {"stop", "completed"}:
                    raise CloudAIUnavailableError(
                        "El modelo cloud no completó la respuesta visible "
                        f"(motivo: {finish_reason})."
                    )
                if not visible_content_emitted:
                    raise CloudAIUnavailableError(
                        "El modelo cloud terminó sin producir una respuesta visible."
                    )
                yield CloudAIStreamEvent(
                    content="",
                    done=True,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    finish_reason=finish_reason or "stop",
                    reasoning_effort=effort,
                    endpoint="chat_completions",
                )
                return

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if (
                    _is_transient_status(status_code)
                    and transient_failures < len(TRANSIENT_RETRY_DELAYS_SECONDS)
                    and not visible_content_emitted
                ):
                    delay = TRANSIENT_RETRY_DELAYS_SECONDS[transient_failures]
                    transient_failures += 1
                    await asyncio.sleep(delay)
                    continue
                self._raise_status_error(exc)
            except httpx.HTTPError as exc:
                raise CloudAIUnavailableError(
                    "No se pudo conectar con el motor cloud."
                ) from exc

    @staticmethod
    def _raise_status_error(exc: httpx.HTTPStatusError) -> None:
        detail = ""
        try:
            detail = exc.response.text[:500]
        except Exception:
            detail = ""
        status_code = exc.response.status_code
        if status_code in {401, 403}:
            raise CloudAIConfigurationError(
                "Groq rechazó la clave configurada para Orion."
            ) from exc
        if status_code == 429:
            # On Groq's free plan this fires on almost every Orion query, because a
            # single review turn is larger than the free per-minute allowance.
            raise CloudAIUnavailableError(
                "Groq frenó la consulta por límite de uso. Si la cuenta sigue en el "
                "plan gratuito, ese límite es más chico que una consulta de Orion."
            ) from exc
        if status_code == 404:
            raise CloudAIConfigurationError(
                "Groq no reconoce el modelo configurado para Orion."
            ) from exc
        raise CloudAIUnavailableError(
            f"El motor cloud respondió con error {status_code}. {detail}".strip()
        ) from exc
