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
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    reasoning_effort: str | None = None
    endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class CloudAIStreamEvent:
    content: str
    done: bool
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    reasoning_effort: str | None = None
    endpoint: str | None = None


TRANSIENT_HTTP_STATUSES = frozenset({502, 503, 504})
TRANSIENT_RETRY_DELAYS_SECONDS = (0.35, 0.9)
STRUCTURED_MIN_MAX_TOKENS = 1536
FINAL_VISIBLE_RESCUE_MIN_MAX_TOKENS = 3072
FINAL_VISIBLE_RESCUE_MAX_MAX_TOKENS = 6144
OUTPUT_LIMIT_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})


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
        settings.cloudflare_quick_reasoning_effort
        if mode is SelectedMode.QUICK
        else settings.cloudflare_deep_reasoning_effort
    )


def _visible_rescue_max_tokens(current: int) -> int:
    return min(
        max(current * 2, FINAL_VISIBLE_RESCUE_MIN_MAX_TOKENS),
        FINAL_VISIBLE_RESCUE_MAX_MAX_TOKENS,
    )


def _is_transient_status(status_code: int) -> bool:
    return status_code in TRANSIENT_HTTP_STATUSES


def _is_gpt_oss(model: str) -> bool:
    return model.startswith("@cf/openai/gpt-oss-")


def _trim_messages(
    messages: list[ChatMessage],
    *,
    max_characters: int,
) -> list[ChatMessage]:
    """Keep recent cloud history inside a bounded transport context."""

    if not messages:
        return []
    remaining = max(1, max_characters)
    selected: list[ChatMessage] = []
    for message in reversed(messages):
        content = message.content
        if len(content) <= remaining:
            selected.append(message)
            remaining -= len(content)
            continue
        if not selected:
            clipped = content[:remaining].strip()
            if clipped:
                selected.append(ChatMessage(role=message.role, content=clipped))
        break
    selected.reverse()
    return selected


def _chat_payload(
    settings: Settings,
    mode: SelectedMode,
    messages: list[ChatMessage],
    system_prompt: str,
    *,
    stream: bool,
    structured: bool = False,
) -> dict[str, object]:
    max_tokens = _max_tokens(settings, mode)
    if structured:
        max_tokens = max(max_tokens, STRUCTURED_MIN_MAX_TOKENS)
    trimmed = _trim_messages(
        messages,
        max_characters=_history_characters(settings, mode),
    )
    payload: dict[str, object] = {
        "model": _selected_model(settings, mode),
        "messages": [
            {"role": "system", "content": system_prompt},
            *[
                {"role": message.role, "content": message.content}
                for message in trimmed
            ],
        ],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": stream,
    }
    if structured:
        payload["response_format"] = {"type": "json_object"}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _responses_payload(
    settings: Settings,
    mode: SelectedMode,
    messages: list[ChatMessage],
    system_prompt: str,
    *,
    stream: bool,
    structured: bool = False,
    reasoning_effort: str | None = None,
) -> dict[str, object]:
    max_output_tokens = _max_tokens(settings, mode)
    if structured:
        max_output_tokens = max(max_output_tokens, STRUCTURED_MIN_MAX_TOKENS)
    effort = _reasoning_effort(settings, mode, reasoning_effort)
    trimmed = _trim_messages(
        messages,
        max_characters=_history_characters(settings, mode),
    )
    payload: dict[str, object] = {
        "model": _selected_model(settings, mode),
        "instructions": system_prompt,
        "input": [
            {"role": message.role, "content": message.content}
            for message in trimmed
        ],
        "reasoning": {"effort": effort},
        "max_output_tokens": max_output_tokens,
        "truncation": "auto",
        "stream": stream,
    }
    # Responses API defaults to temperature=1/top_p=1. Leaving those defaults in
    # place matches the sampling recommendation published with gpt-oss.
    if structured:
        payload["text"] = {"format": {"type": "json_object"}}
    return payload


def _completion_content(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts).strip()
    return ""


def _completion_usage(payload: dict[str, object]) -> tuple[int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return (
        prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens if isinstance(completion_tokens, int) else None,
    )


def _completion_finish_reason(payload: dict[str, object]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    value = first.get("finish_reason")
    return value if isinstance(value, str) and value else None


def _response_object(payload: dict[str, object]) -> dict[str, object]:
    result = payload.get("result")
    return result if isinstance(result, dict) else payload


def _responses_content(payload: dict[str, object]) -> str:
    response = _response_object(payload)
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
    return "".join(parts).strip()


def _responses_usage(
    payload: dict[str, object],
) -> tuple[int | None, int | None, int | None]:
    response = _response_object(payload)
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    reasoning_tokens: int | None = None
    details = usage.get("output_tokens_details")
    if isinstance(details, dict):
        raw_reasoning = details.get("reasoning_tokens")
        if isinstance(raw_reasoning, int):
            reasoning_tokens = raw_reasoning
    return (
        input_tokens if isinstance(input_tokens, int) else None,
        output_tokens if isinstance(output_tokens, int) else None,
        reasoning_tokens,
    )


def _responses_finish_reason(payload: dict[str, object]) -> str | None:
    response = _response_object(payload)
    status = response.get("status")
    if status == "completed":
        return "completed"
    if status == "incomplete":
        details = response.get("incomplete_details")
        if isinstance(details, dict):
            reason = details.get("reason")
            if isinstance(reason, str) and reason:
                return reason
        return "incomplete"
    if status == "failed":
        error = response.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, str) and code:
                return f"failed:{code}"
        return "failed"
    return status if isinstance(status, str) and status else None


def _is_output_limit_reason(reason: str | None) -> bool:
    return bool(reason and reason.lower() in OUTPUT_LIMIT_REASONS)


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
    """Workers AI client optimized for reasoning models.

    GPT-OSS uses Cloudflare's OpenAI-compatible Responses API so Orion can control
    reasoning effort and observe incomplete generations/reasoning token usage. Other
    future cloud models retain a Chat Completions fallback. Hidden reasoning content is
    never surfaced or stored; only token counts and completion status are observed.
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
        structured: bool = False,
        reasoning_effort: str | None = None,
    ) -> CloudAIResult:
        model = _selected_model(self.settings, mode)
        if _is_gpt_oss(model):
            return await self._responses_chat(
                mode=mode,
                messages=messages,
                system_prompt=system_prompt,
                structured=structured,
                reasoning_effort=reasoning_effort,
            )
        return await self._chat_completions_chat(
            mode=mode,
            messages=messages,
            system_prompt=system_prompt,
            structured=structured,
        )

    async def _responses_chat(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        structured: bool,
        reasoning_effort: str | None,
    ) -> CloudAIResult:
        model = _selected_model(self.settings, mode)
        effort = _reasoning_effort(self.settings, mode, reasoning_effort)
        payload = _responses_payload(
            self.settings,
            mode,
            messages,
            system_prompt,
            stream=False,
            structured=structured,
            reasoning_effort=effort,
        )
        transient_failures = 0
        empty_failures = 0
        rescue_used = False

        while True:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/responses",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    raw = response.json()
                if not isinstance(raw, dict):
                    raise CloudAIUnavailableError(
                        "El motor cloud devolvió un formato inesperado."
                    )

                content = _responses_content(raw)
                prompt_tokens, completion_tokens, reasoning_tokens = _responses_usage(raw)
                finish_reason = _responses_finish_reason(raw)

                if _is_output_limit_reason(finish_reason) and not rescue_used:
                    rescue_used = True
                    payload = dict(payload)
                    payload["max_output_tokens"] = _visible_rescue_max_tokens(
                        int(payload["max_output_tokens"])
                    )
                    continue

                if finish_reason not in {None, "completed"}:
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
                        finish_reason=finish_reason or "completed",
                        reasoning_effort=effort,
                        endpoint="responses",
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

    async def _chat_completions_chat(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        structured: bool,
    ) -> CloudAIResult:
        model = _selected_model(self.settings, mode)
        payload = _chat_payload(
            self.settings,
            mode,
            messages,
            system_prompt,
            stream=False,
            structured=structured,
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
                finish_reason = _completion_finish_reason(raw)
                if _is_output_limit_reason(finish_reason) and not rescue_used:
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
                        finish_reason=finish_reason or "stop",
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
        if _is_gpt_oss(model):
            async for event in self._responses_stream(
                mode=mode,
                messages=messages,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
            ):
                yield event
            return
        async for event in self._chat_completions_stream(
            mode=mode,
            messages=messages,
            system_prompt=system_prompt,
        ):
            yield event

    async def _responses_stream(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        reasoning_effort: str | None,
    ) -> AsyncIterator[CloudAIStreamEvent]:
        model = _selected_model(self.settings, mode)
        effort = _reasoning_effort(self.settings, mode, reasoning_effort)
        payload = _responses_payload(
            self.settings,
            mode,
            messages,
            system_prompt,
            stream=True,
            reasoning_effort=effort,
        )
        transient_failures = 0
        rescue_used = False

        while True:
            visible_content_emitted = False
            terminal_response: dict[str, object] | None = None
            terminal_type: str | None = None
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/responses",
                        headers=self.headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            data = parse_sse_data(line)
                            if data is None or data.get("done") is True:
                                continue
                            event_type = data.get("type")
                            if event_type == "response.output_text.delta":
                                delta = data.get("delta")
                                if isinstance(delta, str) and delta:
                                    visible_content_emitted = True
                                    yield CloudAIStreamEvent(
                                        content=delta,
                                        done=False,
                                        model=model,
                                        reasoning_effort=effort,
                                        endpoint="responses",
                                    )
                            elif event_type in {
                                "response.completed",
                                "response.incomplete",
                                "response.failed",
                            }:
                                raw_response = data.get("response")
                                if isinstance(raw_response, dict):
                                    terminal_response = raw_response
                                    terminal_type = str(event_type)

                if terminal_response is None:
                    raise CloudAIUnavailableError(
                        "El modelo cloud cerró el stream antes de informar su estado final."
                    )

                wrapped = {"result": terminal_response}
                prompt_tokens, completion_tokens, reasoning_tokens = _responses_usage(wrapped)
                finish_reason = _responses_finish_reason(wrapped)

                if (
                    _is_output_limit_reason(finish_reason)
                    and not visible_content_emitted
                    and not rescue_used
                ):
                    rescue_used = True
                    payload = dict(payload)
                    payload["max_output_tokens"] = _visible_rescue_max_tokens(
                        int(payload["max_output_tokens"])
                    )
                    continue

                if terminal_type != "response.completed" or finish_reason != "completed":
                    raise CloudAIUnavailableError(
                        "El modelo cloud no completó la respuesta visible "
                        f"(motivo: {finish_reason or terminal_type})."
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
                    finish_reason="completed",
                    reasoning_effort=effort,
                    endpoint="responses",
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

    async def _chat_completions_stream(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> AsyncIterator[CloudAIStreamEvent]:
        model = _selected_model(self.settings, mode)
        payload = _chat_payload(
            self.settings,
            mode,
            messages,
            system_prompt,
            stream=True,
        )
        transient_failures = 0
        rescue_used = False

        while True:
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
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
                            raw_finish = first.get("finish_reason")
                            if isinstance(raw_finish, str) and raw_finish:
                                finish_reason = raw_finish
                            delta = first.get("delta")
                            if not isinstance(delta, dict):
                                continue
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                visible_content_emitted = True
                                yield CloudAIStreamEvent(
                                    content=content,
                                    done=False,
                                    model=model,
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
                    finish_reason=finish_reason or "stop",
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
