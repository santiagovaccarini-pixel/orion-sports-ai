"""Pieces shared by every provider that speaks the OpenAI Chat Completions dialect.

Cloudflare Workers AI and Groq both expose the same wire format, so the payload
builder, the response parsers and the retry policy live here instead of being
copied per provider. Anything that reads provider-specific settings (which model,
which credentials, which reasoning knob) stays in that provider's own module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

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


def _visible_rescue_max_tokens(current: int) -> int:
    return min(
        max(current * 2, FINAL_VISIBLE_RESCUE_MIN_MAX_TOKENS),
        FINAL_VISIBLE_RESCUE_MAX_MAX_TOKENS,
    )


def _is_transient_status(status_code: int) -> bool:
    return status_code in TRANSIENT_HTTP_STATUSES


def _is_output_limit_reason(reason: str | None) -> bool:
    return bool(reason and reason.lower() in OUTPUT_LIMIT_REASONS)


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


def build_chat_payload(
    *,
    model: str,
    messages: list[ChatMessage],
    system_prompt: str,
    max_tokens: int,
    history_characters: int,
    stream: bool,
    structured: bool = False,
    reasoning_effort: str | None = None,
) -> dict[str, object]:
    """Assemble a Chat Completions request body.

    `reasoning_effort` is only sent when a caller passes it: providers that do not
    understand the field reject the whole request rather than ignoring it.
    """

    if structured:
        max_tokens = max(max_tokens, STRUCTURED_MIN_MAX_TOKENS)
    trimmed = _trim_messages(messages, max_characters=history_characters)
    payload: dict[str, object] = {
        "model": model,
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
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if structured:
        payload["response_format"] = {"type": "json_object"}
    if stream:
        payload["stream_options"] = {"include_usage": True}
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


def _completion_reasoning_tokens(payload: dict[str, object]) -> int | None:
    """Hidden reasoning tokens, when the provider reports them separately.

    Groq nests the count under `usage.completion_tokens_details.reasoning_tokens`.
    Only the count is read; the reasoning text itself is never stored or surfaced.
    """

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("reasoning_tokens")
    return value if isinstance(value, int) else None


def _completion_finish_reason(payload: dict[str, object]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    value = first.get("finish_reason")
    return value if isinstance(value, str) and value else None


def _delta_content(payload: dict[str, object]) -> str:
    """Visible text out of one streamed Chat Completions chunk.

    A chunk that only carries hidden reasoning contributes nothing here, which is
    what keeps that reasoning out of the transcript.
    """

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)
    return ""


def _as_dict(value: object) -> dict[str, object] | None:
    """Accept a dict as-is, or a JSON-encoded string that decodes to one."""

    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def parse_sse_data(line: str) -> dict[str, object] | None:
    line = line.strip()
    if not line:
        return None
    if line.startswith("data:"):
        payload = line.removeprefix("data:").strip()
    elif line.startswith("{"):
        # Algunos despliegues emiten JSON por línea sin framing SSE.
        payload = line
    else:
        return None
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
