from __future__ import annotations

import re
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage
from backend.app.providers.cloudflare_ai import (
    CloudAIConfigurationError,
    CloudAIUnavailableError,
    CloudflareAIClient,
)
from backend.app.providers.ollama import (
    ModelNotInstalledError,
    OllamaClient,
    OllamaUnavailableError,
)


class ModelProviderUnavailableError(RuntimeError):
    """The selected model provider cannot complete the request right now."""


class ModelProviderConfigurationError(RuntimeError):
    """The selected provider is missing or has invalid configuration."""


class ModelProviderModelError(RuntimeError):
    """The requested model is not available in the selected provider."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"El modelo {model} no está disponible.")


@dataclass(frozen=True, slots=True)
class ModelProviderStatus:
    online: bool
    installed_models: tuple[str, ...] = ()
    loaded_models: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelResult:
    content: str
    model: str
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    reasoning_effort: str | None = None
    endpoint: str | None = None
    tokens_per_second: float | None = None
    thread_limit: int = 0


@dataclass(frozen=True, slots=True)
class ModelStreamEvent:
    content: str
    done: bool
    model: str
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    reasoning_effort: str | None = None
    endpoint: str | None = None
    tokens_per_second: float | None = None
    thread_limit: int = 0
    recovery_reason: str | None = None


class ModelProvider(Protocol):
    name: str
    uses_local_resources: bool

    def model_for(self, mode: SelectedMode) -> str: ...

    async def status(self) -> ModelProviderStatus: ...

    async def preflight(self, mode: SelectedMode) -> None: ...

    async def chat(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        structured: bool = False,
        reasoning_effort: str | None = None,
    ) -> ModelResult: ...

    def chat_stream(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...


RECOVERY_WORDS_PER_CHUNK = 4


def _chunk_recovered_text(
    text: str, *, words_per_chunk: int = RECOVERY_WORDS_PER_CHUNK
) -> tuple[str, ...]:
    """Split an already-complete answer into small pieces for perceived streaming.

    Cloudflare's Responses API stream for gpt-oss does not reliably deliver
    incremental content for this deployment (confirmed empirically: the primary
    stream consistently closes after an empty snapshot). Recovery gets the full
    answer in one non-streaming call; splitting it here lets the client still
    render a progressive reveal instead of the whole answer appearing at once.
    """

    tokens = re.findall(r"\S+\s*", text)
    if not tokens:
        return (text,) if text else ()
    chunks = [
        "".join(tokens[index : index + words_per_chunk])
        for index in range(0, len(tokens), words_per_chunk)
    ]
    return tuple(chunk for chunk in chunks if chunk)


def _model_is_installed(model: str, installed_models: tuple[str, ...]) -> bool:
    return any(
        installed == model
        or installed.startswith(f"{model}:")
        or model.startswith(f"{installed}:")
        for installed in installed_models
    )


class OllamaModelProvider:
    name = "ollama"
    uses_local_resources = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OllamaClient(settings)

    def model_for(self, mode: SelectedMode) -> str:
        return (
            self.settings.quick_model
            if mode is SelectedMode.QUICK
            else self.settings.deep_model
        )

    async def status(self) -> ModelProviderStatus:
        status = await self.client.status()
        return ModelProviderStatus(
            online=status.online,
            installed_models=status.installed_models,
            loaded_models=status.loaded_models,
        )

    async def preflight(self, mode: SelectedMode) -> None:
        model = self.model_for(mode)
        status = await self.status()
        if not status.online:
            raise ModelProviderUnavailableError(
                "No se pudo conectar con Ollama en esta computadora."
            )
        if not _model_is_installed(model, status.installed_models):
            raise ModelProviderModelError(model)

    async def chat(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        structured: bool = False,
        reasoning_effort: str | None = None,
    ) -> ModelResult:
        # Ollama retains its current transport. reasoning_effort is a cloud-facing
        # capability and is intentionally ignored by this local provider.
        _ = (structured, reasoning_effort)
        model = self.model_for(mode)
        try:
            result = await self.client.chat(
                model=model,
                mode=mode,
                messages=messages,
                system_prompt=system_prompt,
            )
        except ModelNotInstalledError as exc:
            raise ModelProviderModelError(exc.model) from exc
        except OllamaUnavailableError as exc:
            raise ModelProviderUnavailableError(str(exc)) from exc

        return ModelResult(
            content=result.content,
            model=model,
            total_duration_ms=result.total_duration_ms,
            load_duration_ms=result.load_duration_ms,
            prompt_eval_duration_ms=result.prompt_eval_duration_ms,
            eval_duration_ms=result.eval_duration_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            tokens_per_second=result.tokens_per_second,
            thread_limit=result.thread_limit,
        )

    async def chat_stream(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = reasoning_effort
        model = self.model_for(mode)
        try:
            async for event in self.client.chat_stream(
                model=model,
                mode=mode,
                messages=messages,
                system_prompt=system_prompt,
            ):
                yield ModelStreamEvent(
                    content=event.content,
                    done=event.done,
                    model=model,
                    total_duration_ms=event.total_duration_ms,
                    load_duration_ms=event.load_duration_ms,
                    prompt_eval_duration_ms=event.prompt_eval_duration_ms,
                    eval_duration_ms=event.eval_duration_ms,
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    tokens_per_second=event.tokens_per_second,
                    thread_limit=event.thread_limit or 0,
                )
        except ModelNotInstalledError as exc:
            raise ModelProviderModelError(exc.model) from exc
        except OllamaUnavailableError as exc:
            raise ModelProviderUnavailableError(str(exc)) from exc


class CloudflareModelProvider:
    name = "cloudflare"
    uses_local_resources = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            self.client = CloudflareAIClient(settings)
        except CloudAIConfigurationError as exc:
            raise ModelProviderConfigurationError(str(exc)) from exc

    def model_for(self, mode: SelectedMode) -> str:
        return (
            self.settings.cloudflare_quick_model
            if mode is SelectedMode.QUICK
            else self.settings.cloudflare_deep_model
        )

    async def status(self) -> ModelProviderStatus:
        models = tuple(
            dict.fromkeys(
                (
                    self.settings.cloudflare_quick_model,
                    self.settings.cloudflare_deep_model,
                )
            )
        )
        return ModelProviderStatus(online=True, installed_models=models)

    async def preflight(self, mode: SelectedMode) -> None:
        _ = self.model_for(mode)

    async def chat(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        structured: bool = False,
        reasoning_effort: str | None = None,
    ) -> ModelResult:
        try:
            result = await self.client.chat(
                mode=mode,
                messages=messages,
                system_prompt=system_prompt,
                structured=structured,
                reasoning_effort=reasoning_effort,
            )
        except CloudAIConfigurationError as exc:
            raise ModelProviderConfigurationError(str(exc)) from exc
        except CloudAIUnavailableError as exc:
            raise ModelProviderUnavailableError(str(exc)) from exc

        return ModelResult(
            content=result.content,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=result.reasoning_tokens,
            finish_reason=result.finish_reason,
            reasoning_effort=result.reasoning_effort,
            endpoint=result.endpoint,
            thread_limit=0,
        )

    async def chat_stream(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        visible_content_emitted = False
        try:
            async for event in self.client.chat_stream(
                mode=mode,
                messages=messages,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
            ):
                if event.content:
                    visible_content_emitted = True
                yield ModelStreamEvent(
                    content=event.content,
                    done=event.done,
                    model=event.model,
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    reasoning_tokens=event.reasoning_tokens,
                    finish_reason=event.finish_reason,
                    reasoning_effort=event.reasoning_effort,
                    endpoint=event.endpoint,
                    thread_limit=0,
                )
        except CloudAIConfigurationError as exc:
            raise ModelProviderConfigurationError(str(exc)) from exc
        except CloudAIUnavailableError as exc:
            if visible_content_emitted:
                raise ModelProviderUnavailableError(str(exc)) from exc
            # Transport resilience only: if Responses streaming closes before any
            # visible text, retry exactly once as a non-streaming Responses request.
            # This preserves the same model, prompt and reasoning effort without
            # duplicating a partially delivered answer.
            try:
                recovered = await self.client.chat(
                    mode=mode,
                    messages=messages,
                    system_prompt=system_prompt,
                    structured=False,
                    reasoning_effort=reasoning_effort,
                )
            except CloudAIConfigurationError as recovery_exc:
                raise ModelProviderConfigurationError(str(recovery_exc)) from recovery_exc
            except CloudAIUnavailableError as recovery_exc:
                raise ModelProviderUnavailableError(str(recovery_exc)) from recovery_exc

            recovery_endpoint = f"{recovered.endpoint or 'responses'}_stream_recovery"
            for chunk in _chunk_recovered_text(recovered.content):
                yield ModelStreamEvent(
                    content=chunk,
                    done=False,
                    model=recovered.model,
                    reasoning_effort=recovered.reasoning_effort,
                    endpoint=recovery_endpoint,
                    thread_limit=0,
                    recovery_reason=str(exc),
                )
            yield ModelStreamEvent(
                content="",
                done=True,
                model=recovered.model,
                prompt_tokens=recovered.prompt_tokens,
                completion_tokens=recovered.completion_tokens,
                reasoning_tokens=recovered.reasoning_tokens,
                finish_reason=recovered.finish_reason,
                reasoning_effort=recovered.reasoning_effort,
                endpoint=recovery_endpoint,
                thread_limit=0,
                recovery_reason=str(exc),
            )


def create_model_provider(settings: Settings) -> ModelProvider:
    if settings.model_provider == "ollama":
        return OllamaModelProvider(settings)
    if settings.model_provider == "cloudflare":
        return CloudflareModelProvider(settings)
    raise ModelProviderConfigurationError(
        f"Proveedor de modelo no soportado: {settings.model_provider}."
    )
