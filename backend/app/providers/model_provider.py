from __future__ import annotations

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
    tokens_per_second: float | None = None
    thread_limit: int = 0


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
    ) -> ModelResult: ...

    def chat_stream(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> AsyncIterator[ModelStreamEvent]: ...


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
    ) -> ModelResult:
        # Ollama keeps its existing transport; the semantic prompt already requests
        # JSON. The structured flag is accepted so the provider interface stays
        # interchangeable with cloud providers.
        _ = structured
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
    ) -> AsyncIterator[ModelStreamEvent]:
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
        # Avoid spending the free inference quota on periodic health checks.
        models = tuple(dict.fromkeys((
            self.settings.cloudflare_quick_model,
            self.settings.cloudflare_deep_model,
        )))
        return ModelProviderStatus(online=True, installed_models=models)

    async def preflight(self, mode: SelectedMode) -> None:
        # Construction already validates credentials. A real inference request is the
        # authoritative availability check and should not be duplicated because the
        # cloud prototype has a finite free daily quota.
        _ = self.model_for(mode)

    async def chat(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
        structured: bool = False,
    ) -> ModelResult:
        try:
            result = await self.client.chat(
                mode=mode,
                messages=messages,
                system_prompt=system_prompt,
                structured=structured,
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
            thread_limit=0,
        )

    async def chat_stream(
        self,
        *,
        mode: SelectedMode,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            async for event in self.client.chat_stream(
                mode=mode,
                messages=messages,
                system_prompt=system_prompt,
            ):
                yield ModelStreamEvent(
                    content=event.content,
                    done=event.done,
                    model=event.model,
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    thread_limit=0,
                )
        except CloudAIConfigurationError as exc:
            raise ModelProviderConfigurationError(str(exc)) from exc
        except CloudAIUnavailableError as exc:
            raise ModelProviderUnavailableError(str(exc)) from exc


def create_model_provider(settings: Settings) -> ModelProvider:
    if settings.model_provider == "ollama":
        return OllamaModelProvider(settings)
    if settings.model_provider == "cloudflare":
        return CloudflareModelProvider(settings)
    raise ModelProviderConfigurationError(
        f"Proveedor de modelo no soportado: {settings.model_provider}."
    )
