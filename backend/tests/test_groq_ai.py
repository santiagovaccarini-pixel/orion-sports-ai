from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage
from backend.app.providers.groq_ai import (
    GroqAIClient,
    _reasoning_effort,
    _selected_model,
)
from backend.app.providers.model_provider import (
    GroqModelProvider,
    ModelProviderConfigurationError,
    ModelProviderUnavailableError,
    create_model_provider,
)
from backend.app.providers.openai_compatible import (
    CloudAIConfigurationError,
    CloudAIUnavailableError,
    _completion_reasoning_tokens,
    _delta_content,
)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "model_provider": "groq",
        "groq_api_key": "clave-de-prueba",
        "quick_max_tokens": 1536,
        "deep_max_tokens": 3072,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _completion_json(content: str, *, reasoning_tokens: int | None = None) -> dict:
    usage: dict[str, object] = {"prompt_tokens": 120, "completion_tokens": 45}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }


def _run_with_transport(client: GroqAIClient, handler, coroutine_factory):
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def client_factory(*_args, **kwargs):
        return real_async_client(transport=transport, timeout=kwargs.get("timeout"))

    with patch(
        "backend.app.providers.groq_ai.httpx.AsyncClient", side_effect=client_factory
    ):
        return asyncio.run(coroutine_factory())


class GroqClientTests(unittest.TestCase):
    def test_client_requires_an_api_key(self) -> None:
        with self.assertRaises(CloudAIConfigurationError):
            GroqAIClient(Settings(model_provider="groq"))

    def test_models_and_effort_are_selected_by_mode(self) -> None:
        settings = _settings(
            groq_quick_model="modelo-rapido",
            groq_deep_model="modelo-profundo",
            groq_quick_reasoning_effort="low",
            groq_deep_reasoning_effort="high",
        )
        self.assertEqual(_selected_model(settings, SelectedMode.QUICK), "modelo-rapido")
        self.assertEqual(_selected_model(settings, SelectedMode.DEEP), "modelo-profundo")
        self.assertEqual(_reasoning_effort(settings, SelectedMode.QUICK), "low")
        self.assertEqual(_reasoning_effort(settings, SelectedMode.DEEP), "high")
        # An explicit per-stage override wins over the configured default.
        self.assertEqual(
            _reasoning_effort(settings, SelectedMode.QUICK, "medium"), "medium"
        )

    def test_structured_request_carries_reasoning_effort_and_json_mode(self) -> None:
        captured_path = ""
        captured_payload: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_path, captured_payload
            captured_path = request.url.path
            captured_payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200, request=request, json=_completion_json('{"objective":"ok"}')
            )

        client = GroqAIClient(_settings(groq_quick_model="openai/gpt-oss-120b"))
        result = _run_with_transport(
            client,
            handler,
            lambda: client.chat(
                mode=SelectedMode.QUICK,
                messages=[ChatMessage(role="user", content="Planificá")],
                system_prompt="Devolvé JSON.",
                structured=True,
                reasoning_effort="low",
            ),
        )

        self.assertEqual(captured_path, "/openai/v1/chat/completions")
        self.assertEqual(captured_payload["model"], "openai/gpt-oss-120b")
        # Groq takes the reasoning knob inline; Cloudflare needs a separate API for it.
        self.assertEqual(captured_payload["reasoning_effort"], "low")
        self.assertEqual(captured_payload["response_format"], {"type": "json_object"})
        self.assertEqual(result.content, '{"objective":"ok"}')
        self.assertEqual(result.reasoning_effort, "low")
        self.assertEqual(result.endpoint, "chat_completions")

    def test_hidden_reasoning_is_counted_but_never_returned_as_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = _completion_json("Respuesta visible", reasoning_tokens=812)
            payload["choices"][0]["message"]["reasoning"] = "borrador interno"
            return httpx.Response(200, request=request, json=payload)

        client = GroqAIClient(_settings())
        result = _run_with_transport(
            client,
            handler,
            lambda: client.chat(
                mode=SelectedMode.QUICK,
                messages=[ChatMessage(role="user", content="Hola")],
                system_prompt="Contestá.",
            ),
        )

        self.assertEqual(result.content, "Respuesta visible")
        self.assertNotIn("borrador interno", result.content)
        self.assertEqual(result.reasoning_tokens, 812)
        self.assertEqual(result.prompt_tokens, 120)
        self.assertEqual(result.completion_tokens, 45)

    def test_streamed_reasoning_deltas_never_reach_the_transcript(self) -> None:
        chunks = [
            {"choices": [{"delta": {"reasoning": "pensando en voz baja"}}]},
            {"choices": [{"delta": {"content": "Hola"}}]},
            {"choices": [{"delta": {"content": " Santiago"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "completion_tokens_details": {"reasoning_tokens": 99},
                },
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request, content=body.encode("utf-8"))

        client = GroqAIClient(_settings())

        async def collect() -> list:
            return [
                event
                async for event in client.chat_stream(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Hola")],
                    system_prompt="Contestá.",
                )
            ]

        events = _run_with_transport(client, handler, collect)

        visible = "".join(event.content for event in events)
        self.assertEqual(visible, "Hola Santiago")
        self.assertNotIn("pensando en voz baja", visible)
        final = events[-1]
        self.assertTrue(final.done)
        self.assertEqual(final.prompt_tokens, 10)
        self.assertEqual(final.completion_tokens, 4)
        self.assertEqual(final.reasoning_tokens, 99)

    def test_rate_limit_explains_that_the_free_plan_is_too_small(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, request=request, text="rate_limit_exceeded")

        client = GroqAIClient(_settings())
        with self.assertRaises(CloudAIUnavailableError) as caught:
            _run_with_transport(
                client,
                handler,
                lambda: client.chat(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Hola")],
                    system_prompt="Contestá.",
                ),
            )
        self.assertIn("gratuito", str(caught.exception))

    def test_rejected_key_is_a_configuration_problem_not_an_outage(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, request=request, text="invalid api key")

        client = GroqAIClient(_settings())
        with self.assertRaises(CloudAIConfigurationError):
            _run_with_transport(
                client,
                handler,
                lambda: client.chat(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Hola")],
                    system_prompt="Contestá.",
                ),
            )


class GroqParsingTests(unittest.TestCase):
    def test_reasoning_tokens_are_read_from_the_nested_usage_details(self) -> None:
        self.assertEqual(
            _completion_reasoning_tokens(
                {"usage": {"completion_tokens_details": {"reasoning_tokens": 7}}}
            ),
            7,
        )
        self.assertIsNone(_completion_reasoning_tokens({"usage": {}}))
        self.assertIsNone(_completion_reasoning_tokens({}))

    def test_a_delta_without_visible_content_contributes_nothing(self) -> None:
        self.assertEqual(_delta_content({"choices": [{"delta": {"content": "x"}}]}), "x")
        self.assertEqual(_delta_content({"choices": [{"delta": {"reasoning": "x"}}]}), "")
        self.assertEqual(_delta_content({"choices": []}), "")


class GroqTokenBudgetTests(unittest.TestCase):
    """Groq must inherit the cloud budget, not Ollama's much smaller local one."""

    def test_groq_gets_the_cloud_token_budget(self) -> None:
        import os
        from backend.app.core.config import get_settings

        previous = {
            key: os.environ.get(key)
            for key in ("ORION_MODEL_PROVIDER", "ORION_GROQ_API_KEY")
        }
        os.environ["ORION_MODEL_PROVIDER"] = "groq"
        os.environ["ORION_GROQ_API_KEY"] = "clave-de-prueba"
        get_settings.cache_clear()
        try:
            settings = get_settings()
            self.assertEqual(settings.model_provider, "groq")
            self.assertEqual(settings.quick_max_tokens, settings.cloudflare_quick_max_tokens)
            self.assertEqual(settings.deep_max_tokens, settings.cloudflare_deep_max_tokens)
            self.assertEqual(settings.groq_quick_model, "openai/gpt-oss-120b")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            get_settings.cache_clear()

    def test_an_unknown_provider_is_rejected_by_name(self) -> None:
        import os
        from backend.app.core.config import get_settings

        previous = os.environ.get("ORION_MODEL_PROVIDER")
        os.environ["ORION_MODEL_PROVIDER"] = "inventado"
        get_settings.cache_clear()
        try:
            with self.assertRaises(RuntimeError) as caught:
                get_settings()
            self.assertIn("groq", str(caught.exception))
        finally:
            if previous is None:
                os.environ.pop("ORION_MODEL_PROVIDER", None)
            else:
                os.environ["ORION_MODEL_PROVIDER"] = previous
            get_settings.cache_clear()


class GroqProviderWiringTests(unittest.TestCase):
    def test_the_factory_builds_the_groq_provider(self) -> None:
        provider = create_model_provider(_settings())
        self.assertIsInstance(provider, GroqModelProvider)
        self.assertEqual(provider.name, "groq")
        self.assertFalse(provider.uses_local_resources)

    def test_a_missing_key_surfaces_as_configuration_not_as_a_crash(self) -> None:
        with self.assertRaises(ModelProviderConfigurationError):
            create_model_provider(Settings(model_provider="groq"))

    def test_status_reports_the_configured_groq_models(self) -> None:
        provider = create_model_provider(
            _settings(groq_quick_model="rapido", groq_deep_model="profundo")
        )
        status = asyncio.run(provider.status())
        self.assertTrue(status.online)
        self.assertEqual(status.installed_models, ("rapido", "profundo"))

    def test_provider_errors_are_mapped_to_orion_level_errors(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, request=request, text="boom")

        provider = create_model_provider(_settings())
        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def client_factory(*_args, **kwargs):
            return real_async_client(transport=transport, timeout=kwargs.get("timeout"))

        with patch(
            "backend.app.providers.groq_ai.httpx.AsyncClient", side_effect=client_factory
        ):
            with self.assertRaises(ModelProviderUnavailableError):
                asyncio.run(
                    provider.chat(
                        mode=SelectedMode.QUICK,
                        messages=[ChatMessage(role="user", content="Hola")],
                        system_prompt="Contestá.",
                    )
                )


if __name__ == "__main__":
    unittest.main()
