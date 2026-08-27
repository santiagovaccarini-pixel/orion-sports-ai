from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage
from backend.app.providers.cloudflare_ai import CloudAIUnavailableError
from backend.app.providers.model_provider import (
    CloudflareModelProvider,
    ModelProviderConfigurationError,
    ModelProviderModelError,
    ModelProviderUnavailableError,
    OllamaModelProvider,
    _chunk_recovered_text,
    _model_is_installed,
    create_model_provider,
)


class ModelProviderTests(unittest.TestCase):
    def test_factory_keeps_ollama_as_default(self) -> None:
        provider = create_model_provider(Settings())
        self.assertIsInstance(provider, OllamaModelProvider)
        self.assertTrue(provider.uses_local_resources)
        self.assertEqual(provider.model_for(SelectedMode.QUICK), "qwen3:4b-instruct")
        self.assertEqual(provider.model_for(SelectedMode.DEEP), "qwen3:8b")

    def test_factory_builds_cloudflare_without_using_local_resources(self) -> None:
        provider = create_model_provider(
            Settings(
                model_provider="cloudflare",
                cloudflare_account_id="account-test",
                cloudflare_api_token="token-test",
                cloudflare_quick_model="cloud-quick",
                cloudflare_deep_model="cloud-deep",
            )
        )
        self.assertIsInstance(provider, CloudflareModelProvider)
        self.assertFalse(provider.uses_local_resources)
        self.assertEqual(provider.model_for(SelectedMode.QUICK), "cloud-quick")
        self.assertEqual(provider.model_for(SelectedMode.DEEP), "cloud-deep")

    def test_cloudflare_requires_credentials_before_chat(self) -> None:
        with self.assertRaises(ModelProviderConfigurationError):
            create_model_provider(Settings(model_provider="cloudflare"))

    def test_cloudflare_preflight_does_not_spend_inference_quota(self) -> None:
        provider = create_model_provider(
            Settings(
                model_provider="cloudflare",
                cloudflare_account_id="account-test",
                cloudflare_api_token="token-test",
            )
        )
        provider.client.chat = AsyncMock(side_effect=AssertionError("No debe inferir"))
        asyncio.run(provider.preflight(SelectedMode.QUICK))
        provider.client.chat.assert_not_called()

    def test_local_preflight_rejects_missing_model(self) -> None:
        provider = OllamaModelProvider(Settings(quick_model="missing-model"))
        provider.client.status = AsyncMock()
        provider.client.status.return_value.online = True
        provider.client.status.return_value.installed_models = ("other-model:latest",)
        provider.client.status.return_value.loaded_models = ()

        with self.assertRaises(ModelProviderModelError):
            asyncio.run(provider.preflight(SelectedMode.QUICK))

    def test_model_name_matching_accepts_ollama_tags(self) -> None:
        self.assertTrue(_model_is_installed("qwen3:4b", ("qwen3:4b",)))
        self.assertTrue(_model_is_installed("qwen3", ("qwen3:4b",)))
        self.assertFalse(_model_is_installed("qwen3:8b", ("qwen3:4b",)))

    def test_cloud_stream_recovers_once_with_non_streaming_response_before_visible_text(self) -> None:
        provider = CloudflareModelProvider(
            Settings(
                model_provider="cloudflare",
                cloudflare_account_id="account-test",
                cloudflare_api_token="token-test",
                cloudflare_quick_model="@cf/openai/gpt-oss-120b",
            )
        )

        async def broken_stream(**_kwargs):
            if False:
                yield None
            raise CloudAIUnavailableError(
                "El modelo cloud cerró el stream antes de informar su estado final."
            )

        provider.client.chat_stream = broken_stream
        provider.client.chat = AsyncMock(
            return_value=SimpleNamespace(
                content="respuesta recuperada",
                model="@cf/openai/gpt-oss-120b",
                prompt_tokens=100,
                completion_tokens=40,
                reasoning_tokens=15,
                finish_reason="completed",
                reasoning_effort="low",
                endpoint="responses",
            )
        )

        async def collect():
            return [
                event
                async for event in provider.chat_stream(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Pregunta")],
                    system_prompt="Respondé.",
                    reasoning_effort="low",
                )
            ]

        events = asyncio.run(collect())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].content, "respuesta recuperada")
        self.assertFalse(events[0].done)
        self.assertTrue(events[1].done)
        self.assertEqual(events[1].finish_reason, "completed")
        self.assertEqual(events[1].reasoning_tokens, 15)
        self.assertEqual(events[1].endpoint, "responses_stream_recovery")
        self.assertEqual(
            events[1].recovery_reason,
            "El modelo cloud cerró el stream antes de informar su estado final.",
        )

    def test_recovered_answer_is_delivered_as_multiple_chunks(self) -> None:
        provider = CloudflareModelProvider(
            Settings(
                model_provider="cloudflare",
                cloudflare_account_id="account-test",
                cloudflare_api_token="token-test",
                cloudflare_quick_model="@cf/openai/gpt-oss-120b",
            )
        )

        async def broken_stream(**_kwargs):
            if False:
                yield None
            raise CloudAIUnavailableError("cerró antes de tiempo")

        long_answer = " ".join(f"palabra{i}" for i in range(1, 13))
        provider.client.chat_stream = broken_stream
        provider.client.chat = AsyncMock(
            return_value=SimpleNamespace(
                content=long_answer,
                model="@cf/openai/gpt-oss-120b",
                prompt_tokens=50,
                completion_tokens=20,
                reasoning_tokens=5,
                finish_reason="completed",
                reasoning_effort="low",
                endpoint="responses",
            )
        )

        async def collect():
            return [
                event
                async for event in provider.chat_stream(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Pregunta")],
                    system_prompt="Respondé.",
                )
            ]

        events = asyncio.run(collect())
        content_events = [event for event in events if not event.done]
        self.assertGreater(len(content_events), 1)
        self.assertEqual(
            "".join(event.content for event in content_events), long_answer
        )
        self.assertTrue(events[-1].done)
        self.assertEqual(events[-1].endpoint, "responses_stream_recovery")
        provider.client.chat.assert_awaited_once()

    def test_cloud_stream_does_not_repeat_inference_after_partial_text(self) -> None:
        provider = CloudflareModelProvider(
            Settings(
                model_provider="cloudflare",
                cloudflare_account_id="account-test",
                cloudflare_api_token="token-test",
                cloudflare_quick_model="@cf/openai/gpt-oss-120b",
            )
        )

        async def partial_stream(**_kwargs):
            yield SimpleNamespace(
                content="texto parcial",
                done=False,
                model="@cf/openai/gpt-oss-120b",
                prompt_tokens=None,
                completion_tokens=None,
                reasoning_tokens=None,
                finish_reason=None,
                reasoning_effort="low",
                endpoint="responses",
            )
            raise CloudAIUnavailableError("stream interrumpido")

        provider.client.chat_stream = partial_stream
        provider.client.chat = AsyncMock(side_effect=AssertionError("No debe repetir"))

        async def consume() -> None:
            async for _ in provider.chat_stream(
                mode=SelectedMode.QUICK,
                messages=[ChatMessage(role="user", content="Pregunta")],
                system_prompt="Respondé.",
                reasoning_effort="low",
            ):
                pass

        with self.assertRaises(ModelProviderUnavailableError):
            asyncio.run(consume())
        provider.client.chat.assert_not_called()


class ChunkRecoveredTextTests(unittest.TestCase):
    def test_short_answer_stays_a_single_chunk(self) -> None:
        self.assertEqual(_chunk_recovered_text("ORION CLOUD OK"), ("ORION CLOUD OK",))

    def test_empty_answer_produces_no_chunks(self) -> None:
        self.assertEqual(_chunk_recovered_text(""), ())

    def test_long_answer_splits_preserving_exact_text(self) -> None:
        text = " ".join(f"palabra{i}" for i in range(1, 13))
        chunks = _chunk_recovered_text(text, words_per_chunk=4)
        self.assertEqual(len(chunks), 3)
        self.assertEqual("".join(chunks), text)

    def test_whitespace_and_newlines_are_preserved_across_chunks(self) -> None:
        text = "Primera línea.\nSegunda línea con  espacios.  Tercera."
        chunks = _chunk_recovered_text(text, words_per_chunk=2)
        self.assertEqual("".join(chunks), text)


if __name__ == "__main__":
    unittest.main()
