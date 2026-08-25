from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage
from backend.app.providers.cloudflare_ai import (
    CloudAIConfigurationError,
    CloudAIUnavailableError,
    CloudflareAIClient,
    _completion_content,
    _is_transient_status,
    _selected_model,
    parse_sse_data,
)


class CloudflareAIProviderTests(unittest.TestCase):
    def test_sse_content_is_parsed(self) -> None:
        payload = parse_sse_data(
            'data: {"choices":[{"delta":{"content":"Hola"}}]}'
        )
        self.assertIsInstance(payload, dict)
        assert payload is not None
        choices = payload["choices"]
        self.assertEqual(choices[0]["delta"]["content"], "Hola")

    def test_sse_done_is_parsed(self) -> None:
        self.assertEqual(parse_sse_data("data: [DONE]"), {"done": True})

    def test_non_data_line_is_ignored(self) -> None:
        self.assertIsNone(parse_sse_data("event: ping"))

    def test_complete_chat_content_is_parsed(self) -> None:
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "Hola"}}]
        }
        self.assertEqual(_completion_content(payload), "Hola")

    def test_complete_chat_supports_text_parts(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Ho"},
                            {"type": "text", "text": "la"},
                        ],
                    }
                }
            ]
        }
        self.assertEqual(_completion_content(payload), "Hola")

    def test_client_requires_cloudflare_credentials(self) -> None:
        with self.assertRaises(CloudAIConfigurationError):
            CloudflareAIClient(Settings())

    def test_models_are_selected_by_mode(self) -> None:
        settings = Settings(
            cloudflare_quick_model="quick-model",
            cloudflare_deep_model="deep-model",
        )
        self.assertEqual(
            _selected_model(settings, SelectedMode.QUICK),
            "quick-model",
        )
        self.assertEqual(
            _selected_model(settings, SelectedMode.DEEP),
            "deep-model",
        )

    def test_only_gateway_failures_are_transient(self) -> None:
        for status_code in (502, 503, 504):
            self.assertTrue(_is_transient_status(status_code))
        for status_code in (400, 401, 403, 404, 429, 500):
            self.assertFalse(_is_transient_status(status_code))

    def test_internal_chat_uses_non_streaming_completion(self) -> None:
        captured_payload: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_payload
            captured_payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"objective":"ok"}',
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def client_factory(*_args, **kwargs):
            return real_async_client(
                transport=transport,
                timeout=kwargs.get("timeout"),
            )

        client = CloudflareAIClient(
            Settings(
                cloudflare_account_id="account",
                cloudflare_api_token="token",
            )
        )
        with patch(
            "backend.app.providers.cloudflare_ai.httpx.AsyncClient",
            side_effect=client_factory,
        ):
            result = asyncio.run(
                client.chat(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Planificá")],
                    system_prompt="Devolvé JSON.",
                )
            )

        self.assertEqual(result.content, '{"objective":"ok"}')
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 5)
        self.assertIs(captured_payload["stream"], False)
        self.assertNotIn("stream_options", captured_payload)

    def test_chat_retries_one_transient_503_before_succeeding(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(503, request=request, text="temporarily unavailable")
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "Hola"}}
                    ]
                },
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def client_factory(*_args, **kwargs):
            return real_async_client(
                transport=transport,
                timeout=kwargs.get("timeout"),
            )

        settings = Settings(
            cloudflare_account_id="account",
            cloudflare_api_token="token",
        )
        client = CloudflareAIClient(settings)

        with (
            patch(
                "backend.app.providers.cloudflare_ai.httpx.AsyncClient",
                side_effect=client_factory,
            ),
            patch(
                "backend.app.providers.cloudflare_ai.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
        ):
            result = asyncio.run(
                client.chat(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Hola")],
                    system_prompt="Respondé breve.",
                )
            )

        self.assertEqual(result.content, "Hola")
        self.assertEqual(attempts, 2)
        sleep.assert_awaited_once()

    def test_internal_chat_retries_empty_completion_before_failing(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"role": "assistant", "content": ""}}]},
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def client_factory(*_args, **kwargs):
            return real_async_client(
                transport=transport,
                timeout=kwargs.get("timeout"),
            )

        client = CloudflareAIClient(
            Settings(
                cloudflare_account_id="account",
                cloudflare_api_token="token",
            )
        )
        with (
            patch(
                "backend.app.providers.cloudflare_ai.httpx.AsyncClient",
                side_effect=client_factory,
            ),
            patch(
                "backend.app.providers.cloudflare_ai.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
        ):
            with self.assertRaises(CloudAIUnavailableError):
                asyncio.run(
                    client.chat(
                        mode=SelectedMode.QUICK,
                        messages=[ChatMessage(role="user", content="Planificá")],
                        system_prompt="Devolvé JSON.",
                    )
                )

        self.assertEqual(attempts, 3)
        self.assertEqual(sleep.await_count, 2)


if __name__ == "__main__":
    unittest.main()
