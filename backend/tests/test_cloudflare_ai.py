from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage
from backend.app.providers.cloudflare_ai import (
    CloudAIConfigurationError,
    CloudflareAIClient,
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
                text=(
                    'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n'
                    'data: [DONE]\n\n'
                ),
                headers={"content-type": "text/event-stream"},
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


if __name__ == "__main__":
    unittest.main()
