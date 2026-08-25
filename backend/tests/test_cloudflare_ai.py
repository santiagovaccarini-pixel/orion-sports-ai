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
    FINAL_VISIBLE_RESCUE_MIN_MAX_TOKENS,
    STRUCTURED_MIN_MAX_TOKENS,
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

    def test_structured_internal_chat_uses_json_mode_and_larger_budget(self) -> None:
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
                    structured=True,
                )
            )

        self.assertEqual(result.content, '{"objective":"ok"}')
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 5)
        self.assertIs(captured_payload["stream"], False)
        self.assertNotIn("stream_options", captured_payload)
        self.assertEqual(captured_payload["response_format"], {"type": "json_object"})
        self.assertGreaterEqual(captured_payload["max_tokens"], STRUCTURED_MIN_MAX_TOKENS)
        self.assertEqual(captured_payload["temperature"], 0.1)

    def test_regular_complete_chat_does_not_force_json_mode(self) -> None:
        captured_payload: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_payload
            captured_payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "Respuesta"}}
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
            asyncio.run(
                client.chat(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Hola")],
                    system_prompt="Respondé.",
                )
            )

        self.assertNotIn("response_format", captured_payload)

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
                        structured=True,
                    )
                )

        self.assertEqual(attempts, 3)
        self.assertEqual(sleep.await_count, 2)

    def test_stream_rescues_answer_when_reasoning_exhausts_visible_budget(self) -> None:
        captured_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            captured_payloads.append(payload)
            current_limit = int(payload["max_tokens"])
            if len(captured_payloads) == 1:
                body = (
                    f'data: {{"usage":{{"prompt_tokens":5000,"completion_tokens":{current_limit}}}}}\n\n'
                    "data: [DONE]\n\n"
                )
            else:
                body = (
                    'data: {"choices":[{"delta":{"content":"60 goles"}}]}\n\n'
                    'data: {"usage":{"prompt_tokens":5000,"completion_tokens":120}}\n\n'
                    "data: [DONE]\n\n"
                )
            return httpx.Response(
                200,
                request=request,
                text=body,
                headers={"content-type": "text/event-stream"},
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
                quick_max_tokens=768,
            )
        )

        async def collect_events():
            return [
                event
                async for event in client.chat_stream(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Pregunta")],
                    system_prompt="Respondé.",
                )
            ]

        with patch(
            "backend.app.providers.cloudflare_ai.httpx.AsyncClient",
            side_effect=client_factory,
        ):
            events = asyncio.run(collect_events())

        self.assertEqual(len(captured_payloads), 2)
        self.assertEqual(captured_payloads[0]["max_tokens"], 768)
        self.assertGreaterEqual(
            captured_payloads[1]["max_tokens"],
            FINAL_VISIBLE_RESCUE_MIN_MAX_TOKENS,
        )
        self.assertEqual("".join(event.content for event in events), "60 goles")
        self.assertTrue(events[-1].done)
        self.assertEqual(events[-1].completion_tokens, 120)


if __name__ == "__main__":
    unittest.main()
