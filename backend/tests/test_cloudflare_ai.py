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
    _responses_content,
    _responses_finish_reason,
    _responses_usage,
    _selected_model,
    _trim_messages,
    parse_sse_data,
)


def _responses_json(
    text: str,
    *,
    status: str = "completed",
    reason: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
    reasoning_tokens: int = 7,
) -> dict[str, object]:
    return {
        "id": "resp_test",
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


class CloudflareAIProviderTests(unittest.TestCase):
    def test_sse_content_is_parsed(self) -> None:
        payload = parse_sse_data('data: {"type":"response.output_text.delta","delta":"Hola"}')
        self.assertIsInstance(payload, dict)
        assert payload is not None
        self.assertEqual(payload["delta"], "Hola")

    def test_sse_done_is_parsed(self) -> None:
        self.assertEqual(parse_sse_data("data: [DONE]"), {"done": True})

    def test_non_data_line_is_ignored(self) -> None:
        self.assertIsNone(parse_sse_data("event: ping"))

    def test_complete_chat_content_is_parsed_for_fallback_models(self) -> None:
        payload = {"choices": [{"message": {"role": "assistant", "content": "Hola"}}]}
        self.assertEqual(_completion_content(payload), "Hola")

    def test_responses_content_usage_and_finish_reason_are_parsed(self) -> None:
        payload = _responses_json("Hola")
        self.assertEqual(_responses_content(payload), "Hola")
        self.assertEqual(_responses_usage(payload), (10, 20, 7))
        self.assertEqual(_responses_finish_reason(payload), "completed")

    def test_responses_incomplete_reason_is_not_silently_treated_as_complete(self) -> None:
        payload = _responses_json("parcial", status="incomplete", reason="max_output_tokens")
        self.assertEqual(_responses_finish_reason(payload), "max_output_tokens")

    def test_client_requires_cloudflare_credentials(self) -> None:
        with self.assertRaises(CloudAIConfigurationError):
            CloudflareAIClient(Settings())

    def test_models_are_selected_by_mode(self) -> None:
        settings = Settings(
            cloudflare_quick_model="quick-model",
            cloudflare_deep_model="deep-model",
        )
        self.assertEqual(_selected_model(settings, SelectedMode.QUICK), "quick-model")
        self.assertEqual(_selected_model(settings, SelectedMode.DEEP), "deep-model")

    def test_only_gateway_failures_are_transient(self) -> None:
        for status_code in (502, 503, 504):
            self.assertTrue(_is_transient_status(status_code))
        for status_code in (400, 401, 403, 404, 429, 500):
            self.assertFalse(_is_transient_status(status_code))

    def test_cloud_history_is_bounded_and_keeps_latest_message(self) -> None:
        messages = [
            ChatMessage(role="user", content="a" * 100),
            ChatMessage(role="assistant", content="b" * 100),
            ChatMessage(role="user", content="pregunta final"),
        ]
        trimmed = _trim_messages(messages, max_characters=120)
        self.assertEqual(trimmed[-1].content, "pregunta final")
        self.assertLessEqual(sum(len(item.content) for item in trimmed), 120)

    def test_gpt_oss_structured_chat_uses_responses_api_reasoning_and_json_mode(self) -> None:
        captured_path = ""
        captured_payload: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_path, captured_payload
            captured_path = request.url.path
            captured_payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, request=request, json=_responses_json('{"objective":"ok"}'))

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def client_factory(*_args, **kwargs):
            return real_async_client(transport=transport, timeout=kwargs.get("timeout"))

        client = CloudflareAIClient(
            Settings(cloudflare_account_id="account", cloudflare_api_token="token")
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
                    reasoning_effort="low",
                )
            )

        self.assertTrue(captured_path.endswith("/ai/v1/responses"))
        self.assertEqual(result.content, '{"objective":"ok"}')
        self.assertEqual(result.reasoning_tokens, 7)
        self.assertEqual(result.finish_reason, "completed")
        self.assertEqual(result.reasoning_effort, "low")
        self.assertEqual(result.endpoint, "responses")
        self.assertEqual(captured_payload["reasoning"], {"effort": "low"})
        self.assertEqual(captured_payload["text"], {"format": {"type": "json_object"}})
        self.assertGreaterEqual(
            int(captured_payload["max_output_tokens"]), STRUCTURED_MIN_MAX_TOKENS
        )
        self.assertNotIn("temperature", captured_payload)
        self.assertNotIn("top_p", captured_payload)

    def test_non_gpt_oss_fallback_uses_recommended_sampling(self) -> None:
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
                            "message": {"role": "assistant", "content": "Respuesta"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3},
                },
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def client_factory(*_args, **kwargs):
            return real_async_client(transport=transport, timeout=kwargs.get("timeout"))

        client = CloudflareAIClient(
            Settings(
                cloudflare_account_id="account",
                cloudflare_api_token="token",
                cloudflare_quick_model="other-model",
            )
        )
        with patch(
            "backend.app.providers.cloudflare_ai.httpx.AsyncClient",
            side_effect=client_factory,
        ):
            result = asyncio.run(
                client.chat(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Hola")],
                    system_prompt="Respondé.",
                )
            )

        self.assertEqual(captured_payload["temperature"], 1.0)
        self.assertEqual(captured_payload["top_p"], 1.0)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.endpoint, "chat_completions")

    def test_responses_chat_retries_one_transient_503_before_succeeding(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(503, request=request, text="temporarily unavailable")
            return httpx.Response(200, request=request, json=_responses_json("Hola"))

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def client_factory(*_args, **kwargs):
            return real_async_client(transport=transport, timeout=kwargs.get("timeout"))

        client = CloudflareAIClient(
            Settings(cloudflare_account_id="account", cloudflare_api_token="token")
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

    def test_responses_chat_retries_output_limit_with_larger_budget(self) -> None:
        captured_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            captured_payloads.append(payload)
            if len(captured_payloads) == 1:
                return httpx.Response(
                    200,
                    request=request,
                    json=_responses_json(
                        "parcial",
                        status="incomplete",
                        reason="max_output_tokens",
                        output_tokens=int(payload["max_output_tokens"]),
                    ),
                )
            return httpx.Response(200, request=request, json=_responses_json("respuesta completa"))

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def client_factory(*_args, **kwargs):
            return real_async_client(transport=transport, timeout=kwargs.get("timeout"))

        client = CloudflareAIClient(
            Settings(
                cloudflare_account_id="account",
                cloudflare_api_token="token",
                quick_max_tokens=1536,
            )
        )
        with patch(
            "backend.app.providers.cloudflare_ai.httpx.AsyncClient",
            side_effect=client_factory,
        ):
            result = asyncio.run(
                client.chat(
                    mode=SelectedMode.QUICK,
                    messages=[ChatMessage(role="user", content="Pregunta")],
                    system_prompt="Respondé.",
                )
            )

        self.assertEqual(len(captured_payloads), 2)
        self.assertGreaterEqual(
            int(captured_payloads[1]["max_output_tokens"]),
            FINAL_VISIBLE_RESCUE_MIN_MAX_TOKENS,
        )
        self.assertEqual(result.content, "respuesta completa")

    def test_responses_stream_exposes_usage_reasoning_and_completion_status(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            completed = _responses_json(
                "60 goles", input_tokens=5000, output_tokens=120, reasoning_tokens=80
            )
            body = (
                'data: {"type":"response.output_text.delta","delta":"60 goles"}\n\n'
                + "data: "
                + json.dumps({"type": "response.completed", "response": completed})
                + "\n\n"
                + "data: [DONE]\n\n"
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
            return real_async_client(transport=transport, timeout=kwargs.get("timeout"))

        client = CloudflareAIClient(
            Settings(cloudflare_account_id="account", cloudflare_api_token="token")
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

        self.assertEqual("".join(event.content for event in events), "60 goles")
        self.assertTrue(events[-1].done)
        self.assertEqual(events[-1].prompt_tokens, 5000)
        self.assertEqual(events[-1].completion_tokens, 120)
        self.assertEqual(events[-1].reasoning_tokens, 80)
        self.assertEqual(events[-1].finish_reason, "completed")
        self.assertEqual(events[-1].reasoning_effort, "low")

    def test_stream_does_not_silently_accept_incomplete_visible_answer(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            incomplete = _responses_json(
                "parcial", status="incomplete", reason="max_output_tokens"
            )
            body = (
                'data: {"type":"response.output_text.delta","delta":"parcial"}\n\n'
                + "data: "
                + json.dumps({"type": "response.incomplete", "response": incomplete})
                + "\n\n"
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
            return real_async_client(transport=transport, timeout=kwargs.get("timeout"))

        client = CloudflareAIClient(
            Settings(cloudflare_account_id="account", cloudflare_api_token="token")
        )

        async def consume() -> None:
            async for _ in client.chat_stream(
                mode=SelectedMode.QUICK,
                messages=[ChatMessage(role="user", content="Pregunta")],
                system_prompt="Respondé.",
            ):
                pass

        with patch(
            "backend.app.providers.cloudflare_ai.httpx.AsyncClient",
            side_effect=client_factory,
        ):
            with self.assertRaises(CloudAIUnavailableError):
                asyncio.run(consume())


if __name__ == "__main__":
    unittest.main()
