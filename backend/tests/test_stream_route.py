from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.api.routes import (
    PreparedChat,
    _insufficient_web_response,
    _prepare_chat,
    _web_is_insufficient,
    require_api_key,
)
from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatRequest, SportContext
from backend.app.main import app
from backend.app.providers.model_provider import (
    ModelProviderStatus,
    ModelStreamEvent,
)


async def fake_prepare_chat(*_args, **_kwargs) -> PreparedChat:
    return PreparedChat(
        selected_mode=SelectedMode.QUICK,
        recommended_mode=SelectedMode.QUICK,
        recommendation_reason="Consulta directa.",
        model="test-model",
        sport=SportContext.FOOTBALL,
    )


async def fake_priority_monitor(stop_event, *_args, **_kwargs) -> None:
    await stop_event.wait()


class FakeProvider:
    name = "ollama"
    uses_local_resources = True

    def model_for(self, mode: SelectedMode) -> str:
        return "test-model"

    async def status(self) -> ModelProviderStatus:
        return ModelProviderStatus(True, ("test-model",), ("test-model",))

    async def preflight(self, mode: SelectedMode) -> None:
        return None

    async def chat(self, **_kwargs):  # pragma: no cover - stream tests use chat_stream
        raise AssertionError("chat() no debía ejecutarse en esta prueba")

    async def chat_stream(self, **_kwargs):
        yield ModelStreamEvent(
            content="Hola ",
            done=False,
            model="test-model",
            thread_limit=8,
        )
        yield ModelStreamEvent(
            content="Orion",
            done=False,
            model="test-model",
            thread_limit=8,
        )
        yield ModelStreamEvent(
            content="",
            done=True,
            model="test-model",
            total_duration_ms=1250.0,
            load_duration_ms=200.0,
            prompt_tokens=80,
            completion_tokens=12,
            tokens_per_second=9.6,
            thread_limit=8,
        )


class FakeCloudProvider(FakeProvider):
    name = "cloudflare"
    uses_local_resources = False

    async def chat_stream(self, **_kwargs):
        yield ModelStreamEvent(
            content="Cloud",
            done=False,
            model="test-cloud-model",
        )
        yield ModelStreamEvent(
            content="",
            done=True,
            model="test-cloud-model",
            prompt_tokens=20,
            completion_tokens=4,
        )

    def model_for(self, mode: SelectedMode) -> str:
        return "test-cloud-model"


class StreamRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        # require_api_key now fails closed when no key is configured, which
        # breaks route-level TestClient calls that don't send a key. These
        # tests exercise streaming/routing behavior, not auth, so bypass it
        # here; auth itself is covered by test_api_key_is_required_when_configured.
        app.dependency_overrides[require_api_key] = lambda: None

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)

    def test_insufficient_web_evidence_never_reaches_model_as_a_claim(self) -> None:
        context = "INVESTIGACIÓN WEB INSUFICIENTE: solo se obtuvieron 2 fuentes"
        self.assertTrue(_web_is_insufficient(context))
        self.assertIn("información preliminar", _insufficient_web_response(context))

    def test_stream_returns_safe_web_answer_without_calling_model(self) -> None:
        provider = FakeProvider()
        provider.chat_stream = AsyncMock()
        with (
            patch(
                "backend.app.api.routes._provider_or_http_error",
                return_value=provider,
            ),
            patch(
                "backend.app.api.routes._prepare_chat",
                side_effect=fake_prepare_chat,
            ),
            patch(
                "backend.app.api.routes._web_context",
                new_callable=AsyncMock,
                return_value=(
                    "INVESTIGACIÓN WEB INSUFICIENTE: solo se obtuvieron 2 fuentes"
                ),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={
                    "messages": [
                        {"role": "user", "content": "Cuantos goles hizo un jugador?"}
                    ]
                },
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(response.status_code, 200)
        self.assertIn("información preliminar", events[1]["content"])
        provider.chat_stream.assert_not_called()

    def test_api_key_is_required_when_configured(self) -> None:
        with patch(
            "backend.app.api.routes.get_settings",
            return_value=Settings(api_key="orion-test-key"),
        ):
            with self.assertRaises(HTTPException) as context:
                require_api_key("wrong-key")

            require_api_key("orion-test-key")

        self.assertEqual(context.exception.status_code, 401)

    def test_api_key_check_fails_closed_when_unconfigured(self) -> None:
        # An unset ORION_API_KEY must never mean "no auth required" - that was
        # a real production misconfiguration hazard. Every request must be
        # rejected until an operator configures the key, not silently allowed.
        with patch(
            "backend.app.api.routes.get_settings",
            return_value=Settings(api_key=None),
        ):
            with self.assertRaises(HTTPException) as context:
                require_api_key("any-key")
            with self.assertRaises(HTTPException) as context_no_header:
                require_api_key(None)

        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail["code"], "server_misconfigured")
        self.assertEqual(context_no_header.exception.status_code, 503)

    def test_stream_emits_meta_content_and_done_events(self) -> None:
        provider = FakeProvider()
        with (
            patch(
                "backend.app.api.routes._provider_or_http_error",
                return_value=provider,
            ),
            patch(
                "backend.app.api.routes._prepare_chat",
                side_effect=fake_prepare_chat,
            ),
            patch(
                "backend.app.api.routes.maintain_ollama_priority",
                new=fake_priority_monitor,
            ),
            patch(
                "backend.app.api.routes.build_system_prompt",
                return_value="PROMPT DE PRUEBA",
            ) as build_prompt,
            patch("backend.app.api.routes.lower_ollama_priority"),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "mode": "quick",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("application/x-ndjson")
        )
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(
            [event["type"] for event in events],
            ["meta", "content", "content", "done"],
        )
        self.assertEqual(events[0]["model"], "test-model")
        self.assertEqual(events[0]["sport"], "football")
        self.assertEqual(events[-1]["completion_tokens"], 12)
        build_prompt.assert_called_once_with(
            SportContext.FOOTBALL,
            SelectedMode.QUICK,
            "Hola",
        )

    def test_stream_emits_chart_before_model_content_when_available(self) -> None:
        chart = {
            "type": "bar",
            "title": "Total Distance por período",
            "unit": "m",
            "source": "sesion.csv",
            "metric": "Total Distance",
            "points": [{"label": "Session", "value": 6330.83}],
        }
        provider = FakeProvider()

        with (
            patch(
                "backend.app.api.routes._provider_or_http_error",
                return_value=provider,
            ),
            patch(
                "backend.app.api.routes._prepare_chat",
                side_effect=fake_prepare_chat,
            ),
            patch("backend.app.api.routes._knowledge_chart", return_value=chart),
            patch(
                "backend.app.api.routes.maintain_ollama_priority",
                new=fake_priority_monitor,
            ),
            patch(
                "backend.app.api.routes.build_system_prompt",
                return_value="PROMPT DE PRUEBA",
            ),
            patch("backend.app.api.routes.lower_ollama_priority"),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={
                    "messages": [
                        {"role": "user", "content": "Graficá la distancia"}
                    ]
                },
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[1]["type"], "chart")
        self.assertEqual(events[1]["chart"], chart)

    def test_cloud_stream_never_uses_local_cpu_protection(self) -> None:
        provider = FakeCloudProvider()
        with (
            patch(
                "backend.app.api.routes._provider_or_http_error",
                return_value=provider,
            ),
            patch("backend.app.api.routes._web_context", new_callable=AsyncMock, return_value=""),
            patch("backend.app.api.routes.read_snapshot") as read_snapshot,
            patch("backend.app.api.routes.evaluate_resources") as evaluate_resources,
            patch("backend.app.api.routes.lower_ollama_priority") as lower_priority,
            patch("backend.app.api.routes.maintain_ollama_priority") as maintain_priority,
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "mode": "quick",
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[0]["model"], "test-cloud-model")
        self.assertEqual(events[-1]["thread_limit"], 0)
        read_snapshot.assert_not_called()
        evaluate_resources.assert_not_called()
        lower_priority.assert_not_called()
        maintain_priority.assert_not_called()

    def test_prepare_chat_uses_cloud_model_name(self) -> None:
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hola"}],
            mode="deep",
        )
        prepared = asyncio.run(
            _prepare_chat(
                request,
                preflight_model=True,
                provider=FakeCloudProvider(),
            )
        )
        self.assertEqual(prepared.model, "test-cloud-model")
        self.assertEqual(prepared.selected_mode, SelectedMode.DEEP)

    def test_web_research_context_is_passed_to_prompt_when_enabled(self) -> None:
        with patch(
            "backend.app.api.routes.build_system_prompt",
            return_value="PROMPT DE PRUEBA",
        ), patch(
            "backend.app.api.routes.get_settings",
            return_value=Settings(web_enabled=True),
        ), patch(
            "backend.app.api.routes.research",
            new_callable=AsyncMock,
            return_value=(),
        ):
            request = ChatRequest(
                messages=[
                    {
                        "role": "user",
                        "content": "Buscá fuentes actuales sobre fuera de juego",
                    }
                ],
            )
            context = asyncio.run(
                __import__(
                    "backend.app.api.routes",
                    fromlist=["_web_context"],
                )._web_context(request)
            )

        self.assertIn("INVESTIGACIÓN WEB INSUFICIENTE", context)


if __name__ == "__main__":
    unittest.main()
