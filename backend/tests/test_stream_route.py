from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.api.routes import PreparedChat, require_api_key
from backend.app.api.routes import _insufficient_web_response, _web_is_insufficient
from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatRequest, SportContext
from backend.app.main import app
from backend.app.providers.ollama import OllamaStreamEvent


async def fake_prepare_chat(*_args, **_kwargs) -> PreparedChat:
    return PreparedChat(
        selected_mode=SelectedMode.QUICK,
        recommended_mode=SelectedMode.QUICK,
        recommendation_reason="Consulta directa.",
        model="qwen3:4b-instruct",
        sport=SportContext.FOOTBALL,
    )


async def fake_chat_stream(*_args, **_kwargs):
    yield OllamaStreamEvent(content="Hola ", done=False, thread_limit=8)
    yield OllamaStreamEvent(content="Orion", done=False, thread_limit=8)
    yield OllamaStreamEvent(
        content="",
        done=True,
        total_duration_ms=1250.0,
        load_duration_ms=200.0,
        prompt_tokens=80,
        completion_tokens=12,
        tokens_per_second=9.6,
        thread_limit=8,
    )


async def fake_priority_monitor(stop_event, *_args, **_kwargs) -> None:
    await stop_event.wait()


class StreamRouteTests(unittest.TestCase):
    def test_insufficient_web_evidence_never_reaches_model_as_a_claim(self) -> None:
        context = "INVESTIGACIÓN WEB INSUFICIENTE: solo se obtuvieron 2 fuentes"
        self.assertTrue(_web_is_insufficient(context))
        self.assertIn("información preliminar", _insufficient_web_response(context))

    def test_stream_returns_safe_web_answer_without_calling_ollama(self) -> None:
        with (
            patch("backend.app.api.routes._prepare_chat", side_effect=fake_prepare_chat),
            patch("backend.app.api.routes._web_context", new_callable=unittest.mock.AsyncMock, return_value="INVESTIGACIÓN WEB INSUFICIENTE: solo se obtuvieron 2 fuentes"),
            patch("backend.app.api.routes.OllamaClient.chat_stream") as model,
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "Cuantos goles hizo un jugador?"}]},
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(response.status_code, 200)
        self.assertIn("información preliminar", events[1]["content"])
        model.assert_not_called()
    def test_api_key_is_required_when_configured(self) -> None:
        with patch(
            "backend.app.api.routes.get_settings",
            return_value=Settings(api_key="orion-test-key"),
        ):
            with self.assertRaises(HTTPException) as context:
                require_api_key("wrong-key")

            require_api_key("orion-test-key")

        self.assertEqual(context.exception.status_code, 401)

    def test_stream_emits_meta_content_and_done_events(self) -> None:
        with (
            patch(
                "backend.app.api.routes._prepare_chat",
                side_effect=fake_prepare_chat,
            ),
            patch(
                "backend.app.api.routes.OllamaClient.chat_stream",
                new=fake_chat_stream,
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
        self.assertEqual([event["type"] for event in events], [
            "meta",
            "content",
            "content",
            "done",
        ])
        self.assertEqual(events[0]["model"], "qwen3:4b-instruct")
        self.assertEqual(events[0]["sport"], "football")
        self.assertEqual(events[-1]["completion_tokens"], 12)
        build_prompt.assert_called_once_with(
            SportContext.FOOTBALL, SelectedMode.QUICK, "Hola"
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

        with (
            patch("backend.app.api.routes._prepare_chat", side_effect=fake_prepare_chat),
            patch("backend.app.api.routes._knowledge_chart", return_value=chart),
            patch("backend.app.api.routes.OllamaClient.chat_stream", new=fake_chat_stream),
            patch("backend.app.api.routes.maintain_ollama_priority", new=fake_priority_monitor),
            patch("backend.app.api.routes.build_system_prompt", return_value="PROMPT DE PRUEBA"),
            patch("backend.app.api.routes.lower_ollama_priority"),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "Graficá la distancia"}]},
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[1]["type"], "chart")
        self.assertEqual(events[1]["chart"], chart)

    def test_web_research_context_is_passed_to_prompt_when_enabled(self) -> None:
        with patch(
            "backend.app.api.routes.build_system_prompt",
            return_value="PROMPT DE PRUEBA",
        ), patch(
            "backend.app.api.routes.get_settings",
            return_value=Settings(web_enabled=True),
        ), patch(
            "backend.app.api.routes.research",
            new_callable=unittest.mock.AsyncMock,
            return_value=(),
        ):
            request = ChatRequest(
                messages=[{"role": "user", "content": "Buscá fuentes actuales sobre fuera de juego"}],
            )
            context = asyncio.run(__import__("backend.app.api.routes", fromlist=["_web_context"])._web_context(request))

        self.assertIn("INVESTIGACIÓN WEB INSUFICIENTE", context)


if __name__ == "__main__":
    unittest.main()
