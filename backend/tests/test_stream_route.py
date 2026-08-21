from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api.routes import PreparedChat
from backend.app.domain.models import SelectedMode
from backend.app.main import app
from backend.app.providers.ollama import OllamaStreamEvent


async def fake_prepare_chat(*_args, **_kwargs) -> PreparedChat:
    return PreparedChat(
        selected_mode=SelectedMode.QUICK,
        recommended_mode=SelectedMode.QUICK,
        recommendation_reason="Consulta directa.",
        model="qwen3:4b-instruct",
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
        self.assertEqual(events[-1]["completion_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
