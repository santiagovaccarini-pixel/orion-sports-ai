from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.index import app
from backend.app.providers.cloudflare import CloudflareStreamEvent


async def fake_cloud_stream(*_args, **_kwargs):
    yield CloudflareStreamEvent(content="Hola ", done=False)
    yield CloudflareStreamEvent(content="desde la nube", done=False)
    yield CloudflareStreamEvent(
        content="",
        done=True,
        total_duration_ms=420.0,
        eval_duration_ms=420.0,
        prompt_tokens=25,
        completion_tokens=7,
        tokens_per_second=16.67,
    )


class CloudEntrypointTests(unittest.TestCase):
    def test_health_does_not_require_model_credentials(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "cloudflare")

    def test_stream_keeps_frontend_ndjson_contract(self) -> None:
        with (
            patch("api.index.CloudflareClient.chat_stream", new=fake_cloud_stream),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "Hola"}],
                    "mode": "quick",
                    "sport": "football",
                },
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("application/x-ndjson")
        )
        self.assertEqual(
            [event["type"] for event in events],
            ["meta", "content", "content", "done"],
        )
        self.assertEqual(events[0]["sport"], "football")
        self.assertEqual(events[-1]["completion_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
