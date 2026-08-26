from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import app


class IdentityRouteTests(unittest.TestCase):
    def test_creator_question_bypasses_model_provider_on_chat(self) -> None:
        settings = Settings(diagnostics_enabled=False)
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch(
                "backend.app.api.routes._provider_or_http_error",
                side_effect=AssertionError("No debe crear proveedor para identidad"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat",
                json={"messages": [{"role": "user", "content": "¿Quién creó Orion?"}]},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("Santiago Vaccarini", payload["content"])
        self.assertEqual(payload["model"], "orion-institutional-identity")
        self.assertEqual(payload["selected_mode"], "quick")

    def test_creator_question_streams_without_model_provider(self) -> None:
        settings = Settings(diagnostics_enabled=False)
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch(
                "backend.app.api.routes._provider_or_http_error",
                side_effect=AssertionError("No debe crear proveedor para identidad"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "¿Quién te creó?"}]},
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[0]["model"], "orion-institutional-identity")
        self.assertIn("Santiago Vaccarini", events[1]["content"])
        self.assertEqual(events[-1]["type"], "done")


if __name__ == "__main__":
    unittest.main()
