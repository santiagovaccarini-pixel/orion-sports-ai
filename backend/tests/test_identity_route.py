from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api.routes import require_api_key
from backend.app.core.config import Settings
from backend.app.main import app


class IdentityRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        # These tests exercise routing behavior, not auth; bypass the
        # (now fail-closed) API key dependency instead of configuring a key.
        app.dependency_overrides[require_api_key] = lambda: None

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)

    def test_creator_question_does_not_bypass_model_pipeline_on_chat(self) -> None:
        settings = Settings(diagnostics_enabled=False)
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch(
                "backend.app.api.routes._provider_or_http_error",
                side_effect=RuntimeError("normal_provider_path_reached"),
            ),
            TestClient(app) as client,
        ):
            with self.assertRaisesRegex(RuntimeError, "normal_provider_path_reached"):
                client.post(
                    "/api/v1/chat",
                    json={
                        "messages": [
                            {"role": "user", "content": "¿Quién creó Orion?"}
                        ]
                    },
                )

    def test_creator_question_does_not_bypass_model_pipeline_on_stream(self) -> None:
        settings = Settings(diagnostics_enabled=False)
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch(
                "backend.app.api.routes._provider_or_http_error",
                side_effect=RuntimeError("normal_provider_path_reached"),
            ),
            TestClient(app) as client,
        ):
            with self.assertRaisesRegex(RuntimeError, "normal_provider_path_reached"):
                client.post(
                    "/api/v1/chat/stream",
                    json={
                        "messages": [
                            {"role": "user", "content": "¿Quién te creó?"}
                        ]
                    },
                )


if __name__ == "__main__":
    unittest.main()
