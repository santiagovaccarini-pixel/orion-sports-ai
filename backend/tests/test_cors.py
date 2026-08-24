from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app


class CorsTests(unittest.TestCase):
    def test_api_key_header_is_allowed_for_browser_preflight_on_local_ports(self) -> None:
        with TestClient(app) as client:
            responses = [
                client.options(
                    "/api/v1/chat/stream",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type,x-orion-api-key",
                    },
                )
                for origin in (
                    "http://127.0.0.1:5173",
                    "http://127.0.0.1:5174",
                )
            ]

        for response in responses:
            self.assertEqual(response.status_code, 200)
            self.assertIn("x-orion-api-key", response.headers["access-control-allow-headers"].lower())


if __name__ == "__main__":
    unittest.main()