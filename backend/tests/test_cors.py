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


    def test_the_policy_covers_every_method_the_api_actually_answers(self) -> None:
        """A method the API serves but CORS omits fails for a reason nobody can read.

        The DELETE routes (withdrawing a document, forgetting a memory entry)
        were added long after this policy was written, and it still announced
        only GET and POST. Deriving the expectation from the routes rather than
        restating a list keeps the two from drifting apart again.
        """

        from fastapi.routing import APIRoute

        def served(routes, found: set[str]) -> set[str]:
            for route in routes:
                inner = getattr(route, "original_router", None)
                if inner is not None:
                    served(inner.routes, found)
                elif isinstance(route, APIRoute):
                    found |= set(route.methods or ()) - {"HEAD", "OPTIONS"}
            return found

        with TestClient(app) as client:
            response = client.options(
                "/api/v1/memory/entries",
                headers={
                    "Origin": "http://127.0.0.1:5173",
                    "Access-Control-Request-Method": "DELETE",
                    "Access-Control-Request-Headers": "x-orion-api-key",
                },
            )

        self.assertEqual(response.status_code, 200)
        allowed = {
            method.strip().upper()
            for method in response.headers["access-control-allow-methods"].split(",")
        }
        self.assertLessEqual(served(app.routes, set()), allowed)


if __name__ == "__main__":
    unittest.main()