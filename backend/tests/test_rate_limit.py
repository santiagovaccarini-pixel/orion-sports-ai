from __future__ import annotations

import asyncio
import hashlib
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api.routes import require_api_key
from backend.app.core.config import Settings
from backend.app.main import app
from backend.app.services.knowledge_base import KnowledgeDocument
from backend.app.services.rate_limit import (
    MAX_TRACKED_CLIENTS,
    SlidingWindowRateLimiter,
    chat_rate_limiter,
    upload_rate_limiter,
)


class SlidingWindowRateLimiterTests(unittest.TestCase):
    def test_allows_up_to_the_limit_then_refuses_with_retry_after(self) -> None:
        limiter = SlidingWindowRateLimiter()

        async def scenario() -> None:
            for _ in range(3):
                decision = await limiter.check("1.2.3.4", limit=3)
                self.assertTrue(decision.allowed)
            blocked = await limiter.check("1.2.3.4", limit=3)
            self.assertFalse(blocked.allowed)
            self.assertGreater(blocked.retry_after_seconds, 0)

        asyncio.run(scenario())

    def test_clients_are_tracked_independently(self) -> None:
        limiter = SlidingWindowRateLimiter()

        async def scenario() -> None:
            await limiter.check("1.1.1.1", limit=1)
            exhausted = await limiter.check("1.1.1.1", limit=1)
            self.assertFalse(exhausted.allowed)
            other = await limiter.check("2.2.2.2", limit=1)
            self.assertTrue(other.allowed)

        asyncio.run(scenario())

    def test_old_hits_leave_the_window(self) -> None:
        limiter = SlidingWindowRateLimiter(window_seconds=0.05)

        async def scenario() -> None:
            self.assertTrue((await limiter.check("1.1.1.1", limit=1)).allowed)
            self.assertFalse((await limiter.check("1.1.1.1", limit=1)).allowed)
            await asyncio.sleep(0.06)
            self.assertTrue((await limiter.check("1.1.1.1", limit=1)).allowed)

        asyncio.run(scenario())

    def test_tracked_client_count_stays_bounded(self) -> None:
        limiter = SlidingWindowRateLimiter()

        async def scenario() -> None:
            for index in range(MAX_TRACKED_CLIENTS + 50):
                await limiter.check(f"10.0.0.{index}", limit=5)
            self.assertLessEqual(len(limiter._hits), MAX_TRACKED_CLIENTS)

        asyncio.run(scenario())


class RateLimitedRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[require_api_key] = lambda: None
        asyncio.run(chat_rate_limiter.reset())
        asyncio.run(upload_rate_limiter.reset())

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)
        asyncio.run(chat_rate_limiter.reset())
        asyncio.run(upload_rate_limiter.reset())

    def test_chat_stream_returns_429_once_the_per_minute_limit_is_exceeded(
        self,
    ) -> None:
        settings = Settings(rate_limit_chat_per_minute=2)
        payload = {"messages": [{"role": "user", "content": "Hola"}]}
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch(
                "backend.app.api.routes._provider_or_http_error",
                side_effect=RuntimeError("reached_provider"),
            ),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            # The first calls get past the limiter and fail deeper in the stack
            # (no provider configured in tests); only the status code matters.
            for _ in range(2):
                allowed = client.post("/api/v1/chat/stream", json=payload)
                self.assertNotEqual(allowed.status_code, 429)

            blocked = client.post("/api/v1/chat/stream", json=payload)

        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.json()["detail"]["code"], "rate_limited")
        self.assertIn("Retry-After", blocked.headers)

    def test_knowledge_upload_is_rate_limited_separately_from_chat(self) -> None:
        settings = Settings(rate_limit_uploads_per_minute=1)
        document = {"name": "notas.txt", "content": "contenido de prueba"}
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch("backend.app.api.routes.KnowledgeBase") as knowledge_base,
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            knowledge_base.return_value.list_documents.return_value = []
            first = client.post("/api/v1/knowledge/documents", json=document)
            second = client.post("/api/v1/knowledge/documents", json=document)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)


class KnowledgeQuotaTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[require_api_key] = lambda: None
        asyncio.run(upload_rate_limiter.reset())

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)
        asyncio.run(upload_rate_limiter.reset())

    def test_upload_is_refused_once_the_document_count_limit_is_reached(self) -> None:
        settings = Settings(knowledge_max_documents=2)
        stored = [
            KnowledgeDocument("aaa", "uno.txt", "contenido"),
            KnowledgeDocument("bbb", "dos.txt", "contenido"),
        ]
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch("backend.app.api.routes.KnowledgeBase") as knowledge_base,
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            knowledge_base.return_value.list_documents.return_value = stored
            response = client.post(
                "/api/v1/knowledge/documents",
                json={"name": "tres.txt", "content": "documento nuevo"},
            )
            knowledge_base.return_value.add_document.assert_not_called()

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            response.json()["detail"]["code"], "knowledge_document_limit_reached"
        )

    def test_upload_is_refused_once_total_storage_would_be_exceeded(self) -> None:
        settings = Settings(knowledge_max_total_characters=50)
        stored = [KnowledgeDocument("aaa", "uno.txt", "x" * 45)]
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch("backend.app.api.routes.KnowledgeBase") as knowledge_base,
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            knowledge_base.return_value.list_documents.return_value = stored
            response = client.post(
                "/api/v1/knowledge/documents",
                json={"name": "dos.txt", "content": "y" * 20},
            )
            knowledge_base.return_value.add_document.assert_not_called()

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            response.json()["detail"]["code"], "knowledge_storage_limit_reached"
        )

    def test_replacing_an_existing_document_is_allowed_at_the_count_limit(self) -> None:
        # Re-uploading identical content yields the same id, so it replaces
        # rather than adds - the quota must not block that.
        settings = Settings(knowledge_max_documents=1)
        document_id = hashlib.sha256(
            "notas.txt\0contenido".encode("utf-8")
        ).hexdigest()[:16]
        stored = [KnowledgeDocument(document_id, "notas.txt", "contenido")]
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch("backend.app.api.routes.KnowledgeBase") as knowledge_base,
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            knowledge_base.return_value.list_documents.return_value = stored
            response = client.post(
                "/api/v1/knowledge/documents",
                json={"name": "notas.txt", "content": "contenido"},
            )
            knowledge_base.return_value.add_document.assert_called_once()

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
