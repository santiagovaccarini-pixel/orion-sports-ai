from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api.routes import require_api_key
from backend.app.core.config import Settings
from backend.app.core.prompt import build_system_prompt
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.main import app
from backend.app.services.database import DatabaseUnavailableError
from backend.app.services.memory_store import MemoryStore, format_memory_context
from backend.app.services.rate_limit import upload_rate_limiter
from backend.app.services.semantic_orchestrator import (
    EvidenceReview,
    conservative_fallback_plan,
    format_reasoning_context,
)


class MemoryRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[require_api_key] = lambda: None
        self._temp = tempfile.TemporaryDirectory()
        self.settings = Settings(memory_path=str(Path(self._temp.name) / "mem.json"))
        asyncio.run(upload_rate_limiter.reset())

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)
        self._temp.cleanup()
        asyncio.run(upload_rate_limiter.reset())

    def test_entries_can_be_created_listed_and_deleted(self) -> None:
        with (
            patch("backend.app.api.routes.get_settings", return_value=self.settings),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            created = client.post(
                "/api/v1/memory/entries",
                json={"content": "Trabajo con el plantel sub-20", "category": "contexto"},
            )
            self.assertEqual(created.status_code, 200)
            entry_id = created.json()["id"]
            self.assertEqual(created.json()["category"], "contexto")

            listed = client.get("/api/v1/memory/entries")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(len(listed.json()), 1)

            deleted = client.delete(f"/api/v1/memory/entries/{entry_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(client.get("/api/v1/memory/entries").json(), [])

    def test_deleting_a_missing_entry_reports_not_found(self) -> None:
        with (
            patch("backend.app.api.routes.get_settings", return_value=self.settings),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.delete("/api/v1/memory/entries/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_delete_all_clears_every_entry(self) -> None:
        with (
            patch("backend.app.api.routes.get_settings", return_value=self.settings),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            for index in range(3):
                client.post(
                    "/api/v1/memory/entries",
                    json={"content": f"dato {index}", "category": "general"},
                )
            self.assertEqual(len(client.get("/api/v1/memory/entries").json()), 3)

            client.delete("/api/v1/memory/entries")
            self.assertEqual(client.get("/api/v1/memory/entries").json(), [])

    def test_entry_count_is_capped(self) -> None:
        settings = Settings(memory_path=self.settings.memory_path, memory_max_entries=2)
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            for index in range(2):
                ok = client.post(
                    "/api/v1/memory/entries",
                    json={"content": f"dato {index}", "category": "general"},
                )
                self.assertEqual(ok.status_code, 200)
            refused = client.post(
                "/api/v1/memory/entries",
                json={"content": "uno de más", "category": "general"},
            )
        self.assertEqual(refused.status_code, 413)
        self.assertEqual(refused.json()["detail"]["code"], "memory_limit_reached")


class MemoryFailureTests(unittest.TestCase):
    """A storage outage must degrade, not take the whole product down."""

    def setUp(self) -> None:
        app.dependency_overrides[require_api_key] = lambda: None

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)

    def test_listing_reports_storage_outage_instead_of_a_generic_error(self) -> None:
        class BrokenStore:
            def list_entries(self):
                raise DatabaseUnavailableError("connection refused")

        with (
            patch("backend.app.api.routes._memory_store", return_value=BrokenStore()),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.get("/api/v1/memory/entries")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "memory_unavailable")

    def test_chat_still_answers_when_memory_storage_is_down(self) -> None:
        # Memory enriches an answer; losing it must never mean losing the answer.
        from backend.app.api.routes import _memory_context

        class BrokenStore:
            def list_entries(self):
                raise DatabaseUnavailableError("connection refused")

        with patch("backend.app.api.routes._memory_store", return_value=BrokenStore()):
            context = asyncio.run(_memory_context())

        self.assertEqual(context, "")


class MemoryContextTests(unittest.TestCase):
    def test_saved_entries_reach_the_final_answer_context(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(Path(folder) / "mem.json")
            store.add_entry("a1", "El equipo entrena martes y jueves", "rutina")
            memory_context = format_memory_context(store.list_entries())

        plan = conservative_fallback_plan(
            [ChatMessage(role="user", content="Pregunta")],
            web_available=False,
            documents=[],
        )
        review = EvidenceReview(
            sufficient=True,
            relevant_source_ids=(),
            discarded_source_ids=(),
            missing_information=(),
            follow_up_web_query=None,
            needs_clarification=False,
            clarifying_question=None,
            resolved_scope=None,
            reason="ok",
        )
        context = format_reasoning_context(
            plan, review, [], [], memory_context=memory_context
        )

        self.assertIn("El equipo entrena martes y jueves", context)
        self.assertIn("MEMORIA PERSONAL DEL USUARIO", context)

    def test_no_saved_entries_adds_nothing(self) -> None:
        self.assertEqual(format_memory_context([]), "")

    def test_system_prompt_no_longer_denies_having_memory(self) -> None:
        # The prompt used to instruct Orion to state it had no persistent
        # memory, which would have made it contradict its own saved entries.
        prompt = build_system_prompt(SportContext.GENERAL, SelectedMode.QUICK)
        self.assertNotIn("no posee memoria", prompt)
        self.assertIn("solo contiene lo que el usuario pidió", prompt)


if __name__ == "__main__":
    unittest.main()
