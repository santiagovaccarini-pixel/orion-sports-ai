"""Conversations must survive what the browser cannot: a reload.

The sidebar used to warn "las conversaciones se pierden al recargar la página"
and it was true - the chat was the only stateful part of Orion living solely in
one tab. These tests cover the whole persistence path the interface uses: open
a thread, append finished exchanges, come back later and find them, delete what
should not remain.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api.routes import require_api_key
from backend.app.core.config import Settings
from backend.app.main import app
from backend.app.services.conversation_store import (
    MAX_CONVERSATIONS,
    MAX_MESSAGES_PER_CONVERSATION,
    ConversationStore,
    StoredMessage,
)
from backend.app.services.rate_limit import upload_rate_limiter


class ConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self._temp.name) / "threads.json")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_a_conversation_round_trips_through_the_file(self) -> None:
        self.store.create_conversation("c1", "Carga del plantel", "football")
        self.store.append_messages(
            "c1",
            [
                StoredMessage("user", "¿Qué es la carga interna?"),
                StoredMessage("assistant", "Es la respuesta fisiológica…"),
            ],
        )
        fresh = ConversationStore(self.store.path)
        loaded = fresh.get_conversation("c1")
        assert loaded is not None
        self.assertEqual(loaded.title, "Carga del plantel")
        self.assertEqual(
            [item.role for item in loaded.messages], ["user", "assistant"]
        )
        self.assertEqual(fresh.list_conversations()[0].message_count, 2)

    def test_the_most_recently_touched_conversation_lists_first(self) -> None:
        self.store.create_conversation("vieja", "Primera", "general")
        self.store.create_conversation("nueva", "Segunda", "general")
        self.store.append_messages("vieja", [StoredMessage("user", "hola")])
        listed = self.store.list_conversations()
        self.assertEqual([item.id for item in listed], ["vieja", "nueva"])

    def test_appending_to_a_missing_conversation_reports_it(self) -> None:
        self.assertFalse(
            self.store.append_messages("fantasma", [StoredMessage("user", "hola")])
        )

    def test_the_history_rolls_instead_of_growing_forever(self) -> None:
        for index in range(MAX_CONVERSATIONS + 5):
            self.store.create_conversation(f"c{index}", f"Título {index}", "general")
        listed = self.store.list_conversations()
        self.assertEqual(len(listed), MAX_CONVERSATIONS)
        remaining_ids = {item.id for item in listed}
        # The oldest fell off; the newest are all there.
        self.assertNotIn("c0", remaining_ids)
        self.assertIn(f"c{MAX_CONVERSATIONS + 4}", remaining_ids)

    def test_one_conversation_rolls_its_own_messages_too(self) -> None:
        self.store.create_conversation("c1", "Larga", "general")
        for index in range(MAX_MESSAGES_PER_CONVERSATION + 10):
            self.store.append_messages("c1", [StoredMessage("user", f"m{index}")])
        loaded = self.store.get_conversation("c1")
        assert loaded is not None
        self.assertEqual(len(loaded.messages), MAX_MESSAGES_PER_CONVERSATION)
        # The newest message survived; the write was never refused.
        self.assertEqual(
            loaded.messages[-1].content, f"m{MAX_MESSAGES_PER_CONVERSATION + 9}"
        )


class ConversationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[require_api_key] = lambda: None
        self._temp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            conversations_path=str(Path(self._temp.name) / "threads.json")
        )
        asyncio.run(upload_rate_limiter.reset())

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)
        self._temp.cleanup()
        asyncio.run(upload_rate_limiter.reset())

    def _client(self):
        return (
            patch("backend.app.api.routes.get_settings", return_value=self.settings),
            TestClient(app, raise_server_exceptions=False),
        )

    def test_the_interface_flow_create_append_reload_delete(self) -> None:
        settings_patch, client = self._client()
        with settings_patch, client:
            created = client.post(
                "/api/v1/conversations",
                json={"title": "¿Qué es la PSE?", "sport": "football"},
            )
            self.assertEqual(created.status_code, 200)
            conversation_id = created.json()["id"]

            appended = client.post(
                f"/api/v1/conversations/{conversation_id}/messages",
                json={
                    "messages": [
                        {"role": "user", "content": "¿Qué es la PSE?"},
                        {"role": "assistant", "content": "La percepción subjetiva…"},
                    ]
                },
            )
            self.assertEqual(appended.status_code, 200)

            # What a reload does: list, pick the latest, load its messages.
            listed = client.get("/api/v1/conversations")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()[0]["id"], conversation_id)
            self.assertEqual(listed.json()[0]["message_count"], 2)

            loaded = client.get(f"/api/v1/conversations/{conversation_id}")
            self.assertEqual(loaded.status_code, 200)
            self.assertEqual(
                [item["role"] for item in loaded.json()["messages"]],
                ["user", "assistant"],
            )

            deleted = client.delete(f"/api/v1/conversations/{conversation_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(client.get("/api/v1/conversations").json(), [])

    def test_missing_conversations_answer_404_not_500(self) -> None:
        settings_patch, client = self._client()
        with settings_patch, client:
            self.assertEqual(
                client.get("/api/v1/conversations/no-existe").status_code, 404
            )
            self.assertEqual(
                client.delete("/api/v1/conversations/no-existe").status_code, 404
            )
            self.assertEqual(
                client.post(
                    "/api/v1/conversations/no-existe/messages",
                    json={"messages": [{"role": "user", "content": "hola"}]},
                ).status_code,
                404,
            )

    def test_a_storage_outage_names_itself_and_spares_the_chat(self) -> None:
        """Persistence failing must read as "not saving", never break answering."""

        from backend.app.services.database import DatabaseUnavailableError

        class BrokenStore:
            def list_conversations(self):
                raise DatabaseUnavailableError("connection refused")

        settings_patch, client = self._client()
        with (
            settings_patch,
            patch(
                "backend.app.api.routes.create_conversation_repository",
                return_value=BrokenStore(),
            ),
            client,
        ):
            response = client.get("/api/v1/conversations")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["code"], "conversations_unavailable"
        )
        self.assertNotIn("connection refused", response.text)


if __name__ == "__main__":
    unittest.main()
