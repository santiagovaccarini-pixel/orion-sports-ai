from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.services.memory_store import MemoryStore, format_memory_context


class MemoryStoreTests(unittest.TestCase):
    def test_memory_crud_is_explicit_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.json")
            created = store.add_entry("m1", "Zona 5 empieza a 24 km/h", "club")
            self.assertEqual(created.id, "m1")
            self.assertEqual(len(store.list_entries()), 1)
            self.assertEqual(store.get_entry("m1"), created)

            updated = store.update_entry("m1", "Zona 5 empieza a 25 km/h")
            self.assertIsNotNone(updated)
            assert updated is not None
            self.assertEqual(updated.content, "Zona 5 empieza a 25 km/h")

            self.assertTrue(store.delete_entry("m1"))
            self.assertEqual(store.list_entries(), [])

    def test_memory_store_has_no_keyword_search_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.json")
            self.assertFalse(hasattr(store, "search"))

    def test_memory_context_does_not_claim_automatic_relevance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.json")
            entry = store.add_entry("m1", "Dato privado", "usuario")
            context = format_memory_context([entry])
            self.assertIn("plan semántico", context)
            self.assertIn("Dato privado", context)


if __name__ == "__main__":
    unittest.main()
