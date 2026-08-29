from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.services.memory_repository import (
    PostgresMemoryStore,
    create_memory_repository,
)
from backend.app.services.memory_store import MemoryStore


class MemoryRepositorySelectionTests(unittest.TestCase):
    def test_file_store_is_used_when_no_database_is_configured(self) -> None:
        # Local development and the test suite run with no database at all;
        # memory must keep working there rather than failing to start.
        with tempfile.TemporaryDirectory() as folder:
            repository = create_memory_repository(
                database_url=None,
                memory_path=str(Path(folder) / "entries.json"),
            )
        self.assertIsInstance(repository, MemoryStore)

    def test_postgres_store_is_used_when_a_database_is_configured(self) -> None:
        repository = create_memory_repository(
            database_url="postgresql://user:pass@example.invalid/db",
            memory_path="unused.json",
        )
        self.assertIsInstance(repository, PostgresMemoryStore)

    def test_empty_database_url_falls_back_to_the_file_store(self) -> None:
        # An unset env var arrives as "" through the config layer; that must
        # mean "no database", not a broken connection string.
        with tempfile.TemporaryDirectory() as folder:
            repository = create_memory_repository(
                database_url="",
                memory_path=str(Path(folder) / "entries.json"),
            )
        self.assertIsInstance(repository, MemoryStore)

    def test_both_backends_expose_the_same_operations(self) -> None:
        # The repository protocol only helps if the file store really satisfies
        # it - otherwise swapping storage breaks at runtime, not at import.
        operations = ("list_entries", "get_entry", "add_entry", "delete_entry", "delete_all")
        for name in operations:
            with self.subTest(operation=name):
                self.assertTrue(callable(getattr(MemoryStore, name, None)))
                self.assertTrue(callable(getattr(PostgresMemoryStore, name, None)))


if __name__ == "__main__":
    unittest.main()
