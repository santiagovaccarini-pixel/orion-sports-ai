from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.services.knowledge_base import KnowledgeBase
from backend.app.services.knowledge_repository import (
    PostgresKnowledgeBase,
    create_knowledge_base,
)


class KnowledgeRepositorySelectionTests(unittest.TestCase):
    def test_file_store_is_used_when_no_database_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            base = create_knowledge_base(
                database_url=None,
                knowledge_path=str(Path(folder) / "documents.json"),
            )
        self.assertIsInstance(base, KnowledgeBase)
        self.assertNotIsInstance(base, PostgresKnowledgeBase)

    def test_postgres_store_is_used_when_a_database_is_configured(self) -> None:
        base = create_knowledge_base(
            database_url="postgresql://user:pass@example.invalid/db",
            knowledge_path="unused.json",
        )
        self.assertIsInstance(base, PostgresKnowledgeBase)

    def test_empty_database_url_falls_back_to_the_file_store(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            base = create_knowledge_base(
                database_url="",
                knowledge_path=str(Path(folder) / "documents.json"),
            )
        self.assertNotIsInstance(base, PostgresKnowledgeBase)

    def test_search_behaviour_is_shared_not_reimplemented(self) -> None:
        # Chunking and ranking must not fork between the two backends: the
        # Postgres store overrides only storage, so search stays literally the
        # same code and cannot drift into different results.
        self.assertIs(PostgresKnowledgeBase.search, KnowledgeBase.search)
        self.assertIsNot(
            PostgresKnowledgeBase.list_documents, KnowledgeBase.list_documents
        )
        self.assertIsNot(
            PostgresKnowledgeBase.add_document, KnowledgeBase.add_document
        )


if __name__ == "__main__":
    unittest.main()
