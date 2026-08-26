from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend.app.domain.schemas import ChatRequest
from backend.app.services.reasoning_pipeline import (
    _deepen_relevant_web_sources,
    _web_read_candidates,
)
from backend.app.services.semantic_orchestrator import EvidenceReview
from backend.app.services.web_reader import PageRead
from backend.app.services.web_research import WebSource


def review(*ids: str) -> EvidenceReview:
    return EvidenceReview(
        sufficient=False,
        relevant_source_ids=tuple(ids),
        discarded_source_ids=(),
        missing_information=("Falta el dato explícito.",),
        follow_up_web_query="segunda búsqueda",
        needs_clarification=False,
        clarifying_question=None,
        resolved_scope=None,
        reason="Las fuentes parecen relevantes pero los snippets no alcanzan.",
    )


class ReasoningWebReadTests(unittest.TestCase):
    def test_candidates_follow_reviewer_ids_not_result_position(self) -> None:
        selected = _web_read_candidates(
            review("W3", "W1", "T1", "W2"),
            attempted_source_ids={"W1"},
            remaining=2,
        )
        self.assertEqual(selected, ("W3", "W2"))

    def test_deepening_opens_only_reviewer_selected_page_and_replaces_snippet(self) -> None:
        request = ChatRequest(
            messages=[{"role": "user", "content": "¿Cuál es el total actualizado?"}]
        )
        sources = (
            WebSource("Uno", "https://uno.example/a", "snippet uno", "uno.example"),
            WebSource("Dos", "https://dos.example/b", "snippet dos", "dos.example"),
        )
        page_reads = (
            PageRead(
                source_id="W2",
                title="Dos completo",
                url="https://dos.example/b",
                domain="dos.example",
                excerpt="La página completa confirma el total 123.",
                published_date="2026-08-26",
            ),
        )
        attempted: set[str] = set()

        with patch(
            "backend.app.services.reasoning_pipeline.read_source_pages",
            new=AsyncMock(return_value=page_reads),
        ) as mocked:
            enriched, changed = asyncio.run(
                _deepen_relevant_web_sources(
                    request,
                    review("W2"),
                    sources,
                    attempted_source_ids=attempted,
                    trace=None,
                )
            )

        self.assertTrue(changed)
        self.assertEqual(attempted, {"W2"})
        self.assertEqual(enriched[0].excerpt, "snippet uno")
        self.assertIn("total 123", enriched[1].excerpt)
        mocked.assert_awaited_once()
        self.assertEqual(mocked.await_args.kwargs["source_ids"], ("W2",))

    def test_failed_page_read_is_not_retried_forever(self) -> None:
        request = ChatRequest(messages=[{"role": "user", "content": "dato"}])
        sources = (
            WebSource("Uno", "https://uno.example/a", "snippet", "uno.example"),
        )
        attempted: set[str] = set()

        with patch(
            "backend.app.services.reasoning_pipeline.read_source_pages",
            new=AsyncMock(return_value=()),
        ) as mocked:
            first_sources, first_changed = asyncio.run(
                _deepen_relevant_web_sources(
                    request,
                    review("W1"),
                    sources,
                    attempted_source_ids=attempted,
                    trace=None,
                )
            )
            second_sources, second_changed = asyncio.run(
                _deepen_relevant_web_sources(
                    request,
                    review("W1"),
                    first_sources,
                    attempted_source_ids=attempted,
                    trace=None,
                )
            )

        self.assertFalse(first_changed)
        self.assertFalse(second_changed)
        self.assertEqual(second_sources, sources)
        self.assertEqual(attempted, {"W1"})
        self.assertEqual(mocked.await_count, 1)


if __name__ == "__main__":
    unittest.main()
