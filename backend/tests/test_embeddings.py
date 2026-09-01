"""Ranking by meaning, and the fallbacks that keep it optional.

Word overlap fails exactly where a person needs retrieval most: the row that
answers "¿cuánto corrió Pérez?" reads "Perez,9800,420" and shares no word with
the question, while a paragraph repeating "cuánto" wins on the count. These
tests use a stand-in embedder with hand-placed vectors, so what is verified is
Orion's own arithmetic and its fallbacks - not a remote model's quality, which
no test here could honestly assert.
"""

from __future__ import annotations

import asyncio
import unittest

from backend.app.core.config import Settings
from backend.app.services.embeddings import (
    NullEmbeddings,
    cosine_similarity,
    create_embedding_provider,
    rank_by_meaning,
    rank_with_timeout,
)
from backend.app.services.knowledge_base import KnowledgeDocument
from backend.app.services.local_retrieval import (
    retrieve_local_chunks,
    retrieve_local_chunks_by_meaning,
)


class FakeEmbeddings:
    """Places each text by which keyword it carries, so meaning is decidable.

    Nothing lexical reaches Orion here: the mapping lives in the test, and the
    code under test only ever sees vectors.
    """

    available = True

    def __init__(self, placements: dict[str, tuple[float, ...]], *, fail: bool = False):
        self.placements = placements
        self.fail = fail
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if self.fail:
            return ()
        vectors = []
        for text in texts:
            for marker, vector in self.placements.items():
                if marker in text:
                    vectors.append(vector)
                    break
            else:
                vectors.append((0.0, 0.0, 1.0))
        return tuple(vectors)


class SimilarityTests(unittest.TestCase):
    def test_identical_directions_score_one_and_opposite_ones_minus_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (1.0, 0.0)), 1.0)
        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (-1.0, 0.0)), -1.0)
        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (0.0, 1.0)), 0.0)

    def test_length_does_not_decide(self) -> None:
        """A long chunk must not outrank a short one for having more words."""

        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (9.0, 0.0)), 1.0)

    def test_a_missing_or_mismatched_vector_scores_zero_instead_of_raising(self) -> None:
        self.assertEqual(cosine_similarity((), (1.0, 0.0)), 0.0)
        self.assertEqual(cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0)), 0.0)
        self.assertEqual(cosine_similarity((0.0, 0.0), (1.0, 0.0)), 0.0)


class RankingTests(unittest.TestCase):
    def test_the_closest_meanings_come_first(self) -> None:
        provider = FakeEmbeddings(
            {
                "distancia": (1.0, 0.0, 0.0),
                "9800": (1.0, 0.0, 0.0),
                "lesion": (0.0, 1.0, 0.0),
            }
        )
        order = asyncio.run(
            rank_by_meaning(
                provider,
                "cuanta distancia recorrio",
                ["parte de lesion", "Perez 9800 metros"],
                limit=1,
            )
        )
        self.assertEqual(order, (1,))

    def test_ties_keep_the_original_order(self) -> None:
        """The same question twice must not reshuffle the evidence underneath."""

        provider = FakeEmbeddings({})
        order = asyncio.run(
            rank_by_meaning(provider, "algo", ["uno", "dos", "tres"], limit=3)
        )
        self.assertEqual(order, (0, 1, 2))

    def test_an_unavailable_provider_returns_none_not_an_empty_list(self) -> None:
        """None means "could not rank"; () would mean "nothing is relevant".

        A caller that cannot tell those apart drops all the evidence the moment
        the embedding service has a bad minute.
        """

        order = asyncio.run(
            rank_by_meaning(FakeEmbeddings({}, fail=True), "algo", ["uno"], limit=1)
        )
        self.assertIsNone(order)

    def test_slow_ranking_is_abandoned_rather_than_extending_the_wait(self) -> None:
        class SlowEmbeddings:
            available = True

            async def embed(self, texts):
                await asyncio.sleep(5)
                return ()

        order = asyncio.run(
            rank_with_timeout(
                SlowEmbeddings(), "algo", ["uno"], limit=1, timeout_seconds=0.05
            )
        )
        self.assertIsNone(order)


class ProviderSelectionTests(unittest.TestCase):
    def test_it_stays_off_until_it_is_enabled(self) -> None:
        provider = create_embedding_provider(Settings())
        self.assertIsInstance(provider, NullEmbeddings)
        self.assertFalse(provider.available)

    def test_enabling_it_without_credentials_falls_back_instead_of_failing(self) -> None:
        provider = create_embedding_provider(Settings(embeddings_enabled=True))
        self.assertIsInstance(provider, NullEmbeddings)

    def test_the_null_provider_returns_nothing_rather_than_raising(self) -> None:
        self.assertEqual(asyncio.run(NullEmbeddings().embed(["algo"])), ())


class LocalRetrievalTests(unittest.TestCase):
    """The case word overlap gets wrong, and the fallbacks around it."""

    def _document(self) -> KnowledgeDocument:
        # The answering row sits in the middle on purpose. At either end, the
        # lexical retriever's positional fallback would reach it by luck rather
        # than by relevance, and the comparison would prove nothing.
        rows = ["Jugador,Distancia,HSR"]
        rows.extend(f"Relleno {index},0,0" for index in range(30))
        rows.append("Perez,9800,420")
        rows.extend(f"Relleno {index},0,0" for index in range(30, 80))
        return KnowledgeDocument("1", "gps.csv", "\n".join(rows))

    def test_meaning_finds_the_row_that_word_overlap_misses(self) -> None:
        document = self._document()
        # The way a coach actually asks. The row says "Perez", "9800": not one
        # word of the question appears in the answer, which is the whole
        # problem with counting shared words.
        question = "cuanta carga tuvo el delantero"
        lexical = retrieve_local_chunks([document], question, max_chunks=3)
        self.assertFalse(any("9800" in chunk.content for chunk in lexical))

        provider = FakeEmbeddings(
            {"delantero": (1.0, 0.0, 0.0), "Perez": (1.0, 0.0, 0.0)}
        )
        ranked = asyncio.run(
            retrieve_local_chunks_by_meaning(
                [document],
                question,
                provider=provider,
                timeout_seconds=5.0,
                max_chunks=3,
            )
        )
        self.assertTrue(any("9800" in chunk.content for chunk in ranked))

    def test_without_a_provider_it_behaves_exactly_as_before(self) -> None:
        document = self._document()
        question = "cuanto corrio Perez"
        lexical = retrieve_local_chunks([document], question, max_chunks=3)
        fallback = asyncio.run(
            retrieve_local_chunks_by_meaning(
                [document],
                question,
                provider=NullEmbeddings(),
                timeout_seconds=5.0,
                max_chunks=3,
            )
        )
        self.assertEqual(
            [chunk.content for chunk in fallback],
            [chunk.content for chunk in lexical],
        )

    def test_a_failing_provider_falls_back_instead_of_returning_nothing(self) -> None:
        document = self._document()
        question = "cuanto corrio Perez"
        lexical = retrieve_local_chunks([document], question, max_chunks=3)
        result = asyncio.run(
            retrieve_local_chunks_by_meaning(
                [document],
                question,
                provider=FakeEmbeddings({}, fail=True),
                timeout_seconds=5.0,
                max_chunks=3,
            )
        )
        self.assertEqual(len(result), len(lexical))

    def test_a_short_document_is_not_sent_for_ranking_at_all(self) -> None:
        """Ranking a list that already fits spends a network call to change nothing."""

        provider = FakeEmbeddings({})
        small = KnowledgeDocument("1", "nota.txt", "una sola idea corta")
        asyncio.run(
            retrieve_local_chunks_by_meaning(
                [small],
                "algo",
                provider=provider,
                timeout_seconds=5.0,
                max_chunks=12,
            )
        )
        self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()


class MemoryRankingTests(unittest.TestCase):
    """The named limitation: every saved entry travelled in every question.

    Memory enters three prompts per question, so a working history of cases
    would eventually crowd out the evidence it exists to support. Past a
    threshold the entries closest in meaning are selected - and below it,
    nothing changes, because ranking a short list spends a call to save
    nothing.
    """

    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.app.core.config import Settings

        self._temp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            memory_path=str(Path(self._temp.name) / "mem.json"),
            memory_ranking_threshold=3,
            memory_ranking_keep=2,
        )

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _seed(self, contents: list[str]) -> None:
        from backend.app.services.memory_store import MemoryStore
        from pathlib import Path

        store = MemoryStore(Path(self.settings.memory_path))
        for index, content in enumerate(contents):
            store.add_entry(f"e{index}", content, "general")

    def _context(self, query: str, provider) -> str:
        from unittest.mock import patch

        from backend.app.api.routes import _memory_context

        with (
            patch("backend.app.api.routes.get_settings", return_value=self.settings),
            patch(
                "backend.app.api.routes.create_embedding_provider",
                return_value=provider,
            ),
        ):
            return asyncio.run(_memory_context(query))

    def test_a_short_memory_travels_whole(self) -> None:
        self._seed(["Dirijo el sub-20", "Me gusta el acai"])
        provider = FakeEmbeddings({})
        context = self._context("cualquier cosa", provider)
        self.assertIn("Dirijo el sub-20", context)
        self.assertIn("Me gusta el acai", context)
        self.assertEqual(provider.calls, 0)

    def test_a_long_memory_sends_what_bears_on_the_question(self) -> None:
        self._seed(
            [
                "Dirijo el sub-20 de Atletico Mineiro",
                "Me gusta el acai",
                "Uso GPS Catapult en los entrenamientos",
                "Prefiero informes cortos",
            ]
        )
        provider = FakeEmbeddings(
            {"GPS": (1.0, 0.0, 0.0), "dispositivo": (1.0, 0.0, 0.0)}
        )
        context = self._context("que dispositivo uso para medir carga", provider)
        self.assertIn("GPS Catapult", context)
        # Two kept out of four: the rest did not bear on the question.
        self.assertEqual(context.count("- ["), 2)

    def test_the_kept_entries_stay_in_saved_order(self) -> None:
        """The panel is something the user reads back; a hidden score must not
        reshuffle it differently on every question."""

        self._seed(["primera", "segunda", "tercera", "cuarta"])
        provider = FakeEmbeddings(
            {"cuarta": (1.0, 0.0, 0.0), "primera": (0.9, 0.1, 0.0)}
        )
        context = self._context("primera y cuarta", provider)
        self.assertLess(context.index("primera"), context.index("cuarta"))

    def test_a_ranking_outage_sends_the_whole_memory_rather_than_none(self) -> None:
        self._seed(["uno", "dos", "tres", "cuatro"])
        context = self._context("algo", FakeEmbeddings({}, fail=True))
        for entry in ("uno", "dos", "tres", "cuatro"):
            self.assertIn(entry, context)
