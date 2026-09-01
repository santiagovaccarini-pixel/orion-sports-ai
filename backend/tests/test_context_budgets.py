"""The evidence the pipeline builds must be the evidence the model receives.

Three limits stack on the reviewer's input: the size this module builds
(MAX_REVIEW_INPUT_CHARACTERS), the message validator (ChatMessage.max_length)
and the provider's transport trim (quick_history_characters). Only the first is
visible from the orchestrator; the other two sit in different files and win
silently. That inversion shipped: the trim was 12.000 for every provider, so
the reviewer - which packs conversation, plan and every fetched page into one
quick-mode message - was judging on the first 17% of its input. The evidence
lives at the tail, so the trim kept the preamble and discarded the pages.

Nothing errored. Answers still improved, because the final stage gets the
evidence through the system prompt, which is never trimmed. Only the quality
gate was blind, which is the one place blindness cannot be seen from outside.

These tests pin the ordering so a future resize of any one limit cannot
quietly invert it again.
"""

from __future__ import annotations

import os
import unittest

from backend.app.core.config import ENDPOINT_DEFAULTS, get_settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import MAX_MESSAGE_CHARACTERS, ChatMessage
from backend.app.providers.openai_compatible import build_chat_payload
from backend.app.services.semantic_orchestrator import (
    MAX_REVIEW_INPUT_CHARACTERS,
    MAX_REVIEW_MEMORY_CHARACTERS,
    REVIEW_DEEPENED_SOURCE_CLIP,
    EvidenceReview,
    LocalEvidence,
    _review_input,
    conservative_fallback_plan,
    format_reasoning_context,
)
from backend.app.services.web_research import WebSource


def _cloud_settings(provider: str = "cerebras"):
    """Resolve settings exactly as production does, for a cloud provider."""

    _base, _model, key_env = ENDPOINT_DEFAULTS.get(
        provider, ENDPOINT_DEFAULTS["cerebras"]
    )
    touched = {"ORION_MODEL_PROVIDER": provider, key_env: "clave-de-prueba"}
    previous = {key: os.environ.get(key) for key in touched}
    os.environ.update(touched)
    get_settings.cache_clear()
    try:
        return get_settings()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


def _plan():
    return conservative_fallback_plan(
        [ChatMessage(role="user", content="Pregunta")],
        web_available=True,
        documents=[],
    )


def _review(**overrides):
    values = dict(
        sufficient=True,
        relevant_source_ids=("W1",),
        discarded_source_ids=(),
        missing_information=(),
        follow_up_web_query=None,
        needs_clarification=False,
        clarifying_question=None,
        resolved_scope=None,
        reason="ok",
    )
    values.update(overrides)
    return EvidenceReview(**values)


class TransportBudgetTests(unittest.TestCase):
    def test_every_cloud_provider_can_carry_a_full_review_input(self) -> None:
        for provider in (*ENDPOINT_DEFAULTS, "cloudflare"):
            with self.subTest(provider=provider):
                settings = _cloud_settings(provider)
                self.assertGreaterEqual(
                    settings.quick_history_characters,
                    MAX_REVIEW_INPUT_CHARACTERS,
                    "the transport trim is smaller than the review input: the "
                    "reviewer will silently judge on a fraction of the evidence",
                )
                self.assertGreaterEqual(
                    settings.deep_history_characters,
                    settings.quick_history_characters,
                )

    def test_the_message_validator_can_carry_a_full_review_input(self) -> None:
        self.assertGreaterEqual(MAX_MESSAGE_CHARACTERS, MAX_REVIEW_INPUT_CHARACTERS)

    def test_the_local_budgets_stay_small_for_the_cpu_engine(self) -> None:
        """The fix must not quietly hand a 4B model on a laptop a 120k prompt."""

        previous = os.environ.get("ORION_MODEL_PROVIDER")
        os.environ["ORION_MODEL_PROVIDER"] = "ollama"
        get_settings.cache_clear()
        try:
            settings = get_settings()
        finally:
            if previous is None:
                os.environ.pop("ORION_MODEL_PROVIDER", None)
            else:
                os.environ["ORION_MODEL_PROVIDER"] = previous
            get_settings.cache_clear()
        self.assertEqual(settings.quick_history_characters, 12_000)
        self.assertEqual(settings.deep_history_characters, 30_000)

    def test_a_full_review_message_survives_the_cloud_trim_intact(self) -> None:
        settings = _cloud_settings()
        review_message = "x" * MAX_REVIEW_INPUT_CHARACTERS
        payload = build_chat_payload(
            model="gpt-oss-120b",
            messages=[ChatMessage(role="user", content=review_message)],
            system_prompt="prompt",
            max_tokens=1536,
            history_characters=settings.quick_history_characters,
            stream=False,
            structured=True,
        )
        sent = payload["messages"][1]["content"]
        self.assertEqual(len(sent), MAX_REVIEW_INPUT_CHARACTERS)


class ReviewInputBudgetTests(unittest.TestCase):
    def test_a_fat_memory_cannot_evict_the_evidence(self) -> None:
        """Memory shares the input with the pages; the pages must win.

        The final clip cuts from the tail and local evidence deliberately sits
        at the tail, so before the memory clip existed, every character of
        saved notes beyond budget evicted a character of evidence.
        """

        memory = "MEMORIA PERSONAL DEL USUARIO:\n" + (
            "- [general] Una nota guardada bastante larga sobre el plantel.\n" * 2_000
        )
        evidence = [
            LocalEvidence("L1", "gps.csv", "EVIDENCIA_LOCAL_MARCADOR " * 60, False, 0)
        ]
        sources = [
            WebSource("Título", "https://x.test/a", "EVIDENCIA_WEB_MARCADOR " * 80, "x.test")
        ]
        text = _review_input(_plan(), sources, evidence, memory_context=memory)
        self.assertLessEqual(len(text), MAX_REVIEW_INPUT_CHARACTERS)
        self.assertIn("EVIDENCIA_WEB_MARCADOR", text)
        self.assertIn("EVIDENCIA_LOCAL_MARCADOR", text)
        # The memory arrived, but bounded.
        self.assertIn("MEMORIA PERSONAL", text)
        memory_portion = text.split("Tratá esta memoria")[0]
        self.assertLessEqual(len(memory_portion), MAX_REVIEW_MEMORY_CHARACTERS + 6_000)


class FinalStageBudgetTests(unittest.TestCase):
    def test_the_answer_stage_reads_at_most_what_the_review_audited(self) -> None:
        """Text beyond the review clip is text no review ever saw.

        The reviewer accepted W1 having read at most 16k of it; shipping the
        full 24k to the answer stage meant ~8k of unaudited page per source
        feeding the final wording.
        """

        deepened = WebSource(
            "Título",
            "https://x.test/a",
            "y" * 24_000,
            "x.test",
            deepened=True,
        )
        context = format_reasoning_context(
            _plan(),
            _review(),
            [deepened],
            [],
            original_user_request="Pregunta",
        )
        # The excerpt block, not the whole context, carries the clip: measure it.
        excerpt = context.split("Extracto: ")[1]
        self.assertLessEqual(
            len(excerpt.split("\n\n")[0]),
            REVIEW_DEEPENED_SOURCE_CLIP + 200,
        )

    def test_memory_is_bounded_in_the_final_stage_too(self) -> None:
        memory = "MEMORIA PERSONAL DEL USUARIO:\n" + ("- [general] nota\n" * 5_000)
        context = format_reasoning_context(
            _plan(),
            _review(relevant_source_ids=()),
            [],
            [],
            original_user_request="Pregunta",
            memory_context=memory,
        )
        self.assertLess(len(context), len(memory))


if __name__ == "__main__":
    unittest.main()
