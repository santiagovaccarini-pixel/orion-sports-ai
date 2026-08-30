from __future__ import annotations

import unittest

from backend.app.services.memory_suggestions import (
    MAX_SUGGESTIONS,
    SUGGESTION_PROMPT,
    parse_suggestions,
)


class MemorySuggestionTests(unittest.TestCase):
    """Orion proposes; the person decides. Nothing here writes to memory."""

    def test_a_well_formed_proposal_is_read(self) -> None:
        suggestions = parse_suggestions(
            '{"suggestions":[{"content":"Santiago trabaja en Atlético Mineiro.",'
            '"reason":"cambia el contexto de futuras respuestas"}]}'
        )
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].content, "Santiago trabaja en Atlético Mineiro.")
        self.assertIn("contexto", suggestions[0].reason)

    def test_a_fenced_empty_list_proposes_nothing(self) -> None:
        self.assertEqual(parse_suggestions('```json\n{"suggestions": []}\n```'), ())

    def test_unparseable_output_proposes_nothing(self) -> None:
        """Losing a proposal costs a sentence the user can still type by hand.

        Guessing at one costs them a memory they never approved, so anything
        malformed is simply not offered.
        """

        for raw in ("no es json", "", "{}", '{"suggestions": "texto"}'):
            with self.subTest(raw=raw):
                self.assertEqual(parse_suggestions(raw), ())

    def test_blank_entries_are_dropped_and_the_count_is_bounded(self) -> None:
        suggestions = parse_suggestions(
            '{"suggestions":[{"content":"   "},{"content":"a"},{"content":"b"},'
            '{"content":"c"}]}'
        )
        self.assertLessEqual(len(suggestions), MAX_SUGGESTIONS)
        self.assertNotIn("", [item.content for item in suggestions])

    def test_the_prompt_asks_for_durable_facts_and_expects_none(self) -> None:
        """The default answer is an empty list; a tool that proposes constantly
        trains the person to accept without reading."""

        self.assertIn("NINGUNA", SUGGESTION_PROMPT)
        self.assertIn("duraderos", SUGGESTION_PROMPT)
        self.assertIn("no haya afirmado él mismo", SUGGESTION_PROMPT)


if __name__ == "__main__":
    unittest.main()
