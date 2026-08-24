from __future__ import annotations

import unittest

from backend.app.core.config import Settings
from backend.app.domain.intent import SemanticPlan
from backend.app.domain.models import SelectedMode
from backend.app.services.response_policy import (
    response_style_instruction,
    response_token_budget,
)


def plan(task_type: str, *, complexity: float = 0.2) -> SemanticPlan:
    return SemanticPlan(
        literal_request="consulta",
        user_goal="resolver consulta",
        task_type=task_type,
        complexity=complexity,
        confidence=0.9,
    )


class ResponsePolicyTests(unittest.TestCase):
    def test_deep_definition_is_capped_for_cpu_latency(self) -> None:
        budget = response_token_budget(
            Settings(),
            SelectedMode.DEEP,
            plan("definition"),
            "Explicá en tres puntos qué es la carga interna.",
        )
        self.assertEqual(budget, 160)

    def test_explicit_detailed_request_keeps_full_deep_budget(self) -> None:
        budget = response_token_budget(
            Settings(),
            SelectedMode.DEEP,
            plan("planning", complexity=0.9),
            "Explicalo en detalle y paso a paso.",
        )
        self.assertEqual(budget, Settings().deep_max_tokens)

    def test_deep_instruction_separates_reasoning_from_verbosity(self) -> None:
        instruction = response_style_instruction(
            SelectedMode.DEEP,
            plan("interpretation", complexity=0.7),
            "Analizá si esta caída de HSR es relevante.",
        )
        self.assertIn("mayor calidad de razonamiento", instruction)
        self.assertIn("síntesis compacta", instruction)


if __name__ == "__main__":
    unittest.main()
