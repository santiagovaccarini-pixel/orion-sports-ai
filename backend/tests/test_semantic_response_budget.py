from __future__ import annotations

import unittest

from backend.app.core.config import Settings
from backend.app.domain.intent import SemanticPlan
from backend.app.domain.models import SelectedMode
from backend.app.services.response_policy import response_token_budget


class SemanticResponseBudgetTests(unittest.TestCase):
    def test_definition_gets_short_deep_budget_from_resolved_task(self) -> None:
        settings = Settings()
        query = "Explicá en tres puntos qué es la carga interna en fútbol."
        plan = SemanticPlan(
            literal_request=query,
            user_goal="definir carga interna",
            task_type="definition",
            inference_type="descriptive",
            concept_ids=["internal_load"],
            concepts=["internal load"],
            complexity=0.22,
            confidence=0.95,
        )

        self.assertEqual(
            response_token_budget(settings, SelectedMode.DEEP, plan, query),
            160,
        )


if __name__ == "__main__":
    unittest.main()
