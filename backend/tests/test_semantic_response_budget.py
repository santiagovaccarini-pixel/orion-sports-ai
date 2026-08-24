from __future__ import annotations

import unittest

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.services.response_policy import response_token_budget
from backend.app.services.semantic_planner import create_semantic_plan


class SemanticResponseBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_point_explanation_is_a_definition_and_gets_short_deep_budget(self) -> None:
        settings = Settings(semantic_planner_enabled=False)
        query = "Explicá en tres puntos qué es la carga interna en fútbol."
        plan = await create_semantic_plan(
            settings,
            [ChatMessage(role="user", content=query)],
            SportContext.FOOTBALL,
            has_local_documents=False,
        )

        self.assertEqual(plan.task_type, "definition")
        self.assertLessEqual(plan.complexity, 0.35)
        self.assertEqual(
            response_token_budget(settings, SelectedMode.DEEP, plan, query),
            160,
        )


if __name__ == "__main__":
    unittest.main()
