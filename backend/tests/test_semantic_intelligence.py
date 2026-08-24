from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend.app.core.config import Settings
from backend.app.domain.intent import SemanticPlan
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.services.knowledge_base import KnowledgeBase, KnowledgeDocument
from backend.app.services.semantic_planner import create_semantic_plan
from backend.app.services.semantic_retriever import search_with_intent


class SemanticPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_resolves_reference_to_previous_context_without_network(self) -> None:
        settings = Settings(semantic_planner_enabled=False)
        messages = [
            ChatMessage(role="user", content="Estamos comparando HSR entre dos partidos."),
            ChatMessage(role="assistant", content="Hay que considerar la exposición."),
            ChatMessage(role="user", content="¿Y eso cambia si jugó 25 minutos menos?"),
        ]

        plan = await create_semantic_plan(
            settings,
            messages,
            SportContext.FOOTBALL,
            has_local_documents=False,
        )

        self.assertTrue(plan.referenced_previous_context)
        self.assertGreaterEqual(plan.complexity, 0.4)

    async def test_structured_planner_can_infer_goal_beyond_literal_words(self) -> None:
        settings = Settings(semantic_planner_enabled=True)
        messages = [
            ChatMessage(
                role="user",
                content="Martín corrió bastante menos que Lucas. ¿Estuvo peor?",
            )
        ]
        structured = {
            "literal_request": "comparar a Martín y Lucas",
            "user_goal": "determinar si menor demanda externa implica peor rendimiento",
            "domain": "physical_performance",
            "task_type": "interpretation",
            "concepts": ["external load", "match exposure", "physical performance"],
            "retrieval_queries": [
                "interpretación de carga externa según exposición y contexto",
                "distancia total no equivale a rendimiento físico",
            ],
            "missing_variables": ["minutos jugados", "posición", "contexto táctico"],
            "needs_global_knowledge": True,
            "needs_private_memory": False,
            "needs_local_data": False,
            "needs_web": False,
            "comparison": True,
            "causal_claim_risk": True,
            "requires_clarification": False,
            "referenced_previous_context": False,
            "ambiguity": 0.35,
            "complexity": 0.78,
            "confidence": 0.93,
        }

        with patch(
            "backend.app.services.semantic_planner.OllamaClient.structured_json",
            new=AsyncMock(return_value=structured),
        ):
            plan = await create_semantic_plan(
                settings,
                messages,
                SportContext.FOOTBALL,
                has_local_documents=False,
            )

        self.assertEqual(plan.domain, "physical_performance")
        self.assertEqual(plan.task_type, "interpretation")
        self.assertTrue(plan.causal_claim_risk)
        self.assertIn("external load", plan.concepts)
        self.assertIn("minutos jugados", plan.missing_variables)


class SemanticRetrieverTests(unittest.TestCase):
    def test_intent_expansion_finds_relevant_document_without_literal_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = KnowledgeBase(Path(directory) / "knowledge.json")
            base.add_document(
                KnowledgeDocument(
                    id="physical-1",
                    name="principios_carga.md",
                    content=(
                        "La carga externa describe el trabajo realizado. La interpretación "
                        "de high-speed running y distancia total debe considerar exposición, "
                        "posición y contexto del partido. Una métrica aislada no equivale a "
                        "rendimiento físico global."
                    ),
                )
            )
            plan = SemanticPlan(
                literal_request="¿Estuvo peor?",
                user_goal="determinar si menor carrera implica peor rendimiento",
                domain="physical_performance",
                task_type="interpretation",
                concepts=["external load", "high-speed running", "match exposure"],
                retrieval_queries=[
                    "carga externa high-speed running exposición rendimiento físico"
                ],
                needs_global_knowledge=True,
                complexity=0.7,
                confidence=0.9,
            )

            results = search_with_intent(base, "¿Estuvo peor?", plan, limit=5)

        self.assertTrue(results)
        self.assertEqual(results[0].document_name, "principios_carga.md")


if __name__ == "__main__":
    unittest.main()
