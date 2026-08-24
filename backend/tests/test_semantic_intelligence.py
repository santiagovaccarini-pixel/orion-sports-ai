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
    async def _fallback(self, text: str, *, local: bool = False) -> SemanticPlan:
        return await create_semantic_plan(
            Settings(semantic_planner_enabled=False),
            [ChatMessage(role="user", content=text)],
            SportContext.FOOTBALL,
            has_local_documents=local,
        )

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
        self.assertEqual(plan.domain, "physical_performance")
        self.assertEqual(plan.task_type, "interpretation")
        self.assertIn("HSR", plan.concepts)
        self.assertIn("match exposure", plan.concepts)
        self.assertGreaterEqual(plan.complexity, 0.4)

    async def test_fallback_detects_ambiguous_workload_comparison(self) -> None:
        plan = await self._fallback("¿Quién trabajó más?", local=True)

        self.assertTrue(plan.comparison)
        self.assertTrue(plan.needs_local_data)
        self.assertTrue(plan.requires_clarification)
        self.assertGreaterEqual(plan.ambiguity, 0.75)
        self.assertIn("training load", plan.concepts)
        self.assertIn("métrica", " ".join(plan.missing_variables))

    async def test_physical_metric_is_not_silently_equated_with_performance(self) -> None:
        plan = await self._fallback("Martín corrió bastante menos que Lucas. ¿Estuvo peor?")

        self.assertEqual(plan.task_type, "interpretation")
        self.assertEqual(plan.domain, "physical_performance")
        self.assertTrue(plan.comparison)
        self.assertTrue(plan.causal_claim_risk)
        self.assertIn("external load", plan.concepts)
        self.assertIn("physical performance", plan.concepts)
        self.assertIn("match exposure", plan.concepts)

    async def test_private_protocol_reference_routes_to_private_memory(self) -> None:
        plan = await self._fallback(
            "Nosotros usamos sprint por encima de 25 km/h. ¿Cómo lo veníamos interpretando?"
        )

        self.assertTrue(plan.needs_private_memory)
        self.assertTrue(plan.referenced_previous_context)
        self.assertIn("sprint", plan.concepts)
        self.assertIn("speed threshold", plan.concepts)

    async def test_tactical_reference_uses_conversation_not_only_last_words(self) -> None:
        settings = Settings(semantic_planner_enabled=False)
        messages = [
            ChatMessage(role="user", content="Estamos analizando la presión alta del equipo."),
            ChatMessage(role="assistant", content="La presión alta depende del comportamiento colectivo."),
            ChatMessage(role="user", content="¿Y eso cambia si el rival empieza a salir largo?"),
        ]
        plan = await create_semantic_plan(
            settings,
            messages,
            SportContext.FOOTBALL,
            has_local_documents=False,
        )

        self.assertTrue(plan.referenced_previous_context)
        self.assertEqual(plan.task_type, "interpretation")
        self.assertEqual(plan.domain, "tactical_analysis")
        self.assertIn("high press", plan.concepts)
        self.assertIn("long ball", plan.concepts)

    async def test_simple_definition_is_normalized_and_kept_simple(self) -> None:
        plan = await self._fallback("¿Qué significa RPE?")

        self.assertEqual(plan.task_type, "definition")
        self.assertEqual(plan.domain, "internal_load")
        self.assertFalse(plan.needs_web)
        self.assertFalse(plan.causal_claim_risk)
        self.assertLessEqual(plan.complexity, 0.35)
        self.assertIn("RPE", plan.concepts)
        self.assertIn("rating of perceived exertion", plan.concepts)

    async def test_current_load_research_maps_to_monitoring_concepts(self) -> None:
        plan = await self._fallback(
            "Buscá fuentes actuales y estudios recientes sobre monitoreo de carga en fútbol."
        )

        self.assertEqual(plan.task_type, "research")
        self.assertTrue(plan.needs_web)
        self.assertTrue(plan.needs_global_knowledge)
        self.assertIn("training load", plan.concepts)
        self.assertIn("load monitoring", plan.concepts)

    async def test_local_distance_chart_keeps_metric_and_period_semantics(self) -> None:
        plan = await self._fallback(
            "Graficame la distancia total de este jugador por período en el archivo.",
            local=True,
        )

        self.assertEqual(plan.task_type, "chart")
        self.assertTrue(plan.needs_local_data)
        self.assertIn("total distance", plan.concepts)
        self.assertIn("period", plan.concepts)

    async def test_injury_load_question_is_marked_as_causal_inference(self) -> None:
        plan = await self._fallback(
            "Las lesiones aumentaron justo cuando subimos la carga. ¿La carga causó las lesiones?"
        )

        self.assertEqual(plan.task_type, "interpretation")
        self.assertTrue(plan.causal_claim_risk)
        self.assertIn("injury", plan.concepts)
        self.assertIn("training load", plan.concepts)
        self.assertIn("causality", plan.concepts)

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
            "domain": "general",
            "task_type": "comparison",
            "concepts": [],
            "retrieval_queries": [],
            "missing_variables": [],
            "needs_global_knowledge": True,
            "needs_private_memory": False,
            "needs_local_data": False,
            "needs_web": False,
            "comparison": True,
            "causal_claim_risk": False,
            "requires_clarification": False,
            "referenced_previous_context": False,
            "ambiguity": 0.35,
            "complexity": 0.5,
            "confidence": 0.7,
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

        # The normalizer repairs obvious omissions from the small planner model.
        self.assertEqual(plan.domain, "physical_performance")
        self.assertEqual(plan.task_type, "interpretation")
        self.assertTrue(plan.causal_claim_risk)
        self.assertIn("external load", plan.concepts)
        self.assertIn("match exposure", plan.concepts)
        self.assertIn("physical performance", plan.concepts)


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
