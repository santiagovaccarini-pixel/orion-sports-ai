from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend.app.core.config import Settings
from backend.app.domain.intent import SemanticPlan
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.services.knowledge_base import KnowledgeBase, KnowledgeDocument
from backend.app.services.semantic_normalizer import normalize_semantic_plan
from backend.app.services.semantic_planner import create_semantic_plan
from backend.app.services.semantic_retriever import search_with_intent


class SemanticPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_planner_uses_neutral_fallback_not_keyword_guessing(self) -> None:
        plan = await create_semantic_plan(
            Settings(semantic_planner_enabled=False),
            [
                ChatMessage(
                    role="user",
                    content="Las lesiones subieron cuando aumentamos la carga. ¿Fue la causa?",
                )
            ],
            SportContext.FOOTBALL,
            has_local_documents=False,
        )

        self.assertEqual(plan.task_type, "direct_answer")
        self.assertEqual(plan.inference_type, "descriptive")
        self.assertEqual(plan.concept_ids, [])
        self.assertGreaterEqual(plan.ambiguity, 0.6)
        self.assertLessEqual(plan.confidence, 0.25)
        self.assertFalse(plan.causal_claim_risk)

    async def test_structured_reasoning_selects_concepts_and_derives_domain(self) -> None:
        structured = {
            "user_goal": "evaluar si una menor demanda externa implica peor rendimiento físico",
            "task_type": "interpretation",
            "inference_type": "causal",
            "concept_ids": ["external_load", "physical_performance", "match_exposure"],
            "missing_variables": ["exposición comparable", "posición y rol"],
            "needs_local_data": False,
            "needs_private_memory": False,
            "needs_web": False,
            "requires_clarification": False,
            "referenced_previous_context": False,
            "confidence": 0.88,
        }
        with patch(
            "backend.app.services.semantic_planner.OllamaClient.structured_json",
            new=AsyncMock(return_value=structured),
        ):
            plan = await create_semantic_plan(
                Settings(semantic_planner_enabled=True),
                [ChatMessage(role="user", content="Uno corrió menos. ¿Eso prueba que rindió peor?")],
                SportContext.FOOTBALL,
                has_local_documents=False,
            )

        self.assertEqual(plan.domain, "physical_performance")
        self.assertEqual(plan.inference_type, "causal")
        self.assertTrue(plan.causal_claim_risk)
        self.assertEqual(
            plan.concept_ids,
            ["external_load", "physical_performance", "match_exposure"],
        )
        self.assertIn("external load", plan.concepts)
        self.assertIn("physical performance", plan.concepts)

    async def test_private_operational_definition_is_scoped_by_ontology(self) -> None:
        structured = {
            "user_goal": "usar la definición operacional propia del club",
            "task_type": "definition",
            "inference_type": "descriptive",
            "concept_ids": ["sprint_threshold", "private_operational_definition"],
            "missing_variables": [],
            "needs_local_data": False,
            "needs_private_memory": True,
            "needs_web": False,
            "requires_clarification": False,
            "referenced_previous_context": True,
            "confidence": 0.94,
        }
        with patch(
            "backend.app.services.semantic_planner.OllamaClient.structured_json",
            new=AsyncMock(return_value=structured),
        ):
            plan = await create_semantic_plan(
                Settings(semantic_planner_enabled=True),
                [ChatMessage(role="user", content="Usá nuestra definición habitual de sprint.")],
                SportContext.FOOTBALL,
                has_local_documents=False,
            )

        self.assertTrue(plan.needs_private_memory)
        self.assertFalse(plan.needs_global_knowledge)
        self.assertTrue(plan.referenced_previous_context)
        self.assertIn("private_operational_definition", plan.concept_ids)

    async def test_comparative_inference_sets_comparison_without_text_rules(self) -> None:
        structured = {
            "user_goal": "comparar dos demandas físicas contextualizando la exposición",
            "task_type": "interpretation",
            "inference_type": "comparative",
            "concept_ids": ["hsr", "match_exposure"],
            "missing_variables": [],
            "needs_local_data": False,
            "needs_private_memory": False,
            "needs_web": False,
            "requires_clarification": False,
            "referenced_previous_context": False,
            "confidence": 0.9,
        }
        with patch(
            "backend.app.services.semantic_planner.OllamaClient.structured_json",
            new=AsyncMock(return_value=structured),
        ):
            plan = await create_semantic_plan(
                Settings(semantic_planner_enabled=True),
                [ChatMessage(role="user", content="Caso A versus caso B")],
                SportContext.FOOTBALL,
                has_local_documents=False,
            )

        self.assertTrue(plan.comparison)
        self.assertEqual(plan.domain, "physical_performance")
        self.assertIn("hsr", plan.concept_ids)
        self.assertIn("match_exposure", plan.concept_ids)

    async def test_invalid_ontology_ids_are_rejected(self) -> None:
        structured = {
            "user_goal": "interpretar una métrica",
            "task_type": "interpretation",
            "inference_type": "interpretive",
            "concept_ids": ["external_load", "invented_magic_concept"],
            "missing_variables": [],
            "needs_local_data": False,
            "needs_private_memory": False,
            "needs_web": False,
            "requires_clarification": False,
            "referenced_previous_context": False,
            "confidence": 0.85,
        }
        with patch(
            "backend.app.services.semantic_planner.OllamaClient.structured_json",
            new=AsyncMock(return_value=structured),
        ):
            plan = await create_semantic_plan(
                Settings(semantic_planner_enabled=True),
                [ChatMessage(role="user", content="Interpretá esta situación")],
                SportContext.FOOTBALL,
                has_local_documents=False,
            )

        self.assertEqual(plan.concept_ids, ["external_load"])
        self.assertNotIn("invented_magic_concept", plan.concept_ids)

    def test_normalizer_does_not_reinterpret_raw_message_wording(self) -> None:
        base_plan = SemanticPlan(
            literal_request="x",
            user_goal="evaluar una comparación física",
            task_type="interpretation",
            inference_type="comparative",
            concept_ids=["external_load", "match_exposure"],
            confidence=0.9,
        )
        plan_a = normalize_semantic_plan(
            base_plan.model_copy(deep=True),
            [ChatMessage(role="user", content="HSR sprint distancia fatiga")],
            SportContext.FOOTBALL,
            has_local_documents=False,
        )
        plan_b = normalize_semantic_plan(
            base_plan.model_copy(deep=True),
            [ChatMessage(role="user", content="Una frase totalmente diferente sin esos términos")],
            SportContext.FOOTBALL,
            has_local_documents=False,
        )

        self.assertEqual(plan_a.model_dump(), plan_b.model_dump())

    def test_causal_flag_is_structural_not_keyword_driven(self) -> None:
        plan = SemanticPlan(
            literal_request="x",
            user_goal="evaluar causalidad de lesión",
            task_type="interpretation",
            inference_type="interpretive",
            concept_ids=["injury_causality"],
            confidence=0.9,
        )
        normalized = normalize_semantic_plan(
            plan,
            [ChatMessage(role="user", content="Texto sin la palabra causa")],
            SportContext.FOOTBALL,
            has_local_documents=False,
        )
        self.assertTrue(normalized.causal_claim_risk)


class SemanticRetrieverTests(unittest.TestCase):
    def _knowledge_base(self, directory: str) -> KnowledgeBase:
        base = KnowledgeBase(Path(directory) / "knowledge.json")
        base.add_document(
            KnowledgeDocument(
                id="physical-1",
                name="principios_carga.md",
                content=(
                    "La carga externa describe el trabajo realizado. High-speed running "
                    "debe interpretarse considerando exposición y contexto."
                ),
            )
        )
        return base

    def test_retrieval_is_skipped_when_reasoning_does_not_require_local_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = self._knowledge_base(directory)
            plan = SemanticPlan(
                literal_request="x",
                user_goal="explicar carga externa",
                task_type="definition",
                inference_type="descriptive",
                concept_ids=["external_load"],
                concepts=["external load"],
                needs_local_data=False,
            )
            results = search_with_intent(base, "carga externa", plan, limit=5)
        self.assertEqual(results, [])

    def test_local_retrieval_runs_only_after_intent_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = self._knowledge_base(directory)
            plan = SemanticPlan(
                literal_request="x",
                user_goal="consultar el documento local sobre carga externa",
                task_type="data_query",
                inference_type="descriptive",
                concept_ids=["external_load"],
                concepts=["external load"],
                retrieval_queries=["external load"],
                needs_local_data=True,
                needs_global_knowledge=False,
            )
            results = search_with_intent(base, "carga externa", plan, limit=5)
        self.assertTrue(results)
        self.assertEqual(results[0].document_name, "principios_carga.md")


if __name__ == "__main__":
    unittest.main()
