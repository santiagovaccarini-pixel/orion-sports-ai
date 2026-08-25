from __future__ import annotations

import asyncio
import unittest

from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.providers.model_provider import ModelResult
from backend.app.services.knowledge_base import KnowledgeDocument
from backend.app.services.semantic_orchestrator import (
    EvidenceReview,
    LocalEvidence,
    SemanticOrchestrationError,
    collect_local_evidence,
    conservative_fallback_plan,
    create_semantic_plan,
    format_reasoning_context,
    merge_web_sources,
    parse_evidence_review,
    parse_semantic_plan,
    review_evidence,
)
from backend.app.services.web_research import WebSource


class FakePlanningProvider:
    name = "cloudflare"
    uses_local_resources = False

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return ModelResult(
            content=self.responses.pop(0),
            model="planner-test",
            thread_limit=0,
        )


class SemanticOrchestratorTests(unittest.TestCase):
    def test_parses_plan_from_json_without_keyword_routing(self) -> None:
        plan = parse_semantic_plan(
            """
            {
              "objective": "Conocer una estadística actual de un jugador",
              "entities": ["Jugador A"],
              "constraints": [],
              "references": [],
              "information_needed": ["estadística actual verificable"],
              "ambiguities": [],
              "use_web": true,
              "use_local_data": false,
              "use_calculator": false,
              "use_chart": false,
              "needs_clarification": false,
              "clarifying_question": null,
              "web_query": "estadística actual Jugador A",
              "local_document_names": [],
              "recommended_mode": "quick",
              "reason": "El dato puede cambiar con el tiempo"
            }
            """
        )
        self.assertTrue(plan.use_web)
        self.assertEqual(plan.recommended_mode, SelectedMode.QUICK)
        self.assertEqual(plan.entities, ("Jugador A",))

    def test_rejects_invalid_structured_plan(self) -> None:
        with self.assertRaises(SemanticOrchestrationError):
            parse_semantic_plan("esto no es json")

    def test_planner_receives_full_recent_conversation_and_capabilities(self) -> None:
        provider = FakePlanningProvider(
            [
                """
                {"objective":"comparar el período anterior con cinco partidos","entities":[],
                "constraints":["cinco partidos"],"references":["lo mismo"],
                "information_needed":[],"ambiguities":[],"use_web":false,
                "use_local_data":true,"use_calculator":false,"use_chart":false,
                "needs_clarification":false,"clarifying_question":null,"web_query":null,
                "local_document_names":["gps.csv"],"recommended_mode":"deep",
                "reason":"requiere conservar el contexto previo"}
                """
            ]
        )
        messages = [
            ChatMessage(role="user", content="Compará los últimos tres partidos."),
            ChatMessage(role="assistant", content="¿Qué variable querés comparar?"),
            ChatMessage(role="user", content="HSR."),
            ChatMessage(role="assistant", content="Entendido."),
            ChatMessage(role="user", content="Ahora hacé lo mismo con cinco."),
        ]
        plan = asyncio.run(
            create_semantic_plan(
                provider,
                messages,
                web_available=True,
                documents=[KnowledgeDocument("1", "gps.csv", "Player,HSR\nA,100")],
                sport=SportContext.FOOTBALL,
            )
        )
        self.assertEqual(plan.constraints, ("cinco partidos",))
        self.assertTrue(plan.use_local_data)
        sent_messages = provider.calls[0]["messages"]
        self.assertEqual(len(sent_messages), len(messages))
        self.assertIn("Catálogo de documentos", provider.calls[0]["system_prompt"])

    def test_conservative_fallback_does_not_classify_by_terms(self) -> None:
        messages = [ChatMessage(role="user", content="Una pregunta totalmente nueva")]
        plan = conservative_fallback_plan(
            messages,
            web_available=True,
            documents=[KnowledgeDocument("1", "archivo.csv", "a,b\n1,2")],
        )
        self.assertTrue(plan.use_web)
        self.assertTrue(plan.use_local_data)
        self.assertEqual(plan.web_query, messages[-1].content)

    def test_local_evidence_uses_model_selected_document_names(self) -> None:
        plan = parse_semantic_plan(
            """
            {"objective":"analizar datos locales","entities":[],"constraints":[],
             "references":[],"information_needed":[],"ambiguities":[],"use_web":false,
             "use_local_data":true,"use_calculator":false,"use_chart":false,
             "needs_clarification":false,"clarifying_question":null,"web_query":null,
             "local_document_names":["b.csv"],"recommended_mode":"quick","reason":"datos locales"}
            """
        )
        evidence = collect_local_evidence(
            [
                KnowledgeDocument("1", "a.csv", "A" * 30),
                KnowledgeDocument("2", "b.csv", "B" * 30),
            ],
            plan,
            max_characters=20,
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].document_name, "b.csv")
        self.assertTrue(evidence[0].truncated)

    def test_review_checks_scope_instead_of_requiring_fixed_source_count(self) -> None:
        review = parse_evidence_review(
            """
            {"sufficient":true,"relevant_source_ids":["W1"],"discarded_source_ids":["W2"],
             "missing_information":[],"follow_up_web_query":null,
             "needs_clarification":false,"clarifying_question":null,
             "resolved_scope":"total oficial acumulado","reason":"W1 es primaria y explícita"}
            """
        )
        self.assertTrue(review.sufficient)
        self.assertEqual(review.relevant_source_ids, ("W1",))
        self.assertEqual(review.resolved_scope, "total oficial acumulado")

    def test_reviewer_can_request_one_semantic_follow_up(self) -> None:
        plan = conservative_fallback_plan(
            [ChatMessage(role="user", content="Pregunta actual")],
            web_available=True,
            documents=[],
        )
        provider = FakePlanningProvider(
            [
                """
                {"sufficient":false,"relevant_source_ids":["W1"],"discarded_source_ids":[],
                 "missing_information":["confirmación del alcance"],
                 "follow_up_web_query":"confirmar alcance exacto de la estadística",
                 "needs_clarification":false,"clarifying_question":null,
                 "resolved_scope":null,"reason":"la primera fuente no define el alcance"}
                """
            ]
        )
        review = asyncio.run(
            review_evidence(
                provider,
                plan,
                [WebSource("Fuente", "https://example.org/a", "Dato parcial", "example.org")],
                [],
            )
        )
        self.assertFalse(review.sufficient)
        self.assertEqual(
            review.follow_up_web_query,
            "confirmar alcance exacto de la estadística",
        )

    def test_merge_web_sources_deduplicates_urls(self) -> None:
        a = WebSource("A", "https://a.test/1", "x", "a.test")
        b = WebSource("B", "https://b.test/2", "y", "b.test")
        merged = merge_web_sources([a], [a, b])
        self.assertEqual([item.url for item in merged], [a.url, b.url])

    def test_reasoning_context_marks_discarded_sources(self) -> None:
        plan = conservative_fallback_plan(
            [ChatMessage(role="user", content="Pregunta")],
            web_available=True,
            documents=[],
        )
        review = EvidenceReview(
            sufficient=True,
            relevant_source_ids=("W1",),
            discarded_source_ids=("W2",),
            missing_information=(),
            follow_up_web_query=None,
            needs_clarification=False,
            clarifying_question=None,
            resolved_scope="alcance validado",
            reason="ok",
        )
        context = format_reasoning_context(
            plan,
            review,
            [
                WebSource("Útil", "https://a.test", "dato", "a.test"),
                WebSource("No comparable", "https://b.test", "otro dato", "b.test"),
            ],
            [LocalEvidence("L1", "archivo.csv", "fila", False)],
        )
        self.assertIn('"discarded_source_ids": ["W2"]', context)
        self.assertIn("No conviertas fuentes descartadas en hechos", context)


if __name__ == "__main__":
    unittest.main()
