from __future__ import annotations

import asyncio
import unittest
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

from backend.app.core.config import Settings
from backend.app.domain.schemas import ChatRequest
from backend.app.providers.model_provider import (
    ModelProviderUnavailableError,
    ModelResult,
)
from backend.app.services.reasoning_pipeline import build_reasoning_bundle
from backend.app.services.web_reader import PageRead
from backend.app.services.web_research import WebSource


class FakeProvider:
    name = "cloudflare"
    uses_local_resources = False

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def chat(self, **_kwargs):
        return ModelResult(
            content=self.responses.pop(0),
            model="test-model",
            thread_limit=0,
        )


class FailingProvider:
    name = "cloudflare"
    uses_local_resources = False

    async def chat(self, **_kwargs):
        raise ModelProviderUnavailableError("etapa interna temporalmente no disponible")


PLAN = """
{
  "objective":"resolver una consulta actual",
  "entities":["Entidad"],
  "constraints":[],
  "references":[],
  "information_needed":["dato actual y alcance"],
  "ambiguities":[],
  "evidence_policy":"external",
  "use_web":true,
  "use_local_data":false,
  "use_calculator":false,
  "use_chart":false,
  "needs_clarification":false,
  "clarifying_question":null,
  "web_query":"consulta inicial semántica",
  "local_document_names":[],
  "recommended_mode":"quick",
  "reason":"el dato debe verificarse"
}
"""

REVIEW_MORE = """
{
  "sufficient":false,
  "relevant_source_ids":["W1"],
  "discarded_source_ids":[],
  "missing_information":["confirmar alcance"],
  "follow_up_web_query":"consulta de seguimiento creada por el revisor",
  "needs_clarification":false,
  "clarifying_question":null,
  "resolved_scope":null,
  "reason":"falta confirmar alcance"
}
"""

REVIEW_SAME_QUERY = """
{
  "sufficient":false,
  "relevant_source_ids":[],
  "discarded_source_ids":[],
  "missing_information":["más evidencia"],
  "follow_up_web_query":"consulta inicial semántica",
  "needs_clarification":false,
  "clarifying_question":null,
  "resolved_scope":null,
  "reason":"propone repetir la misma búsqueda"
}
"""

REVIEW_OK = """
{
  "sufficient":true,
  "relevant_source_ids":["W1","W2"],
  "discarded_source_ids":[],
  "missing_information":[],
  "follow_up_web_query":null,
  "needs_clarification":false,
  "clarifying_question":null,
  "resolved_scope":"alcance confirmado por evidencia",
  "reason":"la evidencia ya permite responder"
}
"""


class ReasoningPipelineTests(unittest.TestCase):
    def test_model_can_request_one_follow_up_search(self) -> None:
        provider = FakeProvider([PLAN, REVIEW_MORE, REVIEW_OK])
        settings = Settings(
            web_enabled=True,
            web_provider="tavily",
            tavily_api_key="test",
            semantic_orchestration=True,
            semantic_max_tool_rounds=2,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "Pregunta de prueba"}]
        )
        first = (WebSource("A", "https://a.test", "evidencia A", "a.test"),)
        second = (WebSource("B", "https://b.test", "evidencia B", "b.test"),)
        with patch(
            "backend.app.services.reasoning_pipeline.research",
            new=AsyncMock(side_effect=[first, second]),
        ) as search:
            bundle = asyncio.run(
                build_reasoning_bundle(provider, request, settings, [])
            )
        self.assertEqual(search.await_count, 2)
        self.assertEqual(
            search.await_args_list[1].args[0],
            "consulta de seguimiento creada por el revisor",
        )
        self.assertTrue(bundle.review.sufficient)
        self.assertEqual(len(bundle.web_sources), 2)

    def test_tool_round_limit_is_enforced_by_python_not_keywords(self) -> None:
        provider = FakeProvider([PLAN, REVIEW_MORE])
        settings = Settings(
            web_enabled=True,
            web_provider="tavily",
            tavily_api_key="test",
            semantic_orchestration=True,
            semantic_max_tool_rounds=1,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "Cualquier redacción"}]
        )
        with patch(
            "backend.app.services.reasoning_pipeline.research",
            new=AsyncMock(
                return_value=(
                    WebSource("A", "https://a.test", "evidencia A", "a.test"),
                )
            ),
        ) as search:
            bundle = asyncio.run(
                build_reasoning_bundle(provider, request, settings, [])
            )
        self.assertEqual(search.await_count, 1)
        self.assertFalse(bundle.review.sufficient)

    def test_duplicate_follow_up_query_is_blocked_without_spending_round(self) -> None:
        provider = FakeProvider([PLAN, REVIEW_SAME_QUERY])
        settings = Settings(
            web_enabled=True,
            semantic_orchestration=True,
            semantic_max_tool_rounds=3,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "Pregunta de prueba"}]
        )
        with patch(
            "backend.app.services.reasoning_pipeline.research",
            new=AsyncMock(
                return_value=(
                    WebSource("A", "https://a.test", "evidencia A", "a.test"),
                )
            ),
        ) as search:
            bundle = asyncio.run(
                build_reasoning_bundle(provider, request, settings, [])
            )
        self.assertEqual(search.await_count, 1)
        self.assertFalse(bundle.review.sufficient)

    def test_provider_unavailability_is_not_silently_converted_to_evidence_failure(self) -> None:
        provider = FailingProvider()
        settings = Settings(
            web_enabled=True,
            semantic_orchestration=True,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "Pregunta sin patrón predefinido"}]
        )
        with self.assertRaises(ModelProviderUnavailableError):
            asyncio.run(build_reasoning_bundle(provider, request, settings, []))

    def test_freshness_backstop_demotes_stale_acceptance_and_deepens(self) -> None:
        volatile_plan = PLAN.replace(
            '"evidence_policy":"external",',
            '"evidence_policy":"external",\n'
            '  "volatile_information":true,\n'
            '  "recency_window_days":7,',
        )
        review_accepts_stale = """
        {"sufficient":true,"relevant_source_ids":["W1"],"discarded_source_ids":[],
         "missing_information":[],"follow_up_web_query":null,
         "needs_clarification":false,"clarifying_question":null,
         "resolved_scope":"resultado reciente","freshness_verified":true,
         "reason":"acepta una fuente sin fecha comprobable"}
        """
        provider = FakeProvider([volatile_plan, review_accepts_stale, review_accepts_stale])
        settings = Settings(
            web_enabled=True,
            web_provider="tavily",
            tavily_api_key="test",
            semantic_orchestration=True,
            semantic_max_tool_rounds=2,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "¿Cómo salió el último partido?"}]
        )
        undated = (WebSource("Sin fecha", "https://a.test", "resultado", "a.test"),)
        dated_read = PageRead(
            source_id="W1",
            title="Crónica",
            url="https://a.test",
            domain="a.test",
            excerpt="resultado actualizado del partido",
            published_date=(date.today() - timedelta(days=1)).isoformat(),
        )
        with (
            patch(
                "backend.app.services.reasoning_pipeline.research",
                new=AsyncMock(return_value=undated),
            ) as search,
            patch(
                "backend.app.services.reasoning_pipeline.read_source_pages",
                new=AsyncMock(return_value=(dated_read,)),
            ) as reader,
        ):
            bundle = asyncio.run(
                build_reasoning_bundle(provider, request, settings, [])
            )
        # La primera aceptación sin fuente fechada se demueve una vez; la ronda
        # extra abre la página, obtiene la fecha y la segunda revisión pasa.
        self.assertEqual(search.await_count, 1)
        reader.assert_awaited_once()
        self.assertTrue(bundle.review.sufficient)
        self.assertEqual(provider.responses, [])
        self.assertTrue(bundle.web_sources[0].deepened)
        self.assertEqual(
            search.await_args.kwargs.get("recency_days"), 7
        )

    def test_freshness_backstop_leaves_insufficient_when_no_dated_source_found(self) -> None:
        volatile_plan = PLAN.replace(
            '"evidence_policy":"external",',
            '"evidence_policy":"external",\n'
            '  "volatile_information":true,\n'
            '  "recency_window_days":7,',
        )
        review_accepts_stale = """
        {"sufficient":true,"relevant_source_ids":["W1"],"discarded_source_ids":[],
         "missing_information":[],"follow_up_web_query":null,
         "needs_clarification":false,"clarifying_question":null,
         "resolved_scope":"resultado reciente","freshness_verified":true,
         "reason":"acepta una fuente sin fecha comprobable"}
        """
        provider = FakeProvider([volatile_plan, review_accepts_stale])
        settings = Settings(
            web_enabled=True,
            web_provider="tavily",
            tavily_api_key="test",
            semantic_orchestration=True,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "¿Cómo salió el último partido?"}]
        )
        undated = (WebSource("Sin fecha", "https://a.test", "resultado", "a.test"),)
        with (
            patch(
                "backend.app.services.reasoning_pipeline.research",
                new=AsyncMock(return_value=undated),
            ),
            patch(
                "backend.app.services.reasoning_pipeline.read_source_pages",
                new=AsyncMock(return_value=()),
            ),
        ):
            bundle = asyncio.run(
                build_reasoning_bundle(provider, request, settings, [])
            )
        # Sin página fechada disponible, la democión persiste: la respuesta final
        # recibe evidencia insuficiente y debe declarar la limitación.
        self.assertFalse(bundle.review.sufficient)
        self.assertTrue(
            any(
                "recencia" in item
                for item in bundle.review.missing_information
            )
        )

    def test_freshness_backstop_passes_with_dated_accepted_source(self) -> None:
        volatile_plan = PLAN.replace(
            '"evidence_policy":"external",',
            '"evidence_policy":"external",\n'
            '  "volatile_information":true,\n'
            '  "recency_window_days":7,',
        )
        review_fresh = """
        {"sufficient":true,"relevant_source_ids":["W1"],"discarded_source_ids":[],
         "missing_information":[],"follow_up_web_query":null,
         "needs_clarification":false,"clarifying_question":null,
         "resolved_scope":"resultado reciente","freshness_verified":true,
         "reason":"fuente fechada dentro de la ventana"}
        """
        provider = FakeProvider([volatile_plan, review_fresh])
        settings = Settings(
            web_enabled=True,
            web_provider="tavily",
            tavily_api_key="test",
            semantic_orchestration=True,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "¿Cómo salió el último partido?"}]
        )
        dated = (
            WebSource(
                "Crónica",
                "https://a.test",
                "resultado",
                "a.test",
                published_date="2026-08-26",
                published_age_days=1,
            ),
        )
        with patch(
            "backend.app.services.reasoning_pipeline.research",
            new=AsyncMock(return_value=dated),
        ):
            bundle = asyncio.run(
                build_reasoning_bundle(provider, request, settings, [])
            )
        self.assertTrue(bundle.review.sufficient)
        self.assertEqual(provider.responses, [])

    def test_partial_values_are_summed_deterministically_into_final_context(self) -> None:
        review_with_partial_values = """
        {
          "sufficient":false,
          "relevant_source_ids":["W1","W2"],
          "discarded_source_ids":[],
          "missing_information":["confirmación con fuente única del total combinado"],
          "follow_up_web_query":null,
          "needs_clarification":false,
          "clarifying_question":null,
          "resolved_scope":null,
          "partial_values":[
            {"source_id":"W1","label":"goles en liga","value":5},
            {"source_id":"W2","label":"goles en copa","value":3}
          ],
          "reason":"ninguna fuente confirma el total combinado, pero ambas partes están verificadas"
        }
        """
        provider = FakeProvider([PLAN, review_with_partial_values])
        settings = Settings(
            web_enabled=True,
            web_provider="tavily",
            tavily_api_key="test",
            semantic_orchestration=True,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "¿Cuántos goles totales lleva el jugador?"}]
        )
        sources = (
            WebSource("Liga", "https://a.test", "cinco goles en liga", "a.test"),
            WebSource("Copa", "https://b.test", "tres goles en copa", "b.test"),
        )
        with patch(
            "backend.app.services.reasoning_pipeline.research",
            new=AsyncMock(return_value=sources),
        ):
            bundle = asyncio.run(
                build_reasoning_bundle(provider, request, settings, [])
            )
        self.assertFalse(bundle.review.sufficient)
        self.assertIn("Suma total = 8.0", bundle.context)
        self.assertIn("goles en liga: 5.0 (fuente W1)", bundle.context)

    def test_clarification_with_core_gap_short_circuits_before_web_and_review(self) -> None:
        plan_with_clarification = PLAN.replace(
            '"needs_clarification":false,\n  "clarifying_question":null,',
            '"needs_clarification":true,\n'
            '  "clarifying_question":"¿Sobre qué período exacto?",',
        ).replace(
            '"information_needed":["dato actual y alcance"],',
            '"information_needed":["dato actual y alcance"],\n'
            '  "missing_for_core":["período de referencia"],',
        )
        provider = FakeProvider([plan_with_clarification])
        settings = Settings(
            web_enabled=True,
            web_provider="tavily",
            tavily_api_key="test",
            semantic_orchestration=True,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "Compará el rendimiento"}]
        )
        with patch(
            "backend.app.services.reasoning_pipeline.research",
            new=AsyncMock(
                side_effect=AssertionError("no debe buscarse antes de aclarar")
            ),
        ) as search:
            bundle = asyncio.run(
                build_reasoning_bundle(provider, request, settings, [])
            )
        search.assert_not_awaited()
        self.assertEqual(provider.responses, [])
        self.assertTrue(bundle.plan.needs_clarification)
        self.assertTrue(bundle.review.needs_clarification)
        self.assertEqual(
            bundle.review.clarifying_question, "¿Sobre qué período exacto?"
        )
        self.assertFalse(bundle.review.audited)
        self.assertEqual(bundle.web_sources, ())

    def test_clarification_without_core_gap_is_downgraded_and_pipeline_runs(self) -> None:
        plan_with_soft_clarification = PLAN.replace(
            '"needs_clarification":false,\n  "clarifying_question":null,',
            '"needs_clarification":true,\n'
            '  "clarifying_question":"¿Querés más detalle?",',
        )
        provider = FakeProvider([plan_with_soft_clarification, REVIEW_OK])
        settings = Settings(
            web_enabled=True,
            web_provider="tavily",
            tavily_api_key="test",
            semantic_orchestration=True,
        )
        request = ChatRequest(
            messages=[{"role": "user", "content": "Pregunta respondible"}]
        )
        sources = (WebSource("A", "https://a.test", "evidencia A", "a.test"),)
        with patch(
            "backend.app.services.reasoning_pipeline.research",
            new=AsyncMock(return_value=sources),
        ) as search:
            bundle = asyncio.run(
                build_reasoning_bundle(provider, request, settings, [])
            )
        search.assert_awaited_once()
        self.assertFalse(bundle.plan.needs_clarification)
        self.assertFalse(bundle.review.needs_clarification)
        self.assertTrue(bundle.review.sufficient)


if __name__ == "__main__":
    unittest.main()
