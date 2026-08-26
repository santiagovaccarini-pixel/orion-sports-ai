from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.main import app
from backend.app.providers.model_provider import ModelProviderStatus, ModelStreamEvent
from backend.app.services.diagnostic_trace import diagnostic_traces
from backend.app.services.reasoning_pipeline import ReasoningBundle
from backend.app.services.semantic_orchestrator import EvidenceReview, SemanticPlan


class RecordingCloudProvider:
    name = "cloudflare"
    uses_local_resources = False

    def __init__(self) -> None:
        self.stream_kwargs = None

    def model_for(self, mode: SelectedMode) -> str:
        return "semantic-test-model"

    async def status(self) -> ModelProviderStatus:
        return ModelProviderStatus(True, ("semantic-test-model",), ())

    async def preflight(self, mode: SelectedMode) -> None:
        return None

    async def chat(self, **_kwargs):  # pragma: no cover
        raise AssertionError("El bundle está mockeado en esta prueba")

    async def chat_stream(self, **kwargs):
        self.stream_kwargs = kwargs
        yield ModelStreamEvent(
            content="Respuesta semántica",
            done=False,
            model="semantic-test-model",
        )
        yield ModelStreamEvent(
            content="",
            done=True,
            model="semantic-test-model",
            prompt_tokens=100,
            completion_tokens=20,
            reasoning_tokens=7,
            finish_reason="completed",
            reasoning_effort="low",
            endpoint="responses",
        )


def semantic_bundle() -> ReasoningBundle:
    plan = SemanticPlan(
        objective="Resolver el objetivo entendido por el modelo",
        entities=("Entidad",),
        constraints=(),
        references=(),
        information_needed=("evidencia",),
        ambiguities=(),
        evidence_policy="external",
        use_web=True,
        use_local_data=False,
        use_calculator=False,
        use_chart=False,
        needs_clarification=False,
        clarifying_question=None,
        web_query="consulta semántica",
        local_document_names=(),
        recommended_mode=SelectedMode.DEEP,
        reason="El modelo entendió que hace falta analizar evidencia.",
    )
    review = EvidenceReview(
        sufficient=True,
        relevant_source_ids=("W1",),
        discarded_source_ids=(),
        missing_information=(),
        follow_up_web_query=None,
        needs_clarification=False,
        clarifying_question=None,
        resolved_scope="alcance validado",
        reason="evidencia suficiente",
    )
    return ReasoningBundle(
        plan=plan,
        review=review,
        web_sources=(),
        local_evidence=(),
        context="CONTEXTO SEMÁNTICO DE PRUEBA",
        selected_mode=SelectedMode.DEEP,
    )


class SemanticStreamRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        diagnostic_traces.clear()

    def test_semantic_stream_uses_model_plan_and_skips_legacy_web_router(self) -> None:
        provider = RecordingCloudProvider()
        settings = Settings(
            model_provider="cloudflare",
            semantic_orchestration=True,
            diagnostics_enabled=True,
            web_enabled=True,
        )
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch("backend.app.api.routes._provider_or_http_error", return_value=provider),
            patch(
                "backend.app.api.routes.build_reasoning_bundle",
                new=AsyncMock(return_value=semantic_bundle()),
            ) as build_bundle,
            patch("backend.app.api.routes._web_context", new=AsyncMock()) as legacy_web,
            patch("backend.app.api.routes._documents", return_value=[]),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "Una redacción cualquiera"}],
                    "mode": "auto",
                    "sport": "football",
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[0]["selected_mode"], "deep")
        self.assertEqual(events[0]["recommended_mode"], "deep")
        self.assertTrue(events[0]["trace_id"].startswith("orion-"))
        self.assertEqual(events[1]["content"], "Respuesta semántica")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["finish_reason"], "completed")
        self.assertEqual(events[-1]["reasoning_tokens"], 7)
        build_bundle.assert_awaited_once()
        legacy_web.assert_not_awaited()
        self.assertIsNotNone(provider.stream_kwargs)
        self.assertIn(
            "CONTEXTO SEMÁNTICO DE PRUEBA",
            provider.stream_kwargs["system_prompt"],
        )

        trace = diagnostic_traces.latest()
        self.assertIsNotNone(trace)
        assert trace is not None
        snapshot = trace.snapshot()
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["final_answer"], "Respuesta semántica")
        self.assertEqual(snapshot["model_calls"][-1]["stage"], "final_answer")
        self.assertEqual(snapshot["model_calls"][-1]["reasoning_tokens"], 7)
        self.assertEqual(snapshot["model_calls"][-1]["finish_reason"], "completed")
        self.assertIsNotNone(snapshot["prompt_metadata"])
        self.assertFalse(snapshot["privacy"]["hidden_chain_of_thought_recorded"])

    def test_semantic_plan_can_ask_clarification_without_final_generation(self) -> None:
        provider = RecordingCloudProvider()
        bundle = semantic_bundle()
        plan = replace(
            bundle.plan,
            needs_clarification=True,
            clarifying_question="¿A qué período te referís?",
        )
        clarification_bundle = ReasoningBundle(
            plan=plan,
            review=bundle.review,
            web_sources=(),
            local_evidence=(),
            context=bundle.context,
            selected_mode=SelectedMode.DEEP,
        )
        settings = Settings(
            model_provider="cloudflare",
            semantic_orchestration=True,
            diagnostics_enabled=True,
        )
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch("backend.app.api.routes._provider_or_http_error", return_value=provider),
            patch(
                "backend.app.api.routes.build_reasoning_bundle",
                new=AsyncMock(return_value=clarification_bundle),
            ),
            patch("backend.app.api.routes._documents", return_value=[]),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "Hacé lo mismo"}]},
            )

        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[1]["content"], "¿A qué período te referís?")
        self.assertIsNone(provider.stream_kwargs)
        trace = diagnostic_traces.latest()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.snapshot()["status"], "completed")


if __name__ == "__main__":
    unittest.main()
