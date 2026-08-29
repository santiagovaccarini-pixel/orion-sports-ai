from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.app.api.routes import require_api_key
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

    def __init__(self, content: str = "Respuesta semántica") -> None:
        self.stream_kwargs = None
        self.content = content

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
            content=self.content,
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
        # These tests exercise routing behavior, not auth; bypass the
        # (now fail-closed) API key dependency instead of configuring a key.
        app.dependency_overrides[require_api_key] = lambda: None

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)

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
        meta_index = next(
            index for index, event in enumerate(events) if event.get("type") == "meta"
        )
        stage_indexes = [
            index for index, event in enumerate(events) if event.get("type") == "stage"
        ]
        self.assertTrue(stage_indexes, "Debe emitirse al menos un frame de etapa")
        self.assertLess(stage_indexes[0], meta_index)
        meta = events[meta_index]
        self.assertEqual(meta["selected_mode"], "deep")
        self.assertEqual(meta["recommended_mode"], "deep")
        self.assertTrue(meta["trace_id"].startswith("orion-"))
        first_content = next(
            event for event in events if event.get("type") == "content"
        )
        self.assertEqual(first_content["content"], "Respuesta semántica")
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
        first_content = next(
            event for event in events if event.get("type") == "content"
        )
        self.assertEqual(first_content["content"], "¿A qué período te referís?")
        self.assertIsNone(provider.stream_kwargs)
        trace = diagnostic_traces.latest()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.snapshot()["status"], "completed")

    def test_unsupported_numbers_are_recorded_as_a_guard_event(self) -> None:
        provider = RecordingCloudProvider(
            content="El jugador acumuló 987 goles en total."
        )
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
            ),
            patch("backend.app.api.routes._web_context", new=AsyncMock()),
            patch("backend.app.api.routes._documents", return_value=[]),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "¿Cuántos goles lleva?"}],
                    "mode": "auto",
                    "sport": "football",
                },
            )

        self.assertEqual(response.status_code, 200)
        trace = diagnostic_traces.latest()
        self.assertIsNotNone(trace)
        assert trace is not None
        guard_events = trace.snapshot()["guard_events"]
        self.assertTrue(
            any(
                event["event"] == "unsupported_numbers_detected"
                and "987" in event["detail"]
                for event in guard_events
            )
        )

    def test_unexpected_exception_still_yields_an_error_frame(self) -> None:
        # Regression: an unanticipated exception (not one of the specific
        # provider errors) used to close the connection with zero bytes sent
        # and no error frame, leaving the client with an empty, unexplained
        # response and no diagnostic signal. Iterates the route's raw
        # body_iterator directly so transport-level chunk buffering in the
        # test client can't mask what the generator actually yields.
        from backend.app.api.routes import chat_stream
        from backend.app.domain.schemas import ChatRequest

        settings = Settings(
            model_provider="cloudflare",
            semantic_orchestration=True,
            diagnostics_enabled=True,
        )
        request = ChatRequest(messages=[{"role": "user", "content": "Pregunta"}])
        with (
            patch("backend.app.api.routes.get_settings", return_value=settings),
            patch(
                "backend.app.api.routes._provider_or_http_error",
                return_value=RecordingCloudProvider(),
            ),
            patch(
                "backend.app.api.routes.build_reasoning_bundle",
                new=AsyncMock(side_effect=RuntimeError("boom inesperado")),
            ),
            patch("backend.app.api.routes._documents", return_value=[]),
        ):

            async def collect() -> list[dict]:
                streaming_response = await chat_stream(request)
                events: list[dict] = []
                with self.assertRaises(RuntimeError):
                    async for chunk in streaming_response.body_iterator:
                        for line in chunk.decode("utf-8").splitlines():
                            if line.strip():
                                events.append(json.loads(line))
                return events

            events = asyncio.run(collect())

        error_events = [event for event in events if event.get("type") == "error"]
        self.assertTrue(error_events, "Debe emitirse un frame de error")
        self.assertEqual(error_events[0]["code"], "internal_error")
        self.assertIn("boom inesperado", error_events[0]["message"])
        trace = diagnostic_traces.latest()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.snapshot()["status"], "error")


if __name__ == "__main__":
    unittest.main()
