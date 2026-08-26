from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.main import app
from backend.app.services.diagnostic_trace import DiagnosticTraceStore


class DiagnosticTraceTests(unittest.TestCase):
    def test_trace_keeps_only_bounded_in_memory_history(self) -> None:
        store = DiagnosticTraceStore(max_traces=2)
        first = store.start(question="uno", sport="football", requested_mode="auto")
        second = store.start(question="dos", sport="football", requested_mode="auto")
        third = store.start(question="tres", sport="football", requested_mode="auto")

        self.assertIsNone(store.get(first.trace_id))
        self.assertIsNotNone(store.get(second.trace_id))
        self.assertEqual(store.latest().trace_id, third.trace_id)  # type: ignore[union-attr]

    def test_trace_records_observable_decisions_without_secret_fields(self) -> None:
        store = DiagnosticTraceStore(max_traces=2)
        trace = store.start(
            question="¿Qué pasó?",
            sport="football",
            requested_mode="auto",
        )
        plan = SimpleNamespace(
            objective="Investigar el dato solicitado",
            entities=("Jugador",),
            constraints=(),
            references=(),
            information_needed=("Dato actualizado",),
            ambiguities=(),
            use_web=True,
            use_local_data=False,
            use_calculator=False,
            use_chart=False,
            needs_clarification=False,
            clarifying_question=None,
            web_query="consulta web",
            local_document_names=(),
            recommended_mode=SimpleNamespace(value="quick"),
        )
        trace.record_plan(plan, fallback=False, duration_ms=12.5)
        snapshot = trace.snapshot()

        self.assertEqual(snapshot["plan"]["web_query"], "consulta web")  # type: ignore[index]
        self.assertFalse(snapshot["privacy"]["hidden_chain_of_thought_recorded"])  # type: ignore[index]
        self.assertFalse(snapshot["privacy"]["credentials_recorded"])  # type: ignore[index]
        serialized = repr(snapshot).lower()
        self.assertNotIn("api_token", serialized)
        self.assertNotIn("authorization", serialized)

    def test_diagnostic_routes_are_mounted(self) -> None:
        paths = set(app.openapi()["paths"])
        self.assertIn("/api/v1/diagnostics/traces/latest", paths)
        self.assertIn("/api/v1/diagnostics/traces/{trace_id}", paths)


if __name__ == "__main__":
    unittest.main()
