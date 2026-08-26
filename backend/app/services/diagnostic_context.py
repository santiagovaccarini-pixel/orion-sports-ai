from __future__ import annotations

from contextvars import ContextVar

from backend.app.services.diagnostic_trace import DiagnosticTrace


_final_trace: ContextVar[DiagnosticTrace | None] = ContextVar(
    "orion_final_diagnostic_trace",
    default=None,
)


def activate_final_trace(trace: DiagnosticTrace) -> None:
    """Attach a trace only to the final answer call in the current async context."""

    _final_trace.set(trace)


def current_final_trace() -> DiagnosticTrace | None:
    return _final_trace.get()


def clear_final_trace() -> None:
    _final_trace.set(None)
