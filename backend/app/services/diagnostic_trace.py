from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Sequence
from uuid import uuid4


MAX_TRACES = 50
MAX_QUESTION_CHARACTERS = 4_000
MAX_FINAL_ANSWER_CHARACTERS = 8_000
MAX_SOURCE_EXCERPT_CHARACTERS = 1_500
MAX_LOCAL_EXCERPT_CHARACTERS = 1_500


def _clip(value: str, limit: int) -> str:
    clean = value.strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "…"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_payload(source: object, source_id: str) -> dict[str, str]:
    return {
        "source_id": source_id,
        "title": _clip(str(getattr(source, "title", "")), 500),
        "url": str(getattr(source, "url", "")),
        "domain": str(getattr(source, "domain", "")),
        "excerpt": _clip(
            str(getattr(source, "excerpt", "")),
            MAX_SOURCE_EXCERPT_CHARACTERS,
        ),
    }


@dataclass(slots=True)
class DiagnosticTrace:
    """Observable execution trace for Orion development diagnostics.

    The trace records structured decisions, tool inputs/outputs, completion metadata
    and timings. It never records hidden chain-of-thought, provider credentials,
    authorization headers or API tokens. Traces are memory-only in this prototype.
    """

    trace_id: str
    created_at: str
    question: str
    sport: str
    requested_mode: str
    status: str = "running"
    model: str | None = None
    plan: dict[str, Any] | None = None
    plan_fallback: bool = False
    plan_error: str | None = None
    contract: dict[str, Any] | None = None
    local_evidence: list[dict[str, Any]] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    guard_events: list[dict[str, Any]] = field(default_factory=list)
    prompt_metadata: dict[str, Any] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    final_answer: str | None = None
    error: str | None = None
    completed_at: str | None = None
    _lock: RLock = field(default_factory=RLock, repr=False)

    def set_model(self, model: str) -> None:
        with self._lock:
            self.model = model

    def record_plan(
        self,
        plan: object,
        *,
        fallback: bool,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self.plan = {
                "objective": getattr(plan, "objective", ""),
                "entities": list(getattr(plan, "entities", ())),
                "constraints": list(getattr(plan, "constraints", ())),
                "references": list(getattr(plan, "references", ())),
                "information_needed": list(getattr(plan, "information_needed", ())),
                "ambiguities": list(getattr(plan, "ambiguities", ())),
                "evidence_policy": getattr(plan, "evidence_policy", None),
                "use_web": bool(getattr(plan, "use_web", False)),
                "use_local_data": bool(getattr(plan, "use_local_data", False)),
                "use_calculator": bool(getattr(plan, "use_calculator", False)),
                "use_chart": bool(getattr(plan, "use_chart", False)),
                "needs_clarification": bool(
                    getattr(plan, "needs_clarification", False)
                ),
                "clarifying_question": getattr(plan, "clarifying_question", None),
                "web_query": getattr(plan, "web_query", None),
                "local_document_names": list(
                    getattr(plan, "local_document_names", ())
                ),
                "recommended_mode": getattr(
                    getattr(plan, "recommended_mode", None), "value", None
                ),
                "resolved_request": getattr(plan, "resolved_request", ""),
                "missing_for_core": list(getattr(plan, "missing_for_core", ())),
                "missing_for_precision": list(
                    getattr(plan, "missing_for_precision", ())
                ),
                "volatile_information": bool(
                    getattr(plan, "volatile_information", False)
                ),
                "recency_window_days": getattr(plan, "recency_window_days", None),
            }
            self.plan_fallback = fallback
            self.plan_error = _clip(error, 1_000) if error else None
            self.timings_ms["planning"] = round(duration_ms, 2)

    def record_local_evidence(self, evidence: Sequence[object]) -> None:
        with self._lock:
            self.local_evidence = [
                {
                    "source_id": getattr(item, "source_id", ""),
                    "document_name": getattr(item, "document_name", ""),
                    "chunk_index": getattr(item, "chunk_index", None),
                    "truncated": bool(getattr(item, "truncated", False)),
                    "excerpt": _clip(
                        str(getattr(item, "content", "")),
                        MAX_LOCAL_EXCERPT_CHARACTERS,
                    ),
                }
                for item in evidence
            ]

    def record_search(
        self,
        *,
        round_number: int,
        query: str,
        sources: Sequence[object],
        duration_ms: float,
    ) -> None:
        with self._lock:
            self.searches.append(
                {
                    "round": round_number,
                    "query": _clip(query, 2_000),
                    "duration_ms": round(duration_ms, 2),
                    "raw_results": [
                        _source_payload(source, f"R{round_number}.{index}")
                        for index, source in enumerate(sources, start=1)
                    ],
                }
            )
            self.timings_ms["web_search_total"] = round(
                sum(float(item["duration_ms"]) for item in self.searches), 2
            )

    def record_review(
        self,
        review: object,
        *,
        round_number: int,
        fallback: bool,
        duration_ms: float,
        web_sources: Sequence[object] = (),
        error: str | None = None,
    ) -> None:
        with self._lock:
            self.reviews.append(
                {
                    "round": round_number,
                    "fallback": fallback,
                    "duration_ms": round(duration_ms, 2),
                    "error": _clip(error, 1_000) if error else None,
                    "sufficient": bool(getattr(review, "sufficient", False)),
                    "relevant_source_ids": list(
                        getattr(review, "relevant_source_ids", ())
                    ),
                    "discarded_source_ids": list(
                        getattr(review, "discarded_source_ids", ())
                    ),
                    "missing_information": list(
                        getattr(review, "missing_information", ())
                    ),
                    "follow_up_web_query": getattr(
                        review, "follow_up_web_query", None
                    ),
                    "needs_clarification": bool(
                        getattr(review, "needs_clarification", False)
                    ),
                    "clarifying_question": getattr(
                        review, "clarifying_question", None
                    ),
                    "resolved_scope": getattr(review, "resolved_scope", None),
                    "corrected_resolved_request": getattr(
                        review, "corrected_resolved_request", None
                    ),
                    "correction_reason": getattr(review, "correction_reason", None),
                    "freshness_verified": getattr(review, "freshness_verified", None),
                    "partial_values": [
                        {
                            "source_id": getattr(item, "source_id", ""),
                            "label": getattr(item, "label", ""),
                            "value": getattr(item, "value", None),
                        }
                        for item in getattr(review, "partial_values", ())
                    ],
                    "audited": bool(getattr(review, "audited", True)),
                    "source_catalog": [
                        _source_payload(source, f"W{index}")
                        for index, source in enumerate(web_sources, start=1)
                    ],
                }
            )
            self.timings_ms["review_total"] = round(
                sum(float(item["duration_ms"]) for item in self.reviews), 2
            )

    def record_model_call(
        self,
        stage: str,
        result: object,
        *,
        duration_ms: float | None = None,
    ) -> None:
        """Record completion metadata only, never hidden reasoning content."""

        with self._lock:
            call = {
                "stage": stage,
                "model": getattr(result, "model", None),
                "endpoint": getattr(result, "endpoint", None),
                "reasoning_effort": getattr(result, "reasoning_effort", None),
                "finish_reason": getattr(result, "finish_reason", None),
                "prompt_tokens": getattr(result, "prompt_tokens", None),
                "completion_tokens": getattr(result, "completion_tokens", None),
                "reasoning_tokens": getattr(result, "reasoning_tokens", None),
                "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
                "recovery_reason": (
                    _clip(str(getattr(result, "recovery_reason")), 1_000)
                    if getattr(result, "recovery_reason", None)
                    else None
                ),
            }
            self.model_calls.append(call)

    def record_prompt_metadata(
        self,
        system_prompt: str,
        messages: Sequence[object],
        *,
        template_version: str,
    ) -> None:
        """Record reproducible prompt metadata without duplicating private prompt text."""

        message_characters = sum(
            len(str(getattr(message, "content", ""))) for message in messages
        )
        with self._lock:
            self.prompt_metadata = {
                "template_version": template_version,
                "system_prompt_sha256": hashlib.sha256(
                    system_prompt.encode("utf-8")
                ).hexdigest(),
                "system_prompt_characters": len(system_prompt),
                "message_count": len(messages),
                "message_characters": message_characters,
                "full_prompt_recorded": False,
            }

    def record_contract(self, contract: object) -> None:
        with self._lock:
            self.contract = {
                "resolved_request": getattr(contract, "resolved_request", ""),
                "objective": getattr(contract, "objective", ""),
                "entities": list(getattr(contract, "entities", ())),
                "constraints": list(getattr(contract, "constraints", ())),
                "ambiguities": list(getattr(contract, "ambiguities", ())),
                "missing_for_core": list(getattr(contract, "missing_for_core", ())),
                "missing_for_precision": list(
                    getattr(contract, "missing_for_precision", ())
                ),
                "evidence_policy": getattr(contract, "evidence_policy", None),
                "resolved_scope": getattr(contract, "resolved_scope", None),
                "corrected": bool(getattr(contract, "corrected", False)),
                "correction_reason": getattr(contract, "correction_reason", None),
                "audited": bool(getattr(contract, "audited", False)),
            }

    def record_guard(self, event: str, detail: str) -> None:
        with self._lock:
            self.guard_events.append(
                {"event": _clip(event, 200), "detail": _clip(detail, 1_000)}
            )

    def set_timing(self, stage: str, duration_ms: float) -> None:
        with self._lock:
            self.timings_ms[stage] = round(duration_ms, 2)

    def complete(self, answer: str) -> None:
        with self._lock:
            self.status = "completed"
            self.final_answer = _clip(answer, MAX_FINAL_ANSWER_CHARACTERS)
            self.error = None
            self.completed_at = _utc_now()

    def fail(self, error: str) -> None:
        with self._lock:
            self.status = "error"
            self.error = _clip(error, 2_000)
            self.completed_at = _utc_now()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "trace_id": self.trace_id,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
                "status": self.status,
                "question": self.question,
                "sport": self.sport,
                "requested_mode": self.requested_mode,
                "model": self.model,
                "plan": self.plan,
                "plan_fallback": self.plan_fallback,
                "plan_error": self.plan_error,
                "contract": dict(self.contract) if self.contract else None,
                "local_evidence": list(self.local_evidence),
                "searches": list(self.searches),
                "reviews": list(self.reviews),
                "model_calls": list(self.model_calls),
                "guard_events": list(self.guard_events),
                "prompt_metadata": dict(self.prompt_metadata) if self.prompt_metadata else None,
                "timings_ms": dict(self.timings_ms),
                "final_answer": self.final_answer,
                "error": self.error,
                "privacy": {
                    "persistence": "memory_only",
                    "max_traces": MAX_TRACES,
                    "hidden_chain_of_thought_recorded": False,
                    "credentials_recorded": False,
                    "full_prompt_recorded": False,
                },
            }


class DiagnosticTraceStore:
    def __init__(self, max_traces: int = MAX_TRACES) -> None:
        self.max_traces = max(1, max_traces)
        self._traces: OrderedDict[str, DiagnosticTrace] = OrderedDict()
        self._lock = RLock()

    def start(self, *, question: str, sport: str, requested_mode: str) -> DiagnosticTrace:
        trace = DiagnosticTrace(
            trace_id=f"orion-{uuid4().hex[:12]}",
            created_at=_utc_now(),
            question=_clip(question, MAX_QUESTION_CHARACTERS),
            sport=sport,
            requested_mode=requested_mode,
        )
        with self._lock:
            self._traces[trace.trace_id] = trace
            while len(self._traces) > self.max_traces:
                self._traces.popitem(last=False)
        return trace

    def get(self, trace_id: str) -> DiagnosticTrace | None:
        with self._lock:
            return self._traces.get(trace_id)

    def latest(self) -> DiagnosticTrace | None:
        with self._lock:
            if not self._traces:
                return None
            return next(reversed(self._traces.values()))

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


diagnostic_traces = DiagnosticTraceStore()
